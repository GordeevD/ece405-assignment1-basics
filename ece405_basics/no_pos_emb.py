from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

from .adamw import AdamW
from .checkpointing import save_checkpoint
from .cross_entropy import cross_entropy
from .data_loading import get_batch
from .embedding import Embedding
from .experiment_log import ExperimentLogger
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import get_lr_cosine_schedule
from .linear import Linear
from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU
from .rmsnorm import RMSNorm
from .training_together import (
	estimate_loss,
	load_token_memmap,
	parse_numpy_dtype,
	parse_torch_dtype,
	resolve_device,
	update_optimizer_lr,
)
from .transformer_lm import TransformerLM


def _iso_stamp() -> str:
	return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _lr_slug(lr: float) -> str:
	return f"{lr:.0e}".replace("+", "")


def _min_required_tokens(context_length: int) -> int:
	return context_length + 1


class NoPETransformerBlock(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		d_ff: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()

		self.attn = MultiHeadSelfAttention(
			d_model=d_model,
			num_heads=num_heads,
			rope=None,
			device=device,
			dtype=dtype,
		)
		self.ffn = SwiGLU(
			d_model=d_model,
			d_ff=d_ff,
			device=device,
			dtype=dtype,
		)
		self.ln1 = RMSNorm(
			d_model=d_model,
			device=device,
			dtype=dtype,
		)
		self.ln2 = RMSNorm(
			d_model=d_model,
			device=device,
			dtype=dtype,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.ln1(x), token_positions=None)
		x = x + self.ffn(self.ln2(x))
		return x


class NoPETransformerLM(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		context_length: int,
		d_model: int,
		num_layers: int,
		num_heads: int,
		d_ff: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()

		self.context_length = context_length

		self.token_embeddings = Embedding(
			num_embeddings=vocab_size,
			embedding_dim=d_model,
			device=device,
			dtype=dtype,
		)

		self.layers = nn.ModuleList(
			[
				NoPETransformerBlock(
					d_model=d_model,
					num_heads=num_heads,
					d_ff=d_ff,
					device=device,
					dtype=dtype,
				)
				for _ in range(num_layers)
			]
		)

		self.ln_final = RMSNorm(
			d_model=d_model,
			device=device,
			dtype=dtype,
		)
		self.lm_head = Linear(
			in_features=d_model,
			out_features=vocab_size,
			device=device,
			dtype=dtype,
		)

	def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
		seq_len = in_indices.shape[-1]
		if seq_len > self.context_length:
			raise ValueError("Input sequence length exceeds context_length.")

		x = self.token_embeddings(in_indices)
		for layer in self.layers:
			x = layer(x)

		x = self.ln_final(x)
		return self.lm_head(x)


@dataclass
class RunResult:
	model_kind: str
	run_dir: Path
	checkpoint_path: Path
	best_val_loss: float | None
	final_val_loss: float | None
	best_val_step: int | None


def _analyze_run_dir(run_dir: Path) -> float | None:
	metrics_path = run_dir / "metrics.jsonl"
	if not metrics_path.exists():
		return None

	final_val_loss: float | None = None
	with metrics_path.open("r", encoding="utf-8") as fp:
		for line in fp:
			record = json.loads(line)
			loss = record.get("loss")
			if record.get("split") == "val" and isinstance(loss, (int, float)):
				final_val_loss = float(loss)

	return final_val_loss


def _build_model(
	*,
	model_kind: str,
	vocab_size: int,
	context_length: int,
	d_model: int,
	num_layers: int,
	num_heads: int,
	d_ff: int,
	rope_theta: float,
	device: torch.device,
	torch_dtype: torch.dtype,
) -> nn.Module:
	if model_kind == "rope":
		return TransformerLM(
			vocab_size=vocab_size,
			context_length=context_length,
			d_model=d_model,
			num_layers=num_layers,
			num_heads=num_heads,
			d_ff=d_ff,
			rope_theta=rope_theta,
			device=device,
			dtype=torch_dtype,
		)

	if model_kind == "nope":
		return NoPETransformerLM(
			vocab_size=vocab_size,
			context_length=context_length,
			d_model=d_model,
			num_layers=num_layers,
			num_heads=num_heads,
			d_ff=d_ff,
			device=device,
			dtype=torch_dtype,
		)

	raise ValueError(f"Unsupported model_kind: {model_kind}")


def _train_single_model(
	*,
	args: argparse.Namespace,
	train_tokens: np.ndarray,
	val_tokens: np.ndarray,
	device: torch.device,
	torch_dtype: torch.dtype,
	model_kind: str,
	run_name: str,
	checkpoint_path: Path,
	max_iters: int,
	warmup_iters: int,
) -> RunResult:
	device_str = str(device)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	model = _build_model(
		model_kind=model_kind,
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
		device=device,
		torch_dtype=torch_dtype,
	)
	optimizer = AdamW(
		model.parameters(),
		lr=args.lr_max,
		betas=(args.beta1, args.beta2),
		eps=args.eps,
		weight_decay=args.weight_decay,
	)

	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	experiment_logger = ExperimentLogger(
		experiment_dir=args.experiment_dir,
		run_name=run_name,
		config={
			**vars(args),
			"ablation": "rope_vs_nope",
			"model_kind": model_kind,
			"max_iters": max_iters,
			"warmup_iters": warmup_iters,
		},
	)

	print(f"\n=== {model_kind} run ({run_name}) ===")
	print(f"Experiment logs: {experiment_logger.run_dir}")

	model.train()
	t0 = perf_counter()

	try:
		for iteration in range(1, max_iters + 1):
			current_lr = get_lr_cosine_schedule(
				t=iteration,
				alpha_max=args.lr_max,
				alpha_min=args.lr_min,
				T_w=max(warmup_iters, 1),
				T_c=max(max_iters, warmup_iters + 1),
			)
			update_optimizer_lr(optimizer, current_lr)

			xb, yb = get_batch(train_tokens, args.batch_size, args.context_length, device_str)
			logits = model(xb)
			loss = cross_entropy(logits, yb)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			if math.isfinite(args.max_grad_norm) and args.max_grad_norm > 0:
				gradient_clipping(model.parameters(), args.max_grad_norm)
			optimizer.step()

			if iteration % args.log_interval == 0:
				tokens_processed = iteration * args.batch_size * args.context_length
				elapsed = max(perf_counter() - t0, 1e-9)
				toks_per_sec = tokens_processed / elapsed
				train_loss = float(loss.detach().item())
				print(
					f"iter={iteration:6d} [{model_kind}] train_loss={train_loss:.4f} "
					f"lr={current_lr:.3e} tok/s={toks_per_sec:.1f} device={device_str}"
				)
				experiment_logger.log_metrics(
					gradient_step=iteration,
					metrics={
						"loss": train_loss,
						"lr": current_lr,
						"tokens_per_sec": toks_per_sec,
					},
					split="train",
				)

			if iteration % args.eval_interval == 0 or iteration == max_iters:
				metrics = estimate_loss(
					model=model,
					train_tokens=train_tokens,
					val_tokens=val_tokens,
					batch_size=args.batch_size,
					context_length=args.context_length,
					device=device_str,
					eval_batches=args.eval_batches,
				)
				print(
					f"iter={iteration:6d} [{model_kind}] eval "
					f"train_loss={metrics['train']:.4f} val_loss={metrics['val']:.4f}"
				)
				experiment_logger.log_metrics(
					gradient_step=iteration,
					metrics={"loss": metrics["train"]},
					split="train_eval",
				)
				experiment_logger.log_metrics(
					gradient_step=iteration,
					metrics={"loss": metrics["val"]},
					split="val",
				)

			if iteration % args.checkpoint_interval == 0 or iteration == max_iters:
				save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=checkpoint_path)
				print(f"Saved checkpoint to {checkpoint_path} at iteration {iteration}")

		save_checkpoint(model=model, optimizer=optimizer, iteration=max_iters, out=checkpoint_path)
	finally:
		experiment_logger.close()

	run_dir = experiment_logger.run_dir
	summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
	best_val = summary.get("best_val_loss")
	best_step = summary.get("best_val_step")
	final_val = _analyze_run_dir(run_dir)

	return RunResult(
		model_kind=model_kind,
		run_dir=run_dir,
		checkpoint_path=checkpoint_path,
		best_val_loss=float(best_val) if isinstance(best_val, (int, float)) else None,
		final_val_loss=final_val,
		best_val_step=int(best_step) if isinstance(best_step, int) else None,
	)


def _load_curve_rows(run_dir: Path, model_kind: str) -> list[dict[str, str]]:
	curve_path = run_dir / "loss_curves.csv"
	if not curve_path.exists():
		return []

	rows: list[dict[str, str]] = []
	with curve_path.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			rows.append(
				{
					"model": model_kind,
					"gradient_step": str(row.get("gradient_step", "")),
					"wallclock_time_s": str(row.get("wallclock_time_s", "")),
					"split": str(row.get("split", "")),
					"loss": str(row.get("loss", "")),
				}
			)
	return rows


def _write_outputs(args: argparse.Namespace, rope_result: RunResult, nope_result: RunResult) -> tuple[Path, Path]:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	stamp = _iso_stamp()

	curves_csv = output_dir / f"no_pos_emb_ablation_curves_{stamp}.csv"
	report_md = output_dir / f"no_pos_emb_ablation_report_{stamp}.md"

	rows = _load_curve_rows(rope_result.run_dir, rope_result.model_kind)
	rows.extend(_load_curve_rows(nope_result.run_dir, nope_result.model_kind))

	with curves_csv.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=["model", "gradient_step", "wallclock_time_s", "split", "loss"],
		)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)

	lines = [
		"# Position-Embedding Ablation (RoPE vs NoPE)",
		"",
		"## Learning-curve files",
		"",
		f"- RoPE run curve: `{rope_result.run_dir / 'loss_curves.csv'}`",
		f"- NoPE run curve: `{nope_result.run_dir / 'loss_curves.csv'}`",
		f"- Combined two-curve CSV: `{curves_csv}`",
		"",
		"## Results summary",
		"",
		f"- RoPE best val loss: {'N/A' if rope_result.best_val_loss is None else f'{rope_result.best_val_loss:.6f}'}",
		f"- RoPE final val loss: {'N/A' if rope_result.final_val_loss is None else f'{rope_result.final_val_loss:.6f}'}",
		f"- RoPE best step: {'N/A' if rope_result.best_val_step is None else rope_result.best_val_step}",
		f"- NoPE best val loss: {'N/A' if nope_result.best_val_loss is None else f'{nope_result.best_val_loss:.6f}'}",
		f"- NoPE final val loss: {'N/A' if nope_result.final_val_loss is None else f'{nope_result.final_val_loss:.6f}'}",
		f"- NoPE best step: {'N/A' if nope_result.best_val_step is None else nope_result.best_val_step}",
	]

	if rope_result.final_val_loss is not None and nope_result.final_val_loss is not None:
		delta = nope_result.final_val_loss - rope_result.final_val_loss
		lines.append(f"- Final val-loss delta (NoPE - RoPE): {delta:+.6f}")

	lines.extend(["", "## Commentary", ""])
	if rope_result.final_val_loss is not None and nope_result.final_val_loss is not None:
		if nope_result.final_val_loss < rope_result.final_val_loss:
			commentary = [
				"In this run, NoPE outperformed RoPE on final validation loss.",
				"Decoder-only attention with causal masking can still recover useful position information without explicit positional embeddings.",
			]
		elif nope_result.final_val_loss > rope_result.final_val_loss:
			commentary = [
				"In this run, removing positional embeddings degraded validation performance relative to RoPE.",
				"RoPE likely improves inductive bias for learning token order under this optimization budget.",
			]
		else:
			commentary = [
				"In this run, RoPE and NoPE reached essentially identical final validation loss.",
				"Under the tested budget and model size, explicit rotational position information did not materially change the endpoint metric.",
			]
	else:
		commentary = [
			"At least one run did not emit a final validation loss in the logs.",
			"Use the per-run `metrics.jsonl` files to diagnose and rerun if needed.",
		]

	for sentence in commentary:
		lines.append(f"- {sentence}")

	report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return curves_csv, report_md


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ablate RoPE by comparing TransformerLM against NoPE.")

	parser.add_argument("--train-data", type=str, default="experiments_tmp/train_tokens.npy")
	parser.add_argument("--val-data", type=str, default="experiments_tmp/val_tokens.npy")
	parser.add_argument("--data-dtype", type=str, default="uint16")

	parser.add_argument("--vocab-size", type=int, default=512)
	parser.add_argument("--context-length", type=int, default=64)
	parser.add_argument("--d-model", type=int, default=64)
	parser.add_argument("--num-layers", type=int, default=2)
	parser.add_argument("--num-heads", type=int, default=4)
	parser.add_argument("--d-ff", type=int, default=128)
	parser.add_argument("--rope-theta", type=float, default=10_000.0)

	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--token-budget", type=int, default=33_000_000)
	parser.add_argument("--lr-max", type=float, default=1e-3)
	parser.add_argument("--lr-min", type=float, default=1e-4)
	parser.add_argument("--warmup-frac", type=float, default=0.03)

	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.95)
	parser.add_argument("--eps", type=float, default=1e-8)
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)

	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--dtype", type=str, default="float32")
	parser.add_argument("--seed", type=int, default=405)
	parser.add_argument("--log-interval", type=int, default=100)
	parser.add_argument("--eval-interval", type=int, default=250)
	parser.add_argument("--eval-batches", type=int, default=20)
	parser.add_argument("--checkpoint-interval", type=int, default=250)

	parser.add_argument("--experiment-dir", type=str, default="experiments")
	parser.add_argument("--checkpoint-dir", type=str, default="experiments_tmp")
	parser.add_argument("--output-dir", type=str, default="experiments")

	args = parser.parse_args()
	if args.token_budget <= 0:
		raise ValueError("--token-budget must be positive")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be positive")
	if args.context_length <= 0:
		raise ValueError("--context-length must be positive")
	if args.eval_interval <= 0:
		raise ValueError("--eval-interval must be > 0")
	if args.log_interval <= 0:
		raise ValueError("--log-interval must be > 0")
	if not (0.0 <= args.warmup_frac < 1.0):
		raise ValueError("--warmup-frac must be in [0, 1)")
	return args


def main() -> None:
	args = parse_args()

	np_dtype = parse_numpy_dtype(args.data_dtype)
	torch_dtype = parse_torch_dtype(args.dtype)
	device = resolve_device(args.device)

	train_tokens = load_token_memmap(args.train_data, np_dtype)
	val_tokens = load_token_memmap(args.val_data, np_dtype)

	min_tokens = _min_required_tokens(args.context_length)
	if len(train_tokens) < min_tokens:
		raise ValueError(
			f"train dataset too small: {len(train_tokens)} tokens, need at least {min_tokens}"
		)
	if len(val_tokens) < min_tokens:
		raise ValueError(f"val dataset too small: {len(val_tokens)} tokens, need at least {min_tokens}")

	denom = args.batch_size * args.context_length
	max_iters = max(1, args.token_budget // denom)
	warmup_iters = int(args.warmup_frac * max_iters)
	stamp = _iso_stamp()
	lr_tag = _lr_slug(args.lr_max)

	print("=== Position-Embedding Ablation (RoPE vs NoPE) ===")
	print(f"device={device}")
	print(f"token_budget={args.token_budget} -> max_iters={max_iters}")

	ckpt_dir = Path(args.checkpoint_dir)
	rope_ckpt = ckpt_dir / f"ckpt_rope_lr_{lr_tag}_{stamp}.pt"
	nope_ckpt = ckpt_dir / f"ckpt_nope_lr_{lr_tag}_{stamp}.pt"

	rope_result = _train_single_model(
		args=args,
		train_tokens=train_tokens,
		val_tokens=val_tokens,
		device=device,
		torch_dtype=torch_dtype,
		model_kind="rope",
		run_name=f"pos-ablation-rope-lr-{lr_tag}",
		checkpoint_path=rope_ckpt,
		max_iters=max_iters,
		warmup_iters=warmup_iters,
	)

	nope_result = _train_single_model(
		args=args,
		train_tokens=train_tokens,
		val_tokens=val_tokens,
		device=device,
		torch_dtype=torch_dtype,
		model_kind="nope",
		run_name=f"pos-ablation-nope-lr-{lr_tag}",
		checkpoint_path=nope_ckpt,
		max_iters=max_iters,
		warmup_iters=warmup_iters,
	)

	curves_csv, report_md = _write_outputs(args, rope_result, nope_result)

	print("\n=== Completed RoPE vs NoPE ablation ===")
	print(f"RoPE run dir: {rope_result.run_dir}")
	print(f"NoPE run dir: {nope_result.run_dir}")
	print(f"Combined curves: {curves_csv}")
	print(f"Report: {report_md}")


if __name__ == "__main__":
	main()
