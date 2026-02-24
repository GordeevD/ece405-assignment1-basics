from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adamw import AdamW
from .cross_entropy import cross_entropy
from .data_loading import get_batch
from .experiment_log import ExperimentLogger
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import get_lr_cosine_schedule
from .training_together import load_token_memmap, parse_numpy_dtype, parse_torch_dtype, resolve_device
from .transformer_lm import TransformerLM


def _timestamp_utc() -> str:
	return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class TrainConfig:
	vocab_size: int = 10_000
	context_length: int = 256
	d_model: int = 512
	d_ff: int = 1_344
	rope_theta: float = 10_000.0
	num_layers: int = 4
	num_heads: int = 16

	total_tokens: int = 40_000_000
	batch_size: int = 64

	lr_min_ratio: float = 0.1
	warmup_ratio: float = 0.03
	beta1: float = 0.9
	beta2: float = 0.95
	eps: float = 1e-8
	weight_decay: float = 0.1
	max_grad_norm: float = 1.0

	eval_interval: int = 100
	eval_batches: int = 20
	log_interval: int = 20

	device: str = "auto"
	dtype: str = "float32"
	seed: int = 405

	target_val_loss: float = 2.0
	raw_train_corpus: str = "tests/fixtures/TinyStoriesV2-GPT4-train.txt"
	raw_val_corpus: str = "tests/fixtures/TinyStoriesV2-GPT4-valid.txt"

	train_data: str = "experiments_tmp/train_tokens.npy"
	val_data: str = "experiments_tmp/val_tokens.npy"
	data_dtype: str = "uint16"

	experiment_dir: str = "experiments"
	run_prefix: str = "lr-sweep"


@dataclass
class SweepResult:
	run_name: str
	lr_max: float
	status: str
	max_iters: int
	final_train_loss: float | None
	final_val_loss: float | None
	best_val_loss: float | None
	best_val_step: int | None
	run_dir: str


def parse_lr_list(value: str) -> list[float]:
	parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
	if not parts:
		raise ValueError("learning-rate list cannot be empty")
	return [float(chunk) for chunk in parts]


def maybe_disable_tf32_for_mps(device: torch.device) -> None:
	if device.type == "mps":
	#	torch.set_float32_matmul_precision("highest")
		return
	if device.type == "cuda":
		torch.set_float32_matmul_precision("high")
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="TinyStories learning-rate sweep harness (MPS-friendly).")

	parser.add_argument("--learning-rates", type=str, default="1e-4,3e-4,1e-3,2e-3")
	parser.add_argument("--total-tokens", type=int, default=40_000_000)
	parser.add_argument("--target-val-loss", type=float, default=2.0)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--context-length", type=int, default=256)
	parser.add_argument("--warmup-ratio", type=float, default=0.03)
	parser.add_argument("--lr-min-ratio", type=float, default=0.1)

	parser.add_argument("--vocab-size", type=int, default=10_000)
	parser.add_argument("--d-model", type=int, default=512)
	parser.add_argument("--d-ff", type=int, default=1_344)
	parser.add_argument("--num-layers", type=int, default=4)
	parser.add_argument("--num-heads", type=int, default=16)
	parser.add_argument("--rope-theta", type=float, default=10_000.0)

	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.95)
	parser.add_argument("--eps", type=float, default=1e-8)
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)

	parser.add_argument("--eval-interval", type=int, default=100)
	parser.add_argument("--eval-batches", type=int, default=20)
	parser.add_argument("--log-interval", type=int, default=20)

	parser.add_argument("--train-data", type=str, default="experiments_tmp/train_tokens.npy")
	parser.add_argument("--val-data", type=str, default="experiments_tmp/val_tokens.npy")
	parser.add_argument("--data-dtype", type=str, default="uint16")
	parser.add_argument("--raw-train-corpus", type=str, default="tests/fixtures/TinyStoriesV2-GPT4-train.txt")
	parser.add_argument("--raw-val-corpus", type=str, default="tests/fixtures/TinyStoriesV2-GPT4-valid.txt")

	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--dtype", type=str, default="float32")
	parser.add_argument("--seed", type=int, default=405)

	parser.add_argument("--experiment-dir", type=str, default="experiments")
	parser.add_argument("--run-prefix", type=str, default="lr-sweep")

	parser.add_argument(
		"--divergence-loss-threshold",
		type=float,
		default=20.0,
		help="Training run is marked divergent if train loss exceeds this threshold after warmup.",
	)

	return parser


@torch.no_grad()
def estimate_loss(
	model: TransformerLM,
	train_tokens: np.ndarray,
	val_tokens: np.ndarray,
	batch_size: int,
	context_length: int,
	device: str,
	eval_batches: int,
) -> dict[str, float]:
	model.eval()
	out: dict[str, float] = {}
	for split, tokens in (("train", train_tokens), ("val", val_tokens)):
		losses = torch.empty(eval_batches, device="cpu")
		for idx in range(eval_batches):
			xb, yb = get_batch(tokens, batch_size=batch_size, context_length=context_length, device=device)
			logits = model(xb)
			losses[idx] = cross_entropy(logits, yb).detach().cpu()
		out[split] = float(losses.mean().item())
	model.train()
	return out


def compute_max_iters(total_tokens: int, batch_size: int, context_length: int) -> int:
	tokens_per_step = batch_size * context_length
	return max(1, total_tokens // tokens_per_step)


def run_single_lr(
	config: TrainConfig,
	lr_max: float,
	train_tokens: np.ndarray,
	val_tokens: np.ndarray,
	device: torch.device,
	torch_dtype: torch.dtype,
	np_dtype: np.dtype,
	divergence_loss_threshold: float,
) -> SweepResult:
	if train_tokens.dtype != np_dtype:
		train_tokens = train_tokens.astype(np_dtype)
	if val_tokens.dtype != np_dtype:
		val_tokens = val_tokens.astype(np_dtype)

	torch.manual_seed(config.seed)
	np.random.seed(config.seed)

	max_iters = compute_max_iters(config.total_tokens, config.batch_size, config.context_length)
	warmup_iters = max(1, int(max_iters * config.warmup_ratio))
	cosine_iters = max_iters
	lr_min = lr_max * config.lr_min_ratio

	run_name = f"{config.run_prefix}-{lr_max:.0e}"
	logger = ExperimentLogger(
		experiment_dir=config.experiment_dir,
		run_name=run_name,
		config={
			**asdict(config),
			"lr_max": lr_max,
			"lr_min": lr_min,
			"warmup_iters": warmup_iters,
			"cosine_iters": cosine_iters,
			"max_iters": max_iters,
		},
	)

	model = TransformerLM(
		vocab_size=config.vocab_size,
		context_length=config.context_length,
		d_model=config.d_model,
		num_layers=config.num_layers,
		num_heads=config.num_heads,
		d_ff=config.d_ff,
		rope_theta=config.rope_theta,
		device=device,
		dtype=torch_dtype,
	)
	optimizer = AdamW(
		model.parameters(),
		lr=lr_max,
		betas=(config.beta1, config.beta2),
		eps=config.eps,
		weight_decay=config.weight_decay,
	)

	status = "completed"
	final_train_loss: float | None = None
	final_val_loss: float | None = None

	try:
		for step in range(1, max_iters + 1):
			lr = get_lr_cosine_schedule(
				t=step,
				alpha_max=lr_max,
				alpha_min=lr_min,
				T_w=warmup_iters,
				T_c=cosine_iters,
			)
			for group in optimizer.param_groups:
				group["lr"] = lr

			xb, yb = get_batch(train_tokens, config.batch_size, config.context_length, str(device))
			logits = model(xb)
			loss = cross_entropy(logits, yb)

			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			if math.isfinite(config.max_grad_norm) and config.max_grad_norm > 0:
				gradient_clipping(model.parameters(), config.max_grad_norm)
			optimizer.step()

			train_loss = float(loss.detach().item())
			final_train_loss = train_loss

			if step % config.log_interval == 0 or step == 1:
				logger.log_metrics(
					gradient_step=step,
					metrics={"loss": train_loss, "lr": lr},
					split="train",
				)

			if not math.isfinite(train_loss):
				status = "diverged_non_finite"
				break
			if step > warmup_iters and train_loss >= divergence_loss_threshold:
				status = "diverged_high_loss"
				break

			if step % config.eval_interval == 0 or step == max_iters:
				metrics = estimate_loss(
					model=model,
					train_tokens=train_tokens,
					val_tokens=val_tokens,
					batch_size=config.batch_size,
					context_length=config.context_length,
					device=str(device),
					eval_batches=config.eval_batches,
				)
				final_val_loss = metrics["val"]
				logger.log_metrics(
					gradient_step=step,
					metrics={"loss": metrics["train"]},
					split="train_eval",
				)
				logger.log_metrics(
					gradient_step=step,
					metrics={"loss": metrics["val"]},
					split="val",
				)

	finally:
		logger.close()

	summary = json.loads((logger.run_dir / "summary.json").read_text(encoding="utf-8"))
	best_val_loss = summary.get("best_val_loss")
	best_val_step = summary.get("best_val_step")

	return SweepResult(
		run_name=run_name,
		lr_max=lr_max,
		status=status,
		max_iters=max_iters,
		final_train_loss=final_train_loss,
		final_val_loss=final_val_loss,
		best_val_loss=best_val_loss,
		best_val_step=best_val_step,
		run_dir=str(logger.run_dir),
	)


def write_sweep_summary(
	results: list[SweepResult],
	config: TrainConfig,
	learning_rates: list[float],
) -> tuple[Path, Path]:
	experiment_root = Path(config.experiment_dir)
	experiment_root.mkdir(parents=True, exist_ok=True)
	ts = _timestamp_utc()
	csv_path = experiment_root / f"lr_sweep_summary_{ts}.csv"
	md_path = experiment_root / f"lr_sweep_summary_{ts}.md"

	with csv_path.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=[
				"lr_max",
				"status",
				"max_iters",
				"final_train_loss",
				"final_val_loss",
				"best_val_loss",
				"best_val_step",
				"run_dir",
			],
		)
		writer.writeheader()
		for r in results:
			writer.writerow(
				{
					"lr_max": r.lr_max,
					"status": r.status,
					"max_iters": r.max_iters,
					"final_train_loss": r.final_train_loss,
					"final_val_loss": r.final_val_loss,
					"best_val_loss": r.best_val_loss,
					"best_val_step": r.best_val_step,
					"run_dir": r.run_dir,
				}
			)

	best_non_divergent = [r for r in results if r.status == "completed" and r.best_val_loss is not None]
	best_non_divergent.sort(key=lambda r: float(r.best_val_loss))
	best_run = best_non_divergent[0] if best_non_divergent else None

	divergent = [r for r in results if r.status.startswith("diverged")]
	sorted_lrs = sorted(learning_rates)
	edge_hint = "N/A"
	if divergent:
		first_divergent = min(divergent, key=lambda r: r.lr_max)
		edge_hint = f"First divergent LR observed at {first_divergent.lr_max:.2e}"

	strategy_text = (
		"Coarse-to-fine around the stability edge: run increasing learning rates to find divergence, "
		"then select the largest stable learning rate with best validation loss."
	)

	lines: list[str] = []
	lines.append("# TinyStories learning-rate sweep summary")
	lines.append("")
	lines.append("## Search strategy")
	lines.append(strategy_text)
	lines.append("")
	lines.append("## Sweep settings")
	lines.append(f"- learning_rates: {', '.join(f'{lr:.2e}' for lr in sorted_lrs)}")
	lines.append(f"- total_tokens: {config.total_tokens}")
	lines.append(f"- context_length: {config.context_length}")
	lines.append(f"- batch_size: {config.batch_size}")
	lines.append(f"- target_val_loss: {config.target_val_loss}")
	lines.append("")
	lines.append("## Results")
	lines.append("")
	lines.append("| lr_max | status | best_val_loss | best_val_step | run_dir |")
	lines.append("|---:|---|---:|---:|---|")
	for r in sorted(results, key=lambda item: item.lr_max):
		best_loss_text = "N/A" if r.best_val_loss is None else f"{r.best_val_loss:.4f}"
		best_step_text = "N/A" if r.best_val_step is None else str(r.best_val_step)
		lines.append(f"| {r.lr_max:.2e} | {r.status} | {best_loss_text} | {best_step_text} | {r.run_dir} |")
	lines.append("")
	lines.append("## Analysis")
	lines.append(f"- Edge-of-stability observation: {edge_hint}.")
	if best_run is None:
		lines.append("- No stable run finished with a measurable validation loss.")
	else:
		lines.append(
			f"- Best stable run: lr={best_run.lr_max:.2e} with best_val_loss={best_run.best_val_loss:.4f} "
			f"at step {best_run.best_val_step}."
		)
		if best_run.best_val_loss is not None and best_run.best_val_loss <= config.target_val_loss:
			lines.append(
				f"- Target met: best validation loss <= {config.target_val_loss:.2f}."
			)
		else:
			lines.append(
				f"- Target not met: best validation loss is above {config.target_val_loss:.2f}."
			)
	lines.append("")
	lines.append("## Learning curves")
	lines.append("Each run contains `loss_curves.csv` under its run directory for plotting per-run learning curves.")

	md_path.write_text("\n".join(lines), encoding="utf-8")
	return csv_path, md_path


def main() -> None:
	args = build_parser().parse_args()
	learning_rates = parse_lr_list(args.learning_rates)

	config = TrainConfig(
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		total_tokens=args.total_tokens,
		batch_size=args.batch_size,
		lr_min_ratio=args.lr_min_ratio,
		warmup_ratio=args.warmup_ratio,
		beta1=args.beta1,
		beta2=args.beta2,
		eps=args.eps,
		weight_decay=args.weight_decay,
		max_grad_norm=args.max_grad_norm,
		eval_interval=args.eval_interval,
		eval_batches=args.eval_batches,
		log_interval=args.log_interval,
		device=args.device,
		dtype=args.dtype,
		seed=args.seed,
		target_val_loss=args.target_val_loss,
		raw_train_corpus=args.raw_train_corpus,
		raw_val_corpus=args.raw_val_corpus,
		train_data=args.train_data,
		val_data=args.val_data,
		data_dtype=args.data_dtype,
		experiment_dir=args.experiment_dir,
		run_prefix=args.run_prefix,
	)

	device = resolve_device(config.device)
	maybe_disable_tf32_for_mps(device)
	torch_dtype = parse_torch_dtype(config.dtype)
	np_dtype = parse_numpy_dtype(config.data_dtype)

	train_tokens = load_token_memmap(config.train_data, np_dtype)
	val_tokens = load_token_memmap(config.val_data, np_dtype)

	min_required = config.context_length + 1
	if len(train_tokens) < min_required:
		raise ValueError(f"train dataset too small for context_length={config.context_length}")
	if len(val_tokens) < min_required:
		raise ValueError(f"val dataset too small for context_length={config.context_length}")

	results: list[SweepResult] = []
	for lr in learning_rates:
		print(f"\n=== Running LR sweep entry: lr_max={lr:.2e} on device={device} ===")
		result = run_single_lr(
			config=config,
			lr_max=lr,
			train_tokens=train_tokens,
			val_tokens=val_tokens,
			device=device,
			torch_dtype=torch_dtype,
			np_dtype=np_dtype,
			divergence_loss_threshold=args.divergence_loss_threshold,
		)
		results.append(result)
		print(
			f"Completed lr={lr:.2e}: status={result.status}, "
			f"best_val_loss={result.best_val_loss}, run_dir={result.run_dir}"
		)

	csv_path, md_path = write_sweep_summary(results, config, learning_rates)
	print("\nSweep complete.")
	print(f"Summary CSV: {csv_path}")
	print(f"Summary MD : {md_path}")


if __name__ == "__main__":
	main()
