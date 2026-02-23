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

from .adamw import AdamW
from .checkpointing import save_checkpoint
from .cross_entropy import cross_entropy
from .data_loading import get_batch
from .experiment_log import ExperimentLogger
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import get_lr_cosine_schedule
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


def _parse_lr_list(raw: str) -> list[float]:
	values: list[float] = []
	for token in raw.split(","):
		candidate = token.strip()
		if not candidate:
			continue
		values.append(float(candidate))
	if not values:
		raise ValueError("No learning rates provided.")
	return values


@dataclass
class RunResult:
	lr: float
	run_dir: Path
	checkpoint_path: Path
	best_val_loss: float | None
	final_val_loss: float | None
	best_val_step: int | None
	diverged: bool
	notes: str = ""


def _replace_rmsnorms_with_identity(module: torch.nn.Module) -> int:
	replaced = 0
	for child_name, child in list(module.named_children()):
		if isinstance(child, RMSNorm):
			setattr(module, child_name, torch.nn.Identity())
			replaced += 1
		else:
			replaced += _replace_rmsnorms_with_identity(child)
	return replaced


def _analyze_run_dir(run_dir: Path) -> tuple[float | None, bool]:
	metrics_path = run_dir / "metrics.jsonl"
	if not metrics_path.exists():
		return None, True

	diverged = False
	final_val_loss: float | None = None
	with metrics_path.open("r", encoding="utf-8") as fp:
		for line in fp:
			record = json.loads(line)
			loss = record.get("loss")
			if isinstance(loss, (int, float)) and not math.isfinite(float(loss)):
				diverged = True
			if record.get("split") == "val" and isinstance(loss, (int, float)):
				final_val_loss = float(loss)

	if final_val_loss is None:
		diverged = True
	return final_val_loss, diverged


def _min_required_tokens(context_length: int) -> int:
	return context_length + 1


def _train_single_lr(
	*,
	args: argparse.Namespace,
	train_tokens: np.ndarray,
	val_tokens: np.ndarray,
	device: torch.device,
	torch_dtype: torch.dtype,
	lr: float,
	run_name: str,
	checkpoint_path: Path,
	max_iters: int,
	warmup_iters: int,
) -> RunResult:
	device_str = str(device)
	rng_seed = args.seed
	torch.manual_seed(rng_seed)
	np.random.seed(rng_seed)

	model = TransformerLM(
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
		device=device,
		dtype=torch_dtype,
	)
	replaced = _replace_rmsnorms_with_identity(model)
	if replaced == 0:
		raise RuntimeError("No RMSNorm layers were found to replace.")

	optimizer = AdamW(
		model.parameters(),
		lr=lr,
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
			"ablation": "remove_all_rmsnorm",
			"rmsnorm_replaced": replaced,
			"lr_max": lr,
			"lr_min": lr * args.lr_min_ratio,
			"max_iters": max_iters,
			"warmup_iters": warmup_iters,
		},
	)

	print(f"\n=== No-RMSNorm run lr={lr:.3e} ({run_name}) ===")
	print(f"Experiment logs: {experiment_logger.run_dir}")

	model.train()
	t0 = perf_counter()

	try:
		for iteration in range(1, max_iters + 1):
			current_lr = get_lr_cosine_schedule(
				t=iteration,
				alpha_max=lr,
				alpha_min=lr * args.lr_min_ratio,
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
					f"iter={iteration:6d} train_loss={train_loss:.4f} lr={current_lr:.3e} "
					f"tok/s={toks_per_sec:.1f} device={device_str}"
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
					f"iter={iteration:6d} eval train_loss={metrics['train']:.4f} "
					f"val_loss={metrics['val']:.4f}"
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
	final_val, metrics_diverged = _analyze_run_dir(run_dir)

	diverged = metrics_diverged
	if isinstance(best_val, (int, float)) and not math.isfinite(float(best_val)):
		diverged = True

	return RunResult(
		lr=lr,
		run_dir=run_dir,
		checkpoint_path=checkpoint_path,
		best_val_loss=float(best_val) if isinstance(best_val, (int, float)) else None,
		final_val_loss=final_val,
		best_val_step=int(best_step) if isinstance(best_step, int) else None,
		diverged=diverged,
	)


def _load_curve_rows(run_dir: Path, lr: float) -> list[dict[str, str]]:
	curve_path = run_dir / "loss_curves.csv"
	if not curve_path.exists():
		return []

	rows: list[dict[str, str]] = []
	with curve_path.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			rows.append(
				{
					"lr": f"{lr:.8g}",
					"gradient_step": str(row.get("gradient_step", "")),
					"wallclock_time_s": str(row.get("wallclock_time_s", "")),
					"split": str(row.get("split", "")),
					"loss": str(row.get("loss", "")),
				}
			)
	return rows


def _write_outputs(args: argparse.Namespace, base_result: RunResult, best_lower: RunResult | None) -> tuple[Path, Path]:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	stamp = _iso_stamp()

	curves_csv = output_dir / f"layer_norm_ablation_curves_{stamp}.csv"
	report_md = output_dir / f"layer_norm_ablation_report_{stamp}.md"

	rows = _load_curve_rows(base_result.run_dir, base_result.lr)
	if best_lower is not None:
		rows.extend(_load_curve_rows(best_lower.run_dir, best_lower.lr))

	with curves_csv.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=["lr", "gradient_step", "wallclock_time_s", "split", "loss"],
		)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)

	lines: list[str] = [
		"# LayerNorm Ablation (Remove RMSNorm)",
		"",
		"## Learning-curve files",
		"",
		f"- Previous optimal LR (no RMSNorm): `{base_result.run_dir / 'loss_curves.csv'}`",
	]
	if best_lower is not None:
		lines.append(f"- Best lower LR (no RMSNorm): `{best_lower.run_dir / 'loss_curves.csv'}`")
	else:
		lines.append("- Best lower LR (no RMSNorm): none stabilized")
	lines.append(f"- Combined two-curve CSV: `{curves_csv}`")

	lines.extend(
		[
			"",
			"## Results summary",
			"",
			f"- Previous optimal LR with RMSNorm: `{args.previous_optimal_lr:.8g}`",
			f"- Previous optimal LR without RMSNorm status: {'diverged' if base_result.diverged else 'stable'}",
			f"- Previous optimal LR final val loss: "
			f"{'N/A' if base_result.final_val_loss is None else f'{base_result.final_val_loss:.6f}'}",
		]
	)

	if best_lower is not None:
		lines.extend(
			[
				f"- Best lower LR without RMSNorm: `{best_lower.lr:.8g}`",
				f"- Best lower LR final val loss: "
				f"{'N/A' if best_lower.final_val_loss is None else f'{best_lower.final_val_loss:.6f}'}",
			]
		)

	commentary: list[str] = []
	if base_result.diverged and best_lower is not None and not best_lower.diverged:
		commentary = [
			"Removing RMSNorm made training unstable at the previous optimal learning rate.",
			"Lowering the learning rate restored stability and produced a finite validation curve.",
			"This indicates RMSNorm improves optimization robustness and allows a more aggressive step size.",
		]
	elif base_result.diverged and (best_lower is None or best_lower.diverged):
		commentary = [
			"Removing RMSNorm caused instability at the previous optimal learning rate.",
			"Within the tested lower learning-rate range, training still did not reliably stabilize.",
			"This suggests RMSNorm is a key stabilizer for this configuration.",
		]
	elif (not base_result.diverged) and best_lower is not None and not best_lower.diverged:
		if (
			base_result.final_val_loss is not None
			and best_lower.final_val_loss is not None
			and best_lower.final_val_loss < base_result.final_val_loss
		):
			commentary = [
				"Training remained stable without RMSNorm at the previous optimal learning rate.",
				"A lower learning rate improved validation loss, indicating less normalization may shift the best LR downward.",
				"RMSNorm still appears beneficial for reaching strong performance at higher learning rates.",
			]
		else:
			commentary = [
				"Training remained stable without RMSNorm at the previous optimal learning rate.",
				"Lower learning rates did not outperform that setting in this sweep.",
				"For this setup, RMSNorm was not required for stability but may still affect convergence speed/quality.",
			]
	else:
		commentary = [
			"Removing RMSNorm changed optimization behavior in this setup.",
			"The provided curves show the direct comparison between the previous optimal LR and the best tested lower LR.",
			"Use final validation loss and stability to decide whether to keep a reduced LR when RMSNorm is removed.",
		]

	lines.extend(["", "## Commentary", ""])
	for sentence in commentary:
		lines.append(f"- {sentence}")

	report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return curves_csv, report_md


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ablate RMSNorm and compare learning-rate stability.")

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
	parser.add_argument("--previous-optimal-lr", type=float, default=1e-3)
	parser.add_argument("--lower-lrs", type=_parse_lr_list, default="5e-4,3e-4,1e-4")
	parser.add_argument("--lr-min-ratio", type=float, default=0.1)
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
	if isinstance(args.lower_lrs, str):
		args.lower_lrs = _parse_lr_list(args.lower_lrs)
	if args.log_interval <= 0 or args.eval_interval <= 0:
		raise ValueError("log_interval and eval_interval must be > 0")
	if args.batch_size <= 0 or args.context_length <= 0:
		raise ValueError("batch_size and context_length must be positive")
	return args


def main() -> None:
	args = parse_args()

	np_dtype = parse_numpy_dtype(args.data_dtype)
	torch_dtype = parse_torch_dtype(args.dtype)
	device = resolve_device(args.device)

	train_tokens = load_token_memmap(args.train_data, np_dtype)
	val_tokens = load_token_memmap(args.val_data, np_dtype)
	min_required = _min_required_tokens(args.context_length)
	if len(train_tokens) < min_required:
		raise ValueError(f"train dataset too small: {len(train_tokens)} tokens, need at least {min_required}")
	if len(val_tokens) < min_required:
		raise ValueError(f"val dataset too small: {len(val_tokens)} tokens, need at least {min_required}")

	tokens_per_iter = args.batch_size * args.context_length
	max_iters = args.token_budget // tokens_per_iter
	if max_iters <= 0:
		raise ValueError(
			"token_budget is too small for chosen batch_size and context_length: "
			f"{args.token_budget=} {tokens_per_iter=}"
		)
	warmup_iters = max(1, int(max_iters * args.warmup_frac))

	print(
		f"No-RMSNorm ablation: token_budget={args.token_budget:,}, tokens/iter={tokens_per_iter:,}, "
		f"max_iters={max_iters:,}, warmup_iters={warmup_iters:,}, device={device}"
	)

	checkpoint_dir = Path(args.checkpoint_dir)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	base_lr = float(args.previous_optimal_lr)
	base_result = _train_single_lr(
		args=args,
		train_tokens=train_tokens,
		val_tokens=val_tokens,
		device=device,
		torch_dtype=torch_dtype,
		lr=base_lr,
		run_name=f"layer-norm-ablation-no-rmsnorm-lr-{_lr_slug(base_lr)}",
		checkpoint_path=checkpoint_dir / f"ckpt_no_rmsnorm_lr_{_lr_slug(base_lr)}.pt",
		max_iters=max_iters,
		warmup_iters=warmup_iters,
	)

	lower_results: list[RunResult] = []
	for lr in args.lower_lrs:
		result = _train_single_lr(
			args=args,
			train_tokens=train_tokens,
			val_tokens=val_tokens,
			device=device,
			torch_dtype=torch_dtype,
			lr=float(lr),
			run_name=f"layer-norm-ablation-no-rmsnorm-lr-{_lr_slug(float(lr))}",
			checkpoint_path=checkpoint_dir / f"ckpt_no_rmsnorm_lr_{_lr_slug(float(lr))}.pt",
			max_iters=max_iters,
			warmup_iters=warmup_iters,
		)
		lower_results.append(result)

	stable_lower = [row for row in lower_results if not row.diverged and row.final_val_loss is not None]
	best_lower = min(stable_lower, key=lambda row: row.final_val_loss) if stable_lower else None

	curves_csv, report_md = _write_outputs(args, base_result=base_result, best_lower=best_lower)

	print("\nLayer norm ablation complete.")
	print(f"- Previous optimal LR run dir: {base_result.run_dir}")
	print(f"- Previous optimal LR status: {'diverged' if base_result.diverged else 'stable'}")
	if best_lower is not None:
		print(f"- Best lower LR without RMSNorm: {best_lower.lr:.8g}")
		print(f"- Best lower LR run dir: {best_lower.run_dir}")
	else:
		print("- No stable lower-LR run found.")
	print(f"- Combined curves CSV: {curves_csv}")
	print(f"- Commentary report: {report_md}")


if __name__ == "__main__":
	main()
