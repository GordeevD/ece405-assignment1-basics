from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .tinystories_example import (
	SweepResult,
	TrainConfig,
	maybe_disable_tf32_for_mps,
	parse_lr_list,
	run_single_lr,
)
from .training_together import load_token_memmap, parse_numpy_dtype, parse_torch_dtype, resolve_device


def _timestamp_utc() -> str:
	return datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class BatchLrTrial:
	batch_size: int
	lr_max: float
	status: str
	best_val_loss: float | None
	best_val_step: int | None
	run_dir: str


@dataclass
class BatchBestResult:
	batch_size: int
	best_lr: float | None
	best_val_loss: float | None
	best_val_step: int | None
	status: str
	run_dir: str | None


def parse_batch_sizes(value: str) -> list[int]:
	parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
	if not parts:
		raise ValueError("batch-size list cannot be empty")
	batch_sizes = [int(chunk) for chunk in parts]
	for batch_size in batch_sizes:
		if batch_size <= 0:
			raise ValueError(f"Invalid batch size: {batch_size}")
	return batch_sizes


def is_oom_error(exc: BaseException) -> bool:
	message = str(exc).lower()
	return "out of memory" in message or "mps backend out of memory" in message


def clear_device_cache(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.empty_cache()
	elif device.type == "mps":
		torch.mps.empty_cache()


def probe_max_batch_size(
	base_config: TrainConfig,
	device: torch.device,
	torch_dtype: torch.dtype,
	np_dtype: np.dtype,
	train_tokens: np.ndarray,
	val_tokens: np.ndarray,
	learning_rate_for_probe: float,
	upper_bound: int,
) -> int:
	low = 1
	high = 1
	best = 1

	def can_run(batch_size: int) -> bool:
		probe_config = replace(
			base_config,
			batch_size=batch_size,
			total_tokens=batch_size * base_config.context_length,
			eval_interval=1,
			eval_batches=1,
			log_interval=1,
			run_prefix="batch-probe",
		)
		try:
			run_single_lr(
				config=probe_config,
				lr_max=learning_rate_for_probe,
				train_tokens=train_tokens,
				val_tokens=val_tokens,
				device=device,
				torch_dtype=torch_dtype,
				np_dtype=np_dtype,
				divergence_loss_threshold=1e9,
			)
			return True
		except RuntimeError as exc:
			if is_oom_error(exc):
				clear_device_cache(device)
				return False
			raise

	while high <= upper_bound and can_run(high):
		best = high
		low = high
		high *= 2

	high = min(high, upper_bound)
	while low < high:
		mid = (low + high + 1) // 2
		if can_run(mid):
			best = mid
			low = mid
		else:
			high = mid - 1

	return best


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="TinyStories batch-size sweep with per-batch LR tuning.")

	parser.add_argument("--batch-sizes", type=str, default="1,8,32,64,128")
	parser.add_argument("--learning-rates", type=str, default="1e-4,3e-4,1e-3,2e-3")
	parser.add_argument("--include-auto-max-batch", action="store_true")
	parser.add_argument("--max-batch-upper-bound", type=int, default=2048)

	parser.add_argument("--total-tokens", type=int, default=40_000_000)
	parser.add_argument("--target-val-loss", type=float, default=2.0)
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
	parser.add_argument("--run-prefix", type=str, default="batch-size")

	parser.add_argument(
		"--divergence-loss-threshold",
		type=float,
		default=20.0,
		help="Run is marked divergent if train loss exceeds this threshold after warmup.",
	)

	return parser


def write_batch_summary(
	trial_results: list[BatchLrTrial],
	best_results: list[BatchBestResult],
	config: TrainConfig,
	learning_rates: list[float],
) -> tuple[Path, Path]:
	experiment_root = Path(config.experiment_dir)
	experiment_root.mkdir(parents=True, exist_ok=True)
	ts = _timestamp_utc()
	csv_path = experiment_root / f"batch_size_summary_{ts}.csv"
	md_path = experiment_root / f"batch_size_summary_{ts}.md"

	with csv_path.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=[
				"batch_size",
				"best_lr",
				"best_val_loss",
				"best_val_step",
				"status",
				"run_dir",
			],
		)
		writer.writeheader()
		for row in best_results:
			writer.writerow(
				{
					"batch_size": row.batch_size,
					"best_lr": row.best_lr,
					"best_val_loss": row.best_val_loss,
					"best_val_step": row.best_val_step,
					"status": row.status,
					"run_dir": row.run_dir,
				}
			)

	stable_best = [row for row in best_results if row.best_val_loss is not None and row.status == "completed"]
	stable_best.sort(key=lambda row: float(row.best_val_loss))
	global_best = stable_best[0] if stable_best else None

	lines: list[str] = []
	lines.append("# TinyStories batch-size experiment summary")
	lines.append("")
	lines.append("## Setup")
	lines.append(f"- total_tokens: {config.total_tokens}")
	lines.append(f"- context_length: {config.context_length}")
	lines.append(f"- learning_rates_per_batch: {', '.join(f'{lr:.2e}' for lr in sorted(learning_rates))}")
	lines.append(f"- target_val_loss: {config.target_val_loss}")
	lines.append("")
	lines.append("## Best run per batch size")
	lines.append("")
	lines.append("| batch_size | best_lr | best_val_loss | best_val_step | status | run_dir |")
	lines.append("|---:|---:|---:|---:|---|---|")
	for row in sorted(best_results, key=lambda item: item.batch_size):
		best_lr_text = "N/A" if row.best_lr is None else f"{row.best_lr:.2e}"
		best_loss_text = "N/A" if row.best_val_loss is None else f"{row.best_val_loss:.4f}"
		best_step_text = "N/A" if row.best_val_step is None else str(row.best_val_step)
		run_dir_text = row.run_dir if row.run_dir is not None else "N/A"
		lines.append(
			f"| {row.batch_size} | {best_lr_text} | {best_loss_text} | {best_step_text} | {row.status} | {run_dir_text} |"
		)

	lines.append("")
	lines.append("## Findings")
	if global_best is None:
		lines.append("No batch size completed stably with measurable validation loss.")
	else:
		lines.append(
			f"Best overall setting: batch_size={global_best.batch_size}, lr={global_best.best_lr:.2e}, "
			f"best_val_loss={global_best.best_val_loss:.4f}."
		)
		if global_best.best_val_loss is not None and global_best.best_val_loss <= config.target_val_loss:
			lines.append(f"Target met: validation loss <= {config.target_val_loss:.2f}.")
		else:
			lines.append(f"Target not met: validation loss > {config.target_val_loss:.2f}.")

	divergent_trials = [t for t in trial_results if t.status.startswith("diverged")]
	oom_trials = [t for t in trial_results if t.status == "oom"]
	if divergent_trials:
		lines.append(
			f"Observed instability at higher learning rates in {len(divergent_trials)} runs, "
			"which supports re-tuning LR for each batch size."
		)
	if oom_trials:
		lines.append(
			f"Observed OOM at {len(oom_trials)} batch/LR combinations; these mark the practical memory limit region."
		)

	lines.append("")
	lines.append("## Learning curves")
	lines.append(
		"Each run directory includes `loss_curves.csv` for plotting train/val curves across batch sizes and tuned LRs."
	)

	md_path.write_text("\n".join(lines), encoding="utf-8")
	return csv_path, md_path


def main() -> None:
	args = build_parser().parse_args()
	learning_rates = parse_lr_list(args.learning_rates)
	batch_sizes = sorted(set(parse_batch_sizes(args.batch_sizes)))

	base_config = TrainConfig(
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		total_tokens=args.total_tokens,
		batch_size=64,
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

	device = resolve_device(base_config.device)
	maybe_disable_tf32_for_mps(device)
	torch_dtype = parse_torch_dtype(base_config.dtype)
	np_dtype = parse_numpy_dtype(base_config.data_dtype)

	train_tokens = load_token_memmap(base_config.train_data, np_dtype)
	val_tokens = load_token_memmap(base_config.val_data, np_dtype)

	min_required = base_config.context_length + 1
	if len(train_tokens) < min_required:
		raise ValueError(f"train dataset too small for context_length={base_config.context_length}")
	if len(val_tokens) < min_required:
		raise ValueError(f"val dataset too small for context_length={base_config.context_length}")

	if args.include_auto_max_batch:
		probe_lr = min(learning_rates)
		max_batch = probe_max_batch_size(
			base_config=base_config,
			device=device,
			torch_dtype=torch_dtype,
			np_dtype=np_dtype,
			train_tokens=train_tokens,
			val_tokens=val_tokens,
			learning_rate_for_probe=probe_lr,
			upper_bound=args.max_batch_upper_bound,
		)
		batch_sizes = sorted(set(batch_sizes + [max_batch]))
		print(f"Auto-probed max batch size: {max_batch}")

	trial_results: list[BatchLrTrial] = []
	best_results: list[BatchBestResult] = []

	for batch_size in batch_sizes:
		print(f"\n=== Batch size {batch_size}: sweeping learning rates ===")
		batch_trials: list[BatchLrTrial] = []
		for lr in learning_rates:
			run_config = replace(base_config, batch_size=batch_size, run_prefix=f"{base_config.run_prefix}-bs{batch_size}")
			try:
				outcome: SweepResult = run_single_lr(
					config=run_config,
					lr_max=lr,
					train_tokens=train_tokens,
					val_tokens=val_tokens,
					device=device,
					torch_dtype=torch_dtype,
					np_dtype=np_dtype,
					divergence_loss_threshold=args.divergence_loss_threshold,
				)
				trial = BatchLrTrial(
					batch_size=batch_size,
					lr_max=lr,
					status=outcome.status,
					best_val_loss=outcome.best_val_loss,
					best_val_step=outcome.best_val_step,
					run_dir=outcome.run_dir,
				)
			except RuntimeError as exc:
				if not is_oom_error(exc):
					raise
				clear_device_cache(device)
				trial = BatchLrTrial(
					batch_size=batch_size,
					lr_max=lr,
					status="oom",
					best_val_loss=None,
					best_val_step=None,
					run_dir="N/A",
				)

			trial_results.append(trial)
			batch_trials.append(trial)
			print(
				f"batch_size={batch_size}, lr={lr:.2e}, status={trial.status}, "
				f"best_val_loss={trial.best_val_loss}, run_dir={trial.run_dir}"
			)

		stable = [trial for trial in batch_trials if trial.status == "completed" and trial.best_val_loss is not None]
		stable.sort(key=lambda trial: float(trial.best_val_loss))
		if stable:
			best_trial = stable[0]
			best_results.append(
				BatchBestResult(
					batch_size=batch_size,
					best_lr=best_trial.lr_max,
					best_val_loss=best_trial.best_val_loss,
					best_val_step=best_trial.best_val_step,
					status="completed",
					run_dir=best_trial.run_dir,
				)
			)
			continue

		oom_count = sum(1 for trial in batch_trials if trial.status == "oom")
		best_results.append(
			BatchBestResult(
				batch_size=batch_size,
				best_lr=None,
				best_val_loss=None,
				best_val_step=None,
				status="oom" if oom_count == len(batch_trials) else "no_stable_run",
				run_dir=None,
			)
		)

	csv_path, md_path = write_batch_summary(trial_results, best_results, base_config, learning_rates)
	print("\nBatch-size sweep complete.")
	print(f"Summary CSV: {csv_path}")
	print(f"Summary MD : {md_path}")


if __name__ == "__main__":
	main()
