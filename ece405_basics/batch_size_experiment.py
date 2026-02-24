from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer
from .training_together import load_token_memmap


def _timestamp_utc() -> str:
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


def _parse_batch_size_list(raw: str) -> list[int]:
	values: list[int] = []
	for token in raw.split(","):
		candidate = token.strip()
		if not candidate:
			continue
		batch_size = int(candidate)
		if batch_size <= 0:
			raise ValueError(f"Batch size must be positive, got {batch_size}")
		values.append(batch_size)
	if not values:
		raise ValueError("No batch sizes provided.")
	return values


@dataclass(frozen=True)
class BatchLrTrialSpec:
	index: int
	batch_size: int
	lr: float
	run_name: str


@dataclass
class BatchLrTrialResult:
	batch_size: int
	lr: float
	run_dir: str
	status: str
	best_val_loss: float | None
	best_val_step: int | None
	final_val_loss: float | None
	notes: str = ""


@dataclass
class BatchBestResult:
	batch_size: int
	best_lr: float | None
	status: str
	best_val_loss: float | None
	best_val_step: int | None
	final_val_loss: float | None
	run_dir: str | None


def _find_latest_run_dir(experiment_dir: Path, run_name: str) -> Path:
	pattern = f"{run_name}-*"
	candidates = [path for path in experiment_dir.glob(pattern) if path.is_dir()]
	if not candidates:
		raise FileNotFoundError(f"No experiment run directory found for run name: {run_name}")
	return max(candidates, key=lambda path: path.stat().st_mtime)


def _iter_text_chunks(path: Path, chunk_size: int = 1 << 20):
	with path.open("r", encoding="utf-8") as fp:
		while True:
			chunk = fp.read(chunk_size)
			if not chunk:
				break
			yield chunk


def _build_tokenizer(vocab_path: Path, merges_path: Path, end_of_text_token: str) -> Tokenizer:
	if not vocab_path.exists():
		raise FileNotFoundError(f"Tokenizer vocab not found: {vocab_path}")
	if not merges_path.exists():
		raise FileNotFoundError(f"Tokenizer merges not found: {merges_path}")
	return Tokenizer.from_files(
		vocab_filepath=str(vocab_path),
		merges_filepath=str(merges_path),
		special_tokens=[end_of_text_token],
	)


def _resolve_token_data_path(raw_path: Path, label: str, args: argparse.Namespace, tokenizer: Tokenizer) -> Path:
	if raw_path.suffix.lower() != ".txt":
		return raw_path

	cached_npy = raw_path.with_suffix(".npy")
	if cached_npy.exists():
		print(f"Using cached tokenized {label} data: {cached_npy}")
		return cached_npy

	print(f"No cached {label} tokens found for {raw_path}; tokenizing to {cached_npy}...")
	ids_iter = tokenizer.encode_iterable(_iter_text_chunks(raw_path))
	token_ids = np.fromiter(ids_iter, dtype=np.int64)
	if token_ids.size == 0:
		raise ValueError(f"Tokenization produced no tokens for {label} corpus: {raw_path}")

	np_dtype = np.dtype(args.data_dtype)
	if np.issubdtype(np_dtype, np.integer):
		dtype_info = np.iinfo(np_dtype)
		max_id = int(token_ids.max())
		if max_id > int(dtype_info.max):
			raise ValueError(
				f"Token id {max_id} exceeds dtype range for --data-dtype {args.data_dtype}. "
				"Choose a wider integer dtype (e.g., uint32)."
			)

	np.save(cached_npy, token_ids.astype(np_dtype, copy=False))
	print(f"Saved tokenized {label} data: {cached_npy} ({token_ids.size} tokens)")
	return cached_npy


def _extract_summary_stats(run_dir: Path) -> tuple[float | None, int | None, bool]:
	summary_path = run_dir / "summary.json"
	if not summary_path.exists():
		return None, None, True

	try:
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return None, None, True

	best_val_loss = summary.get("best_val_loss")
	best_val_step = summary.get("best_val_step")

	if isinstance(best_val_loss, (int, float)) and not math.isfinite(float(best_val_loss)):
		return None, int(best_val_step) if isinstance(best_val_step, int) else None, True

	parsed_best = float(best_val_loss) if isinstance(best_val_loss, (int, float)) else None
	parsed_step = int(best_val_step) if isinstance(best_val_step, int) else None
	return parsed_best, parsed_step, False


def _extract_final_val_loss(run_dir: Path) -> float | None:
	curve_path = run_dir / "loss_curves.csv"
	if not curve_path.exists():
		return None

	last_val: float | None = None
	with curve_path.open("r", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			if row.get("split") != "val":
				continue
			try:
				last_val = float(row["loss"])
			except (KeyError, TypeError, ValueError):
				continue
	return last_val


def _is_capacity_limit_output(stdout: str, stderr: str) -> bool:
	text = (stdout + "\n" + stderr).lower()
	return any(
		pattern in text
		for pattern in (
			"out of memory",
			"mps backend out of memory",
			"dimension length > int_max",
			"mpsndarray",
		)
	)


def _build_training_cmd(
	*,
	args: argparse.Namespace,
	train_data: Path,
	val_data: Path,
	experiment_dir: Path,
	checkpoint_path: Path,
	run_name: str,
	batch_size: int,
	lr: float,
	max_iters: int,
	warmup_iters: int,
	eval_interval: int,
	eval_batches: int,
	log_interval: int,
	checkpoint_interval: int,
) -> list[str]:
	cmd = [
		sys.executable,
		"-m",
		"ece405_basics.training_together",
		"--train-data",
		str(train_data),
		"--val-data",
		str(val_data),
		"--data-dtype",
		args.data_dtype,
		"--vocab-size",
		str(args.vocab_size),
		"--context-length",
		str(args.context_length),
		"--d-model",
		str(args.d_model),
		"--num-layers",
		str(args.num_layers),
		"--num-heads",
		str(args.num_heads),
		"--d-ff",
		str(args.d_ff),
		"--rope-theta",
		str(args.rope_theta),
		"--batch-size",
		str(batch_size),
		"--max-iters",
		str(max_iters),
		"--lr-max",
		str(lr),
		"--lr-min",
		str(lr * args.lr_min_ratio),
		"--warmup-iters",
		str(warmup_iters),
		"--cosine-iters",
		str(max_iters),
		"--beta1",
		str(args.beta1),
		"--beta2",
		str(args.beta2),
		"--eps",
		str(args.eps),
		"--weight-decay",
		str(args.weight_decay),
		"--max-grad-norm",
		str(args.max_grad_norm),
		"--device",
		args.device,
		"--dtype",
		args.dtype,
		"--seed",
		str(args.seed),
		"--log-interval",
		str(log_interval),
		"--eval-interval",
		str(eval_interval),
		"--eval-batches",
		str(eval_batches),
		"--checkpoint-path",
		str(checkpoint_path),
		"--checkpoint-interval",
		str(checkpoint_interval),
		"--experiment-dir",
		str(experiment_dir),
		"--run-name",
		run_name,
		"--use-wandb",
		"true" if args.use_wandb else "false",
		"--wandb-project",
		args.wandb_project,
	]
	if args.wandb_entity:
		cmd.extend(["--wandb-entity", args.wandb_entity])
	cmd.extend(["--wandb-run-name", run_name])
	return cmd


def _run_trial(
	trial: BatchLrTrialSpec,
	*,
	args: argparse.Namespace,
	train_data: Path,
	val_data: Path,
	experiment_dir: Path,
	checkpoint_dir: Path,
	max_iters: int,
	warmup_iters: int,
) -> BatchLrTrialResult:
	run_slug = f"bs{trial.batch_size}_lr{_lr_slug(trial.lr)}"
	checkpoint_path = checkpoint_dir / f"ckpt_{run_slug}_{trial.index:03d}.pt"

	cmd = _build_training_cmd(
		args=args,
		train_data=train_data,
		val_data=val_data,
		experiment_dir=experiment_dir,
		checkpoint_path=checkpoint_path,
		run_name=trial.run_name,
		batch_size=trial.batch_size,
		lr=trial.lr,
		max_iters=max_iters,
		warmup_iters=warmup_iters,
		eval_interval=args.eval_interval,
		eval_batches=args.eval_batches,
		log_interval=args.log_interval,
		checkpoint_interval=args.checkpoint_interval,
	)

	status = "ok"
	notes = ""
	try:
		subprocess.run(cmd, check=True)
	except subprocess.CalledProcessError as exc:
		status = "failed"
		notes = f"training process failed with exit code {exc.returncode}"

	try:
		run_dir = _find_latest_run_dir(experiment_dir, trial.run_name)
	except FileNotFoundError:
		return BatchLrTrialResult(
			batch_size=trial.batch_size,
			lr=trial.lr,
			run_dir="N/A",
			status="failed",
			best_val_loss=None,
			best_val_step=None,
			final_val_loss=None,
			notes=(notes + "; no run directory found").strip("; "),
		)

	best_val_loss, best_val_step, summary_diverged = _extract_summary_stats(run_dir)
	final_val_loss = _extract_final_val_loss(run_dir)

	if summary_diverged:
		status = "diverged" if status == "ok" else status

	return BatchLrTrialResult(
		batch_size=trial.batch_size,
		lr=trial.lr,
		run_dir=str(run_dir),
		status=status,
		best_val_loss=best_val_loss,
		best_val_step=best_val_step,
		final_val_loss=final_val_loss,
		notes=notes,
	)


def _can_run_probe(
	batch_size: int,
	*,
	args: argparse.Namespace,
	train_data: Path,
	val_data: Path,
	experiment_dir: Path,
	checkpoint_dir: Path,
	probe_lr: float,
) -> bool:
	run_name = f"batch-probe-{batch_size}"
	checkpoint_path = checkpoint_dir / f"probe_bs{batch_size}.pt"
	cmd = _build_training_cmd(
		args=args,
		train_data=train_data,
		val_data=val_data,
		experiment_dir=experiment_dir,
		checkpoint_path=checkpoint_path,
		run_name=run_name,
		batch_size=batch_size,
		lr=probe_lr,
		max_iters=args.probe_max_iters,
		warmup_iters=1,
		eval_interval=1,
		eval_batches=1,
		log_interval=1,
		checkpoint_interval=1,
	)
	try:
		proc = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=max(1, int(args.probe_timeout_seconds)),
		)
	except subprocess.TimeoutExpired:
		return False
	if proc.returncode == 0:
		return True
	if _is_capacity_limit_output(proc.stdout, proc.stderr):
		return False
	err_tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-20:])
	raise RuntimeError(
		"Batch-size probe failed for non-capacity reason at "
		f"batch_size={batch_size}.\nLast output lines:\n{err_tail}"
	)


def probe_max_batch_size(
	*,
	args: argparse.Namespace,
	train_data: Path,
	val_data: Path,
	probe_lr: float,
) -> int:
	probe_experiment_dir = Path(args.probe_experiment_dir)
	probe_checkpoint_dir = Path(args.probe_checkpoint_dir)
	probe_experiment_dir.mkdir(parents=True, exist_ok=True)
	probe_checkpoint_dir.mkdir(parents=True, exist_ok=True)

	upper_bound = max(1, int(args.max_batch_upper_bound))
	low = 1
	high = 1
	best = 1

	while high <= upper_bound and _can_run_probe(
		high,
		args=args,
		train_data=train_data,
		val_data=val_data,
		experiment_dir=probe_experiment_dir,
		checkpoint_dir=probe_checkpoint_dir,
		probe_lr=probe_lr,
	):
		best = high
		low = high
		high *= 2

	high = min(high, upper_bound)
	while low < high:
		mid = (low + high + 1) // 2
		if _can_run_probe(
			mid,
			args=args,
			train_data=train_data,
			val_data=val_data,
			experiment_dir=probe_experiment_dir,
			checkpoint_dir=probe_checkpoint_dir,
			probe_lr=probe_lr,
		):
			best = mid
			low = mid
		else:
			high = mid - 1

	return best


def _build_batch_size_list(args: argparse.Namespace, max_batch_size: int | None) -> list[int]:
	candidate_sizes = set(_parse_batch_size_list(args.batch_sizes))
	candidate_sizes.add(1)

	if args.include_typical_batch_sizes:
		candidate_sizes.update({64, 128})

	if max_batch_size is not None:
		candidate_sizes.add(max_batch_size)
		if args.add_power_of_two_intermediates:
			value = 1
			while value <= max_batch_size:
				candidate_sizes.add(value)
				value *= 2

	return sorted(size for size in candidate_sizes if size > 0)


def run_sweep(args: argparse.Namespace) -> tuple[list[BatchLrTrialResult], list[BatchBestResult], int | None]:
	train_data = Path(args.train_data)
	val_data = Path(args.val_data)
	experiment_dir = Path(args.experiment_dir)
	checkpoint_dir = Path(args.checkpoint_dir)

	experiment_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	if args.context_length <= 0:
		raise ValueError("context_length must be positive")

	tokenizer = _build_tokenizer(
		vocab_path=Path(args.tokenizer_vocab),
		merges_path=Path(args.tokenizer_merges),
		end_of_text_token=args.end_of_text_token,
	)

	train_data = _resolve_token_data_path(train_data, "train", args, tokenizer)
	val_data = _resolve_token_data_path(val_data, "val", args, tokenizer)

	try:
		np_dtype = np.dtype(args.data_dtype)
	except TypeError as exc:
		raise ValueError(f"Unsupported numpy dtype: {args.data_dtype}") from exc

	train_tokens = load_token_memmap(str(train_data), np_dtype)
	val_tokens = load_token_memmap(str(val_data), np_dtype)

	min_required_tokens = args.context_length + 1
	if len(train_tokens) < min_required_tokens:
		raise ValueError(
			f"train dataset too small: {len(train_tokens)} tokens, need at least {min_required_tokens}"
		)
	if len(val_tokens) < min_required_tokens:
		raise ValueError(
			f"val dataset too small: {len(val_tokens)} tokens, need at least {min_required_tokens}"
		)

	max_token_id = max(int(train_tokens.max()), int(val_tokens.max()))
	if max_token_id >= args.vocab_size:
		new_vocab_size = max_token_id + 1
		print(
			f"Adjusting --vocab-size from {args.vocab_size} to {new_vocab_size} "
			f"to fit max token id {max_token_id}."
		)
		args.vocab_size = new_vocab_size

	max_batch_size: int | None = None
	if args.include_auto_max_batch:
		probe_lr = min(args.lrs)
		print(f"Probing max feasible batch size (probe lr={probe_lr:.2e})...")
		max_batch_size = probe_max_batch_size(
			args=args,
			train_data=train_data,
			val_data=val_data,
			probe_lr=probe_lr,
		)
		print(f"Auto-probed max feasible batch size: {max_batch_size}")

	batch_sizes = _build_batch_size_list(args, max_batch_size)
	if max_batch_size is not None:
		batch_sizes = [size for size in batch_sizes if size <= max_batch_size]
	if not batch_sizes:
		raise ValueError("No valid batch sizes to run after filtering.")

	trial_specs: list[BatchLrTrialSpec] = []
	for batch_size in batch_sizes:
		for lr in args.lrs:
			idx = len(trial_specs) + 1
			trial_specs.append(
				BatchLrTrialSpec(
					index=idx,
					batch_size=batch_size,
					lr=lr,
					run_name=f"batch-sweep-bs{batch_size}-lr{_lr_slug(lr)}-{idx:03d}",
				)
			)

	trial_results: list[BatchLrTrialResult] = []
	best_results: list[BatchBestResult] = []

	for batch_size in batch_sizes:
		tokens_per_iter = batch_size * args.context_length
		max_iters = args.token_budget // tokens_per_iter
		if max_iters <= 0:
			print(
				f"Skipping batch_size={batch_size}: token_budget={args.token_budget} "
				f"is too small for tokens/iter={tokens_per_iter}."
			)
			best_results.append(
				BatchBestResult(
					batch_size=batch_size,
					best_lr=None,
					status="skipped",
					best_val_loss=None,
					best_val_step=None,
					final_val_loss=None,
					run_dir=None,
				)
			)
			continue

		warmup_iters = max(1, int(max_iters * args.warmup_frac))
		print(
			f"\n=== batch_size={batch_size}: max_iters={max_iters}, warmup_iters={warmup_iters} ==="
		)

		batch_trials: list[BatchLrTrialResult] = []
		for trial in [spec for spec in trial_specs if spec.batch_size == batch_size]:
			print(f"Running batch_size={batch_size}, lr={trial.lr:.3e} ({trial.run_name})")
			result = _run_trial(
				trial,
				args=args,
				train_data=train_data,
				val_data=val_data,
				experiment_dir=experiment_dir,
				checkpoint_dir=checkpoint_dir,
				max_iters=max_iters,
				warmup_iters=warmup_iters,
			)
			trial_results.append(result)
			batch_trials.append(result)
			print(
				f"Result batch_size={batch_size}, lr={trial.lr:.3e}: status={result.status}, "
				f"final_val={result.final_val_loss}, best_val={result.best_val_loss}"
			)

		completed = [
			row
			for row in batch_trials
			if row.status == "ok" and row.final_val_loss is not None and row.best_val_loss is not None
		]
		if completed:
			best = min(completed, key=lambda row: float(row.final_val_loss))
			best_results.append(
				BatchBestResult(
					batch_size=batch_size,
					best_lr=best.lr,
					status="ok",
					best_val_loss=best.best_val_loss,
					best_val_step=best.best_val_step,
					final_val_loss=best.final_val_loss,
					run_dir=best.run_dir,
				)
			)
		else:
			status = "no_stable_run"
			if any(row.status == "failed" for row in batch_trials):
				status = "failed"
			best_results.append(
				BatchBestResult(
					batch_size=batch_size,
					best_lr=None,
					status=status,
					best_val_loss=None,
					best_val_step=None,
					final_val_loss=None,
					run_dir=None,
				)
			)

	return trial_results, best_results, max_batch_size


def _write_combined_learning_curves(best_results: list[BatchBestResult], out_dir: Path) -> Path:
	stamp = _timestamp_utc()
	combined_curve_path = out_dir / f"batch_size_learning_curves_{stamp}.csv"

	with combined_curve_path.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=["batch_size", "lr", "gradient_step", "wallclock_time_s", "split", "loss", "run_dir"],
		)
		writer.writeheader()
		for result in sorted(best_results, key=lambda item: item.batch_size):
			if result.status != "ok" or result.run_dir is None or result.best_lr is None:
				continue
			curve_path = Path(result.run_dir) / "loss_curves.csv"
			if not curve_path.exists():
				continue
			with curve_path.open("r", encoding="utf-8") as in_fp:
				reader = csv.DictReader(in_fp)
				for row in reader:
					writer.writerow(
						{
							"batch_size": result.batch_size,
							"lr": f"{result.best_lr:.8g}",
							"gradient_step": row.get("gradient_step", ""),
							"wallclock_time_s": row.get("wallclock_time_s", ""),
							"split": row.get("split", ""),
							"loss": row.get("loss", ""),
							"run_dir": result.run_dir,
						}
					)

	return combined_curve_path


def _discussion_lines(best_results: list[BatchBestResult]) -> list[str]:
	stable = [row for row in best_results if row.status == "ok" and row.final_val_loss is not None and row.best_lr is not None]
	stable.sort(key=lambda row: row.batch_size)

	if not stable:
		return [
			"No batch size completed stably with measurable validation loss.",
			"This suggests the current LR grid or token budget should be adjusted before drawing conclusions.",
		]

	overall_best = min(stable, key=lambda row: float(row.final_val_loss))
	lines = [
		(
			f"After re-tuning learning rate per batch size, the best observed configuration is "
			f"batch_size={overall_best.batch_size} with lr={overall_best.best_lr:.2e}, "
			f"final_val_loss={overall_best.final_val_loss:.4f}."
		)
	]

	if len(stable) >= 2:
		smallest = stable[0]
		largest = stable[-1]
		delta = float(largest.final_val_loss) - float(smallest.final_val_loss)
		trend = "improves" if delta < 0 else "worsens"
		lines.append(
			(
				f"Comparing smallest and largest stable batches ({smallest.batch_size} -> {largest.batch_size}), "
				f"validation loss {trend} by {abs(delta):.4f} in this setup."
			)
		)

		small_lr = float(smallest.best_lr)
		large_lr = float(largest.best_lr)
		if large_lr > small_lr:
			lr_trend = "increases"
		elif large_lr < small_lr:
			lr_trend = "decreases"
		else:
			lr_trend = "stays roughly constant"
		lines.append(
			f"The tuned learning rate generally {lr_trend} as batch size grows, so LR re-optimization is important."
		)
	else:
		lines.append("Only one stable batch size was observed, so trend analysis is limited.")

	return lines


def write_summary(
	trial_results: list[BatchLrTrialResult],
	best_results: list[BatchBestResult],
	out_dir: Path,
	learning_rates: list[float],
	max_batch_size: int | None,
) -> tuple[Path, Path, Path]:
	out_dir.mkdir(parents=True, exist_ok=True)
	stamp = _timestamp_utc()
	trial_csv = out_dir / f"batch_lr_trials_{stamp}.csv"
	best_csv = out_dir / f"batch_size_summary_{stamp}.csv"
	md_path = out_dir / f"batch_size_summary_{stamp}.md"

	with trial_csv.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=[
				"batch_size",
				"lr",
				"status",
				"final_val_loss",
				"best_val_loss",
				"best_val_step",
				"run_dir",
				"notes",
			],
		)
		writer.writeheader()
		for row in trial_results:
			writer.writerow(
				{
					"batch_size": row.batch_size,
					"lr": f"{row.lr:.8g}",
					"status": row.status,
					"final_val_loss": "" if row.final_val_loss is None else f"{row.final_val_loss:.6f}",
					"best_val_loss": "" if row.best_val_loss is None else f"{row.best_val_loss:.6f}",
					"best_val_step": "" if row.best_val_step is None else str(row.best_val_step),
					"run_dir": row.run_dir,
					"notes": row.notes,
				}
			)

	with best_csv.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=[
				"batch_size",
				"best_lr",
				"status",
				"final_val_loss",
				"best_val_loss",
				"best_val_step",
				"run_dir",
			],
		)
		writer.writeheader()
		for row in sorted(best_results, key=lambda item: item.batch_size):
			writer.writerow(
				{
					"batch_size": row.batch_size,
					"best_lr": "" if row.best_lr is None else f"{row.best_lr:.8g}",
					"status": row.status,
					"final_val_loss": "" if row.final_val_loss is None else f"{row.final_val_loss:.6f}",
					"best_val_loss": "" if row.best_val_loss is None else f"{row.best_val_loss:.6f}",
					"best_val_step": "" if row.best_val_step is None else str(row.best_val_step),
					"run_dir": "" if row.run_dir is None else row.run_dir,
				}
			)

	combined_curves_path = _write_combined_learning_curves(best_results, out_dir)

	lines: list[str] = [
		"# Batch-size sweep summary",
		"",
		"## Setup",
		f"- learning rates swept per batch size: {', '.join(f'{lr:.2e}' for lr in sorted(learning_rates))}",
	]
	if max_batch_size is not None:
		lines.append(f"- auto-probed GPU memory limit batch size: {max_batch_size}")
	lines.extend(
		[
			"",
			"## Best tuned result per batch size",
			"",
			"| batch_size | best_lr | status | final_val_loss | best_val_loss | best_val_step | run_dir |",
		]
	)
	for row in sorted(best_results, key=lambda item: item.batch_size):
		best_lr = "-" if row.best_lr is None else f"{row.best_lr:.2e}"
		final_loss = "-" if row.final_val_loss is None else f"{row.final_val_loss:.6f}"
		best_loss = "-" if row.best_val_loss is None else f"{row.best_val_loss:.6f}"
		best_step = "-" if row.best_val_step is None else str(row.best_val_step)
		run_dir = "-" if row.run_dir is None else row.run_dir
		lines.append(
			f"| {row.batch_size} | {best_lr} | {row.status} | {final_loss} | {best_loss} | {best_step} | {run_dir} |"
		)

	lines.extend(
		[
			"",
			"## Learning curves deliverable",
			"",
			"- Combined curves CSV for best-tuned run per batch size:",
			f"  - {combined_curves_path}",
			"- Per-run learning curves are also available in each run directory as `loss_curves.csv`.",
			"",
			"## Findings",
		]
	)
	lines.extend(f"- {sentence}" for sentence in _discussion_lines(best_results))

	md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return trial_csv, best_csv, md_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run TinyStories batch-size experiments with per-batch learning-rate re-tuning."
	)

	parser.add_argument(
		"--train-data",
		type=str,
		default="tests/fixtures/TinyStoriesV2-GPT4-train.txt",
	)
	parser.add_argument(
		"--val-data",
		type=str,
		default="tests/fixtures/TinyStoriesV2-GPT4-valid.txt",
	)
	parser.add_argument("--data-dtype", type=str, default="uint16")

	parser.add_argument("--vocab-size", type=int, default=512)
	parser.add_argument("--context-length", type=int, default=256)
	parser.add_argument("--d-model", type=int, default=64)
	parser.add_argument("--num-layers", type=int, default=2)
	parser.add_argument("--num-heads", type=int, default=4)
	parser.add_argument("--d-ff", type=int, default=128)
	parser.add_argument("--rope-theta", type=float, default=10_000.0)

	parser.add_argument("--token-budget", type=int, default=40_000_000)
	parser.add_argument("--batch-sizes", type=str, default="1,8,16,32,64,128")
	parser.add_argument("--lrs", type=_parse_lr_list, default="1e-4,3e-4,1e-3,2e-3,3e-3")
	parser.add_argument("--lr-min-ratio", type=float, default=0.1)
	parser.add_argument("--warmup-frac", type=float, default=0.03)

	parser.add_argument("--include-auto-max-batch", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--max-batch-upper-bound", type=int, default=2048)
	parser.add_argument("--probe-max-iters", type=int, default=2)
	parser.add_argument("--probe-timeout-seconds", type=int, default=120)
	parser.add_argument("--include-typical-batch-sizes", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--add-power-of-two-intermediates", action=argparse.BooleanOptionalAction, default=True)

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
	parser.add_argument("--probe-experiment-dir", type=str, default="experiments_tmp/batch_probe")
	parser.add_argument("--probe-checkpoint-dir", type=str, default="experiments_tmp/batch_probe_ckpts")
	parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--wandb-project", type=str, default="ece405-basics")
	parser.add_argument("--wandb-entity", type=str, default="")

	parser.add_argument("--tokenizer-vocab", type=str, default="ece405_basics/bpe_output/vocab.json")
	parser.add_argument("--tokenizer-merges", type=str, default="ece405_basics/bpe_output/merges.json")
	parser.add_argument("--end-of-text-token", type=str, default="<|endoftext|>")

	args = parser.parse_args()
	if isinstance(args.lrs, str):
		args.lrs = _parse_lr_list(args.lrs)
	return args


def main() -> None:
	args = parse_args()
	trial_results, best_results, max_batch_size = run_sweep(args)
	trial_csv, best_csv, md_path = write_summary(
		trial_results=trial_results,
		best_results=best_results,
		out_dir=Path(args.experiment_dir),
		learning_rates=args.lrs,
		max_batch_size=max_batch_size,
	)

	print("\nBatch-size sweep completed.")
	print(f"- Per-trial CSV: {trial_csv}")
	print(f"- Best-per-batch CSV: {best_csv}")
	print(f"- Markdown summary (includes findings): {md_path}")


if __name__ == "__main__":
	main()
