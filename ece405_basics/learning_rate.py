from __future__ import annotations

import argparse
import csv
import concurrent.futures
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer
from .training_together import load_token_memmap


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
class SweepResult:
	lr: float
	run_dir: str
	best_val_loss: float | None
	final_val_loss: float | None
	best_val_step: int | None
	diverged: bool
	notes: str = ""


@dataclass(frozen=True)
class TrialSpec:
	index: int
	lr: float
	run_name: str


def _find_latest_run_dir(experiment_dir: Path, run_name: str) -> Path:
	pattern = f"{run_name}-*"
	candidates = [path for path in experiment_dir.glob(pattern) if path.is_dir()]
	if not candidates:
		raise FileNotFoundError(f"No experiment run directory found for run name: {run_name}")
	return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_val_stats(run_dir: Path) -> tuple[float | None, bool]:
	"""Extract final validation loss from ExperimentLogger's summary.json."""
	summary_path = run_dir / "summary.json"
	if not summary_path.exists():
		return None, True

	try:
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
		best_val_loss = summary.get("best_val_loss")
		
		# Check if loss is finite
		if isinstance(best_val_loss, (int, float)) and not math.isfinite(float(best_val_loss)):
			return None, True
		
		return float(best_val_loss) if isinstance(best_val_loss, (int, float)) else None, False
	except (json.JSONDecodeError, FileNotFoundError):
		return None, True


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


def _run_single_trial(
	trial: TrialSpec,
	*,
	args: argparse.Namespace,
	train_data: Path,
	val_data: Path,
	experiment_dir: Path,
	checkpoint_dir: Path,
	max_iters: int,
	warmup_iters: int,
) -> SweepResult:
	run_slug = _lr_slug(trial.lr)
	checkpoint_path = checkpoint_dir / f"ckpt_lr_{run_slug}_{trial.index:03d}.pt"

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
		str(args.batch_size),
		"--max-iters",
		str(max_iters),
		"--lr-max",
		str(trial.lr),
		"--lr-min",
		str(trial.lr * args.lr_min_ratio),
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
		str(args.log_interval),
		"--eval-interval",
		str(args.eval_interval),
		"--eval-batches",
		str(args.eval_batches),
		"--checkpoint-path",
		str(checkpoint_path),
		"--checkpoint-interval",
		str(args.checkpoint_interval),
		"--experiment-dir",
		str(experiment_dir),
		"--run-name",
		trial.run_name,
		"--use-wandb",
		"true" if args.use_wandb else "false",
		"--wandb-project",
		args.wandb_project,
		"--wandb-run-name",
		trial.run_name,
	]
	if args.wandb_entity:
		cmd.extend(["--wandb-entity", args.wandb_entity])

	diverged = False
	notes = ""
	try:
		subprocess.run(cmd, check=True)
	except subprocess.CalledProcessError as exc:
		diverged = True
		notes = f"training process failed with exit code {exc.returncode}"

	try:
		run_dir = _find_latest_run_dir(experiment_dir, trial.run_name)
	except FileNotFoundError:
		return SweepResult(
			lr=trial.lr,
			run_dir="N/A",
			best_val_loss=None,
			final_val_loss=None,
			best_val_step=None,
			diverged=True,
			notes=(notes + "; no run directory found").strip("; "),
		)

	summary_path = run_dir / "summary.json"

	best_val_loss = None
	best_val_step = None
	if summary_path.exists():
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
		best_val_loss = summary.get("best_val_loss")
		best_val_step = summary.get("best_val_step")
		if isinstance(best_val_loss, (int, float)) and not math.isfinite(float(best_val_loss)):
			diverged = True

	final_val_loss, metrics_diverged = _extract_val_stats(run_dir)
	diverged = diverged or metrics_diverged

	return SweepResult(
		lr=trial.lr,
		run_dir=str(run_dir),
		best_val_loss=float(best_val_loss) if isinstance(best_val_loss, (int, float)) else None,
		final_val_loss=final_val_loss,
		best_val_step=int(best_val_step) if isinstance(best_val_step, int) else None,
		diverged=diverged,
		notes=notes,
	)


def run_sweep(args: argparse.Namespace) -> list[SweepResult]:
	train_data = Path(args.train_data)
	val_data = Path(args.val_data)
	experiment_dir = Path(args.experiment_dir)
	checkpoint_dir = Path(args.checkpoint_dir)

	experiment_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	if args.batch_size <= 0 or args.context_length <= 0:
		raise ValueError("batch_size and context_length must be positive")

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

	tokens_per_iter = args.batch_size * args.context_length
	max_iters = args.token_budget // tokens_per_iter
	if max_iters <= 0:
		raise ValueError(
			"token_budget is too small for the chosen batch_size and context_length: "
			f"{args.token_budget=} {tokens_per_iter=}"
		)

	warmup_iters = max(1, int(max_iters * args.warmup_frac))

	print(
		f"Sweep setup: token_budget={args.token_budget:,}, tokens/iter={tokens_per_iter:,}, "
		f"max_iters={max_iters:,}, warmup_iters={warmup_iters:,}"
	)

	trial_specs = [
		TrialSpec(index=idx, lr=lr, run_name=f"lr-sweep-{_lr_slug(lr)}-{idx:03d}")
		for idx, lr in enumerate(args.lrs, start=1)
	]

	requested_workers = max(1, int(args.parallel_workers))
	workers = requested_workers
	if workers > 1 and args.device != "cpu":
		print(
			"Parallel workers requested on non-CPU device; parallel mode is only supported when --device=cpu."
		)
		workers = 1

	results_by_index: dict[int, SweepResult] = {}

	def run_and_report(spec: TrialSpec) -> tuple[int, SweepResult]:
		print(f"\n=== Running lr={spec.lr:.3e} ({spec.run_name}) ===")
		result = _run_single_trial(
			trial=spec,
			args=args,
			train_data=train_data,
			val_data=val_data,
			experiment_dir=experiment_dir,
			checkpoint_dir=checkpoint_dir,
			max_iters=max_iters,
			warmup_iters=warmup_iters,
		)
		status = "diverged" if result.diverged else "ok"
		print(
			f"Result lr={spec.lr:.3e}: status={status} "
			f"final_val={result.final_val_loss} best_val={result.best_val_loss}"
		)
		return spec.index, result

	if workers == 1:
		for spec in trial_specs:
			index, result = run_and_report(spec)
			results_by_index[index] = result
	else:
		print(f"Running {len(trial_specs)} trials with {workers} parallel workers.")
		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
			futures = [executor.submit(run_and_report, spec) for spec in trial_specs]
			for future in concurrent.futures.as_completed(futures):
				index, result = future.result()
				results_by_index[index] = result

	results = [results_by_index[idx] for idx in sorted(results_by_index)]

	return results


def write_summary(results: list[SweepResult], out_dir: Path) -> tuple[Path, Path]:
	"""Write sweep results to CSV and Markdown summaries.
	
	Uses ExperimentLogger's timestamp format for consistency.
	"""
	from datetime import datetime, timezone
	
	out_dir.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
	csv_path = out_dir / f"lr_sweep_summary_{stamp}.csv"
	md_path = out_dir / f"lr_sweep_summary_{stamp}.md"

	with csv_path.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.DictWriter(
			fp,
			fieldnames=[
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
		for row in results:
			writer.writerow(
				{
					"lr": f"{row.lr:.8g}",
					"status": "diverged" if row.diverged else "ok",
					"final_val_loss": "" if row.final_val_loss is None else f"{row.final_val_loss:.6f}",
					"best_val_loss": "" if row.best_val_loss is None else f"{row.best_val_loss:.6f}",
					"best_val_step": "" if row.best_val_step is None else str(row.best_val_step),
					"run_dir": row.run_dir,
					"notes": row.notes,
				}
			)

	completed = [row for row in results if not row.diverged and row.final_val_loss is not None]
	best = min(completed, key=lambda row: row.final_val_loss) if completed else None

	lines = [
		"# Learning-rate sweep summary",
		"",
		"| lr | status | final val loss | best val loss | best step | run dir |",
	]
	for row in results:
		status = "diverged" if row.diverged else "ok"
		final_val = "-" if row.final_val_loss is None else f"{row.final_val_loss:.6f}"
		best_val = "-" if row.best_val_loss is None else f"{row.best_val_loss:.6f}"
		best_step = "-" if row.best_val_step is None else str(row.best_val_step)
		lines.append(f"| {row.lr:.8g} | {status} | {final_val} | {best_val} | {best_step} | {row.run_dir} |")

	if best is not None:
		lines.extend(
			[
				"",
				f"Best non-diverged learning rate by final val loss: **{best.lr:.8g}** "
				f"(final val loss={best.final_val_loss:.6f}).",
			]
		)

	md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return csv_path, md_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run a TinyStories learning-rate sweep.")
	parser.add_argument(
		"--train-data",
		type=str,
		# default="tests/fixtures/tinystories_sample_5M.txt",
		default="tests/fixtures/TinyStoriesV2-GPT4-train.txt",
	)
	parser.add_argument(
		"--val-data",
		type=str,
		# default="tests/fixtures/tinystories_sample.txt",
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

	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--token-budget", type=int, default=40_000_000)
	parser.add_argument("--lrs", type=_parse_lr_list, default="1e-4,3e-4,1e-3,2e-3,3e-3")
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
	parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--wandb-project", type=str, default="ece405-basics")
	parser.add_argument("--wandb-entity", type=str, default="")
	parser.add_argument(
		"--parallel-workers",
		type=int,
		default=10,
		help="Number of concurrent LR trials. Parallel mode is enabled only when --device=cpu.",
	)
	parser.add_argument("--tokenizer-vocab", type=str, default="ece405_basics/bpe_output/vocab.json")
	parser.add_argument("--tokenizer-merges", type=str, default="ece405_basics/bpe_output/merges.json")
	parser.add_argument("--end-of-text-token", type=str, default="<|endoftext|>")

	args = parser.parse_args()
	if isinstance(args.lrs, str):
		args.lrs = _parse_lr_list(args.lrs)
	return args


def main() -> None:
	args = parse_args()
	results = run_sweep(args)
	csv_path, md_path = write_summary(results, Path(args.experiment_dir))

	print("\nSweep completed.")
	print(f"- CSV summary: {csv_path}")
	print(f"- Markdown summary: {md_path}")


if __name__ == "__main__":
	main()
