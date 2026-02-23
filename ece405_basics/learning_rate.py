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
class SweepResult:
	lr: float
	run_dir: str
	best_val_loss: float | None
	final_val_loss: float | None
	best_val_step: int | None
	diverged: bool
	notes: str = ""


def _find_latest_run_dir(experiment_dir: Path, run_name: str) -> Path:
	pattern = f"{run_name}-*"
	candidates = [path for path in experiment_dir.glob(pattern) if path.is_dir()]
	if not candidates:
		raise FileNotFoundError(f"No experiment run directory found for run name: {run_name}")
	return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_val_stats(run_dir: Path) -> tuple[float | None, bool]:
	metrics_path = run_dir / "metrics.jsonl"
	if not metrics_path.exists():
		return None, True

	final_val_loss: float | None = None
	diverged = False
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


def run_sweep(args: argparse.Namespace) -> list[SweepResult]:
	train_data = Path(args.train_data)
	val_data = Path(args.val_data)
	experiment_dir = Path(args.experiment_dir)
	checkpoint_dir = Path(args.checkpoint_dir)

	experiment_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	if args.batch_size <= 0 or args.context_length <= 0:
		raise ValueError("batch_size and context_length must be positive")

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

	results: list[SweepResult] = []
	for lr in args.lrs:
		run_slug = _lr_slug(lr)
		run_name = f"lr-sweep-{run_slug}"
		checkpoint_path = checkpoint_dir / f"ckpt_lr_{run_slug}.pt"

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
			run_name,
		]

		print(f"\n=== Running lr={lr:.3e} ({run_name}) ===")
		diverged = False
		notes = ""
		try:
			subprocess.run(cmd, check=True)
		except subprocess.CalledProcessError as exc:
			diverged = True
			notes = f"training process failed with exit code {exc.returncode}"

		run_dir = _find_latest_run_dir(experiment_dir, run_name)
		summary_path = run_dir / "summary.json"
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
		best_val = summary.get("best_val_loss")
		best_step = summary.get("best_val_step")
		if isinstance(best_val, (int, float)) and not math.isfinite(float(best_val)):
			diverged = True

		final_val, metrics_diverged = _extract_val_stats(run_dir)
		diverged = diverged or metrics_diverged

		result = SweepResult(
			lr=lr,
			run_dir=str(run_dir),
			best_val_loss=float(best_val) if isinstance(best_val, (int, float)) else None,
			final_val_loss=final_val,
			best_val_step=int(best_step) if isinstance(best_step, int) else None,
			diverged=diverged,
			notes=notes,
		)
		results.append(result)

		status = "diverged" if result.diverged else "ok"
		print(
			f"Result lr={lr:.3e}: status={status} "
			f"final_val={result.final_val_loss} best_val={result.best_val_loss}"
		)

	return results


def write_summary(results: list[SweepResult], out_dir: Path) -> tuple[Path, Path]:
	out_dir.mkdir(parents=True, exist_ok=True)
	stamp = _iso_stamp()
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
		"|---:|:------:|---------------:|--------------:|----------:|:--------|",
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
