from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


def _iso_utc_now() -> str:
	return datetime.now(tz=timezone.utc).isoformat()


def _sanitize_run_name(name: str) -> str:
	cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name.strip())
	cleaned = cleaned.strip("_")
	return cleaned or "run"


@dataclass
class ExperimentPaths:
	run_dir: Path
	config_json: Path
	metrics_jsonl: Path
	loss_curves_csv: Path
	summary_json: Path


class ExperimentLogger:
	def __init__(self, experiment_dir: str | Path, run_name: str, config: dict[str, Any] | None = None):
		experiment_root = Path(experiment_dir)
		experiment_root.mkdir(parents=True, exist_ok=True)

		timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
		safe_name = _sanitize_run_name(run_name)
		base_run_dir = experiment_root / f"{safe_name}-{timestamp}"
		run_dir = base_run_dir
		for attempt in range(1_000):
			try:
				run_dir.mkdir(parents=True, exist_ok=False)
				break
			except FileExistsError:
				run_dir = experiment_root / f"{base_run_dir.name}-{attempt + 1:03d}"
		else:
			raise RuntimeError(f"Unable to create unique run directory under {experiment_root}")

		self.paths = ExperimentPaths(
			run_dir=run_dir,
			config_json=run_dir / "config.json",
			metrics_jsonl=run_dir / "metrics.jsonl",
			loss_curves_csv=run_dir / "loss_curves.csv",
			summary_json=run_dir / "summary.json",
		)
		self._start = perf_counter()
		self._closed = False

		self._metrics_fp = self.paths.metrics_jsonl.open("w", encoding="utf-8")
		self._curve_fp = self.paths.loss_curves_csv.open("w", newline="", encoding="utf-8")
		self._curve_writer = csv.DictWriter(
			self._curve_fp,
			fieldnames=["gradient_step", "wallclock_time_s", "split", "loss"],
		)
		self._curve_writer.writeheader()

		self._summary: dict[str, Any] = {
			"started_at": _iso_utc_now(),
			"run_dir": str(self.paths.run_dir),
			"best_val_loss": None,
			"best_val_step": None,
			"last_step": 0,
			"total_wallclock_time_s": 0.0,
		}

		config_payload = {
			"created_at": _iso_utc_now(),
			"run_dir": str(self.paths.run_dir),
			"config": config or {},
		}
		self.paths.config_json.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

	@property
	def run_dir(self) -> Path:
		return self.paths.run_dir

	def elapsed_seconds(self) -> float:
		return perf_counter() - self._start

	def log_metrics(self, gradient_step: int, metrics: dict[str, Any], split: str) -> None:
		if self._closed:
			raise RuntimeError("Cannot log after close().")

		wallclock_time_s = self.elapsed_seconds()
		record: dict[str, Any] = {
			"timestamp": _iso_utc_now(),
			"gradient_step": int(gradient_step),
			"wallclock_time_s": float(wallclock_time_s),
			"split": split,
		}
		record.update(metrics)

		self._metrics_fp.write(json.dumps(record) + "\n")
		self._metrics_fp.flush()

		if "loss" in metrics:
			self._curve_writer.writerow(
				{
					"gradient_step": int(gradient_step),
					"wallclock_time_s": float(wallclock_time_s),
					"split": split,
					"loss": float(metrics["loss"]),
				}
			)
			self._curve_fp.flush()

		if split == "val" and "loss" in metrics:
			val_loss = float(metrics["loss"])
			best_val = self._summary["best_val_loss"]
			if best_val is None or val_loss < best_val:
				self._summary["best_val_loss"] = val_loss
				self._summary["best_val_step"] = int(gradient_step)

		self._summary["last_step"] = int(gradient_step)
		self._summary["total_wallclock_time_s"] = float(wallclock_time_s)

	def close(self) -> None:
		if self._closed:
			return

		self._summary["finished_at"] = _iso_utc_now()
		self.paths.summary_json.write_text(json.dumps(self._summary, indent=2), encoding="utf-8")

		self._metrics_fp.close()
		self._curve_fp.close()
		self._closed = True

	def __enter__(self) -> "ExperimentLogger":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()
