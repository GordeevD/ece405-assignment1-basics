from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .adamw import AdamW
from .checkpointing import load_checkpoint, save_checkpoint
from .cross_entropy import cross_entropy
from .data_loading import get_batch
from .experiment_log import ExperimentLogger
from .gradient_clipping import gradient_clipping
from .learning_rate_schedule import get_lr_cosine_schedule
from .transformer_lm import TransformerLM


def str2bool(value: str) -> bool:
	if isinstance(value, bool):
		return value
	normalized = value.strip().lower()
	if normalized in {"1", "true", "t", "yes", "y", "on"}:
		return True
	if normalized in {"0", "false", "f", "no", "n", "off"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
	mapping = {
		"float32": torch.float32,
		"float": torch.float32,
		"fp32": torch.float32,
		"bfloat16": torch.bfloat16,
		"bf16": torch.bfloat16,
		"float16": torch.float16,
		"fp16": torch.float16,
	}
	key = dtype_name.strip().lower()
	if key not in mapping:
		raise ValueError(f"Unsupported dtype: {dtype_name}")
	return mapping[key]


def parse_numpy_dtype(dtype_name: str) -> np.dtype:
	try:
		return np.dtype(dtype_name)
	except TypeError as exc:
		raise ValueError(f"Unsupported numpy dtype: {dtype_name}") from exc


def resolve_device(device_arg: str) -> torch.device:
	if device_arg == "auto":
		if torch.cuda.is_available():
			return torch.device("cuda")
		if torch.backends.mps.is_available():
			return torch.device("mps")
		return torch.device("cpu")
	return torch.device(device_arg)


def load_token_memmap(path: str, dtype: np.dtype) -> np.ndarray:
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"Data file not found: {file_path}")

	if file_path.suffix == ".npy":
		arr = np.load(file_path, mmap_mode="r")
		if arr.dtype != dtype:
			arr = arr.astype(dtype)
		return arr

	return np.memmap(file_path, dtype=dtype, mode="r")


def update_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr


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
	metrics: dict[str, float] = {}

	for split, tokens in (("train", train_tokens), ("val", val_tokens)):
		losses = torch.empty(eval_batches, device="cpu")
		for idx in range(eval_batches):
			xb, yb = get_batch(tokens, batch_size=batch_size, context_length=context_length, device=device)
			logits = model(xb)
			loss = cross_entropy(logits, yb)
			losses[idx] = loss.detach().cpu()
		metrics[split] = float(losses.mean().item())

	model.train()
	return metrics


def create_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train TransformerLM on memmapped token data.")

	parser.add_argument("--train-data", type=str, required=True)
	parser.add_argument("--val-data", type=str, required=True)
	parser.add_argument("--data-dtype", type=str, default="uint16")

	parser.add_argument("--vocab-size", type=int, required=True)
	parser.add_argument("--context-length", type=int, default=128)
	parser.add_argument("--d-model", type=int, default=256)
	parser.add_argument("--num-layers", type=int, default=4)
	parser.add_argument("--num-heads", type=int, default=8)
	parser.add_argument("--d-ff", type=int, default=1024)
	parser.add_argument("--rope-theta", type=float, default=10_000.0)

	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--max-iters", type=int, default=1_000)
	parser.add_argument("--lr-max", type=float, default=3e-4)
	parser.add_argument("--lr-min", type=float, default=3e-5)
	parser.add_argument("--warmup-iters", type=int, default=100)
	parser.add_argument("--cosine-iters", type=int, default=1_000)
	parser.add_argument("--beta1", type=float, default=0.9)
	parser.add_argument("--beta2", type=float, default=0.95)
	parser.add_argument("--eps", type=float, default=1e-8)
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)

	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--dtype", type=str, default="float32")
	parser.add_argument("--seed", type=int, default=405)

	parser.add_argument("--log-interval", type=int, default=10)
	parser.add_argument("--eval-interval", type=int, default=100)
	parser.add_argument("--eval-batches", type=int, default=20)

	parser.add_argument("--checkpoint-path", type=str, required=True)
	parser.add_argument("--checkpoint-interval", type=int, default=100)
	parser.add_argument("--resume-from", type=str, default="")
	parser.add_argument("--experiment-dir", type=str, default="experiments")
	parser.add_argument("--run-name", type=str, default="")

	parser.add_argument("--use-wandb", type=str2bool, default=False)
	parser.add_argument("--wandb-project", type=str, default="ece405-basics")
	parser.add_argument("--wandb-entity", type=str, default="")
	parser.add_argument("--wandb-run-name", type=str, default="")

	return parser


def main() -> None:
	args = create_arg_parser().parse_args()
	if args.eval_interval <= 0:
		raise ValueError("--eval-interval must be > 0 to periodically evaluate validation loss")
	if args.log_interval <= 0:
		raise ValueError("--log-interval must be > 0")

	np_dtype = parse_numpy_dtype(args.data_dtype)
	torch_dtype = parse_torch_dtype(args.dtype)
	device = resolve_device(args.device)
	device_str = str(device)

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	train_tokens = load_token_memmap(args.train_data, np_dtype)
	val_tokens = load_token_memmap(args.val_data, np_dtype)

	min_required_tokens = args.context_length + 1
	if len(train_tokens) < min_required_tokens:
		raise ValueError(
			f"train dataset too small: {len(train_tokens)} tokens, need at least {min_required_tokens}"
		)
	if len(val_tokens) < min_required_tokens:
		raise ValueError(f"val dataset too small: {len(val_tokens)} tokens, need at least {min_required_tokens}")

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
	optimizer = AdamW(
		model.parameters(),
		lr=args.lr_max,
		betas=(args.beta1, args.beta2),
		eps=args.eps,
		weight_decay=args.weight_decay,
	)

	start_iteration = 0
	if args.resume_from:
		start_iteration = load_checkpoint(args.resume_from, model, optimizer)
		print(f"Resumed from {args.resume_from} at iteration {start_iteration}")

	if hasattr(torch, "compile"):
		try:
			if device.type == "cpu":
				model = torch.compile(model)
				print("Enabled torch.compile for CPU training.")
			elif device.type == "mps":
				model = torch.compile(model, backend="aot_eager")
				print("Enabled torch.compile(..., backend='aot_eager') for MPS training.")
		except Exception as exc:
			print(f"Warning: torch.compile setup failed; continuing without compile ({exc}).")

	checkpoint_path = Path(args.checkpoint_path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	run_name = args.run_name or args.wandb_run_name or checkpoint_path.stem
	experiment_logger = ExperimentLogger(
		experiment_dir=args.experiment_dir,
		run_name=run_name,
		config=vars(args),
	)
	print(f"Experiment logs: {experiment_logger.run_dir}")

	wandb_run = None
	if args.use_wandb:
		wandb_module = importlib.import_module("wandb")

		wandb_kwargs = {
			"project": args.wandb_project,
			"config": vars(args),
		}
		if args.wandb_entity:
			wandb_kwargs["entity"] = args.wandb_entity
		if args.wandb_run_name:
			wandb_kwargs["name"] = args.wandb_run_name
		wandb_run = wandb_module.init(**wandb_kwargs)

	model.train()
	t0 = perf_counter()

	try:
		for iteration in range(start_iteration + 1, args.max_iters + 1):
			lr = get_lr_cosine_schedule(
				t=iteration,
				alpha_max=args.lr_max,
				alpha_min=args.lr_min,
				T_w=max(args.warmup_iters, 1),
				T_c=max(args.cosine_iters, args.warmup_iters + 1),
			)
			update_optimizer_lr(optimizer, lr)

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
				wallclock_time_s = experiment_logger.elapsed_seconds()
				train_loss = float(loss.detach().item())
				print(
					f"iter={iteration:6d} train_loss={train_loss:.4f} lr={lr:.3e} "
					f"tok/s={toks_per_sec:.1f} device={device_str}"
				)
				experiment_logger.log_metrics(
					gradient_step=iteration,
					metrics={
						"loss": train_loss,
						"lr": lr,
						"tokens_per_sec": toks_per_sec,
					},
					split="train",
				)
				if wandb_run is not None:
					wandb_run.log(
						{
							"iter": iteration,
							"step": iteration,
							"train/wallclock_time_s": wallclock_time_s,
							"train/loss": train_loss,
							"train/lr": lr,
							"train/tokens_per_sec": toks_per_sec,
						},
						step=iteration,
					)

			if iteration % args.eval_interval == 0 or iteration == args.max_iters:
				wallclock_time_s = experiment_logger.elapsed_seconds()
				metrics = estimate_loss(
					model=model,
					train_tokens=train_tokens,
					val_tokens=val_tokens,
					batch_size=args.batch_size,
					context_length=args.context_length,
					device=device_str,
					eval_batches=args.eval_batches,
				)
				print(f"iter={iteration:6d} eval train_loss={metrics['train']:.4f} val_loss={metrics['val']:.4f}")
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
				if wandb_run is not None:
					wandb_run.log(
						{
							"iter": iteration,
							"step": iteration,
							"eval/wallclock_time_s": wallclock_time_s,
							"eval/train_loss": metrics["train"],
							"eval/val_loss": metrics["val"],
						},
						step=iteration,
					)

			if iteration % args.checkpoint_interval == 0 or iteration == args.max_iters:
				save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=checkpoint_path)
				print(f"Saved checkpoint to {checkpoint_path} at iteration {iteration}")

		save_checkpoint(model=model, optimizer=optimizer, iteration=args.max_iters, out=checkpoint_path)
	finally:
		experiment_logger.close()
		if wandb_run is not None:
			wandb_run.finish()


if __name__ == "__main__":
	main()
