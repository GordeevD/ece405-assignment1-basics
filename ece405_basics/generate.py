from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .decoding import generate
from .tokenizer import Tokenizer
from .training_together import parse_torch_dtype, resolve_device
from .transformer_lm import TransformerLM


@dataclass
class ModelConfig:
	vocab_size: int
	context_length: int
	d_model: int
	num_layers: int
	num_heads: int
	d_ff: int
	rope_theta: float
	dtype: str = "float32"


def create_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint.")

	parser.add_argument("--checkpoint-path", type=str, default="experiments_tmp/ckpt_lr_1e-03.pt")
	parser.add_argument("--config-path", type=str, default="")
	parser.add_argument("--tokenizer-vocab", type=str, default="ece405_basics/bpe_output/vocab.json")
	parser.add_argument("--tokenizer-merges", type=str, default="ece405_basics/bpe_output/merges.json")

	parser.add_argument("--prompt", type=str, default="Once upon a time")
	parser.add_argument("--max-new-tokens", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.9)
	parser.add_argument("--top-p", type=float, default=0.95)
	parser.add_argument("--seed", type=int, default=405)

	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--output-path", type=str, default="")

	parser.add_argument("--vocab-size", type=int, default=0)
	parser.add_argument("--context-length", type=int, default=0)
	parser.add_argument("--d-model", type=int, default=0)
	parser.add_argument("--num-layers", type=int, default=0)
	parser.add_argument("--num-heads", type=int, default=0)
	parser.add_argument("--d-ff", type=int, default=0)
	parser.add_argument("--rope-theta", type=float, default=0.0)
	parser.add_argument("--dtype", type=str, default="")

	return parser


def _try_extract_config_object(path: Path) -> dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(obj, dict) and "config" in obj and isinstance(obj["config"], dict):
		return obj["config"]
	if isinstance(obj, dict):
		return obj
	raise ValueError(f"Unsupported config format in {path}")


def _find_config_for_checkpoint(checkpoint_path: Path, search_root: Path) -> Path | None:
	if not search_root.exists():
		return None

	checkpoint_name = checkpoint_path.name
	for config_path in search_root.glob("**/config.json"):
		try:
			config_obj = _try_extract_config_object(config_path)
		except Exception:
			continue

		checkpoint_ref = config_obj.get("checkpoint_path")
		if not isinstance(checkpoint_ref, str):
			continue

		if Path(checkpoint_ref).name == checkpoint_name:
			return config_path

	return None


def _require_positive(value: int | float, name: str) -> None:
	if value <= 0:
		raise ValueError(f"{name} must be > 0, got {value}")


def load_model_config(args: argparse.Namespace, root: Path) -> tuple[ModelConfig, Path | None]:
	config_path: Path | None = None
	config_obj: dict[str, Any] = {}

	if args.config_path:
		config_path = (root / args.config_path).resolve() if not Path(args.config_path).is_absolute() else Path(args.config_path)
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {config_path}")
		config_obj = _try_extract_config_object(config_path)
	else:
		checkpoint_path = (root / args.checkpoint_path).resolve() if not Path(args.checkpoint_path).is_absolute() else Path(args.checkpoint_path)
		candidate = _find_config_for_checkpoint(checkpoint_path=checkpoint_path, search_root=root / "experiments")
		if candidate is not None:
			config_path = candidate
			config_obj = _try_extract_config_object(candidate)

	def pick_int(arg_value: int, key: str) -> int:
		if arg_value > 0:
			return arg_value
		value = config_obj.get(key)
		if isinstance(value, int) and value > 0:
			return value
		if isinstance(value, float) and value.is_integer() and value > 0:
			return int(value)
		raise ValueError(f"Missing required model hyperparameter '{key}'. Provide --{key.replace('_', '-')} or config.")

	def pick_float(arg_value: float, key: str) -> float:
		if arg_value > 0:
			return float(arg_value)
		value = config_obj.get(key)
		if isinstance(value, (int, float)) and float(value) > 0:
			return float(value)
		raise ValueError(f"Missing required model hyperparameter '{key}'. Provide --{key.replace('_', '-')} or config.")

	def pick_dtype(arg_value: str, key: str, default: str = "float32") -> str:
		if arg_value.strip():
			return arg_value.strip()
		value = config_obj.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
		return default

	model_config = ModelConfig(
		vocab_size=pick_int(args.vocab_size, "vocab_size"),
		context_length=pick_int(args.context_length, "context_length"),
		d_model=pick_int(args.d_model, "d_model"),
		num_layers=pick_int(args.num_layers, "num_layers"),
		num_heads=pick_int(args.num_heads, "num_heads"),
		d_ff=pick_int(args.d_ff, "d_ff"),
		rope_theta=pick_float(args.rope_theta, "rope_theta"),
		dtype=pick_dtype(args.dtype, "dtype"),
	)

	_require_positive(model_config.vocab_size, "vocab_size")
	_require_positive(model_config.context_length, "context_length")
	_require_positive(model_config.d_model, "d_model")
	_require_positive(model_config.num_layers, "num_layers")
	_require_positive(model_config.num_heads, "num_heads")
	_require_positive(model_config.d_ff, "d_ff")
	_require_positive(model_config.rope_theta, "rope_theta")

	return model_config, config_path


def _build_quality_factors(model_config: ModelConfig, temperature: float, top_p: float, max_new_tokens: int) -> list[str]:
	factors: list[str] = []
	factors.append(
		f"Training compute/model capacity: quality depends on tokens seen during training and model size (here d_model={model_config.d_model}, layers={model_config.num_layers})."
	)
	factors.append(
		f"Decoding stochasticity: temperature={temperature:.2f} and top_p={top_p:.2f} control coherence-vs-creativity tradeoff."
	)
	factors.append(
		f"Generation length: longer continuations (max_new_tokens={max_new_tokens}) increase risk of drift/repetition as errors compound."
	)
	return factors


def _simple_fluency_comment(text: str, token_count: int, stopped_on_eot: bool) -> str:
	words = text.split()
	avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0
	alpha_fraction = (sum(ch.isalpha() or ch.isspace() or ch in ".,!?;:'\"-" for ch in text) / max(len(text), 1))
	replacement_char_count = text.count("�")

	if replacement_char_count > 0 or alpha_fraction < 0.75:
		base = "Output is weakly fluent: fragments resemble English, but there is substantial noise and incoherence."
	elif len(words) >= 40 and avg_word_len >= 3.0:
		base = "Output is generally fluent and mostly English-like with readable sentence structure."
	elif len(words) >= 20:
		base = "Output is partially fluent but includes noticeable awkward phrasing or local incoherence."
	else:
		base = "Output has limited fluency and often breaks coherence quickly."

	ending = "Generation stopped at <|endoftext|>." if stopped_on_eot else "Generation reached the token budget."
	return f"{base} Produced {token_count} new tokens. {ending}"


def main() -> None:
	args = create_arg_parser().parse_args()
	root = Path(__file__).resolve().parent.parent

	if args.max_new_tokens <= 0:
		raise ValueError("--max-new-tokens must be > 0")
	if args.top_p <= 0 or args.top_p > 1:
		raise ValueError("--top-p must be in (0, 1]")

	checkpoint_path = (root / args.checkpoint_path).resolve() if not Path(args.checkpoint_path).is_absolute() else Path(args.checkpoint_path)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

	tokenizer_vocab_path = (root / args.tokenizer_vocab).resolve() if not Path(args.tokenizer_vocab).is_absolute() else Path(args.tokenizer_vocab)
	tokenizer_merges_path = (root / args.tokenizer_merges).resolve() if not Path(args.tokenizer_merges).is_absolute() else Path(args.tokenizer_merges)
	if not tokenizer_vocab_path.exists():
		raise FileNotFoundError(f"Tokenizer vocab file not found: {tokenizer_vocab_path}")
	if not tokenizer_merges_path.exists():
		raise FileNotFoundError(f"Tokenizer merges file not found: {tokenizer_merges_path}")

	device = resolve_device(args.device)
	model_config, config_path = load_model_config(args=args, root=root)
	torch_dtype = parse_torch_dtype(model_config.dtype)

	torch.manual_seed(args.seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(args.seed)

	tokenizer = Tokenizer.from_files(
		vocab_filepath=str(tokenizer_vocab_path),
		merges_filepath=str(tokenizer_merges_path),
		special_tokens=["<|endoftext|>"],
	)

	model = TransformerLM(
		vocab_size=model_config.vocab_size,
		context_length=model_config.context_length,
		d_model=model_config.d_model,
		num_layers=model_config.num_layers,
		num_heads=model_config.num_heads,
		d_ff=model_config.d_ff,
		rope_theta=model_config.rope_theta,
		device=device,
		dtype=torch_dtype,
	)

	checkpoint = torch.load(checkpoint_path, map_location=device)
	if "model_state_dict" not in checkpoint:
		raise ValueError("Checkpoint missing key 'model_state_dict'")
	
	# Handle torch.compile prefix (_orig_mod.) if present
	state_dict = checkpoint["model_state_dict"]
	if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
		state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
	
	model.load_state_dict(state_dict)
	model.eval()

	prompt_token_ids = tokenizer.encode(args.prompt)
	eot_id = tokenizer.bytes_to_id.get(b"<|endoftext|>")
	generator = torch.Generator(device=device)
	generator.manual_seed(args.seed)

	all_token_ids = generate(
		model=model,
		prompt_token_ids=prompt_token_ids,
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		end_of_text_token_id=eot_id,
		device=device,
		generator=generator,
	)

	generated_ids = all_token_ids[len(prompt_token_ids) :]
	stopped_on_eot = bool(generated_ids) and (eot_id is not None) and generated_ids[-1] == eot_id

	generated_text = tokenizer.decode(all_token_ids)
	fluency_comment = _simple_fluency_comment(text=generated_text, token_count=len(generated_ids), stopped_on_eot=stopped_on_eot)
	quality_factors = _build_quality_factors(
		model_config=model_config,
		temperature=args.temperature,
		top_p=args.top_p,
		max_new_tokens=args.max_new_tokens,
	)

	lines: list[str] = []
	lines.append("=== Generation Report ===")
	lines.append(f"checkpoint: {checkpoint_path}")
	lines.append(f"config: {config_path if config_path is not None else 'N/A (CLI overrides only)'}")
	lines.append(f"device: {device}")
	lines.append(f"prompt: {args.prompt!r}")
	lines.append(f"new_tokens: {len(generated_ids)}")
	lines.append(f"temperature: {args.temperature}")
	lines.append(f"top_p: {args.top_p}")
	lines.append("")
	lines.append("--- Text dump ---")
	lines.append(generated_text)
	lines.append("")
	lines.append("--- Fluency comment ---")
	lines.append(fluency_comment)
	lines.append("")
	lines.append("--- Factors affecting output quality ---")
	for idx, factor in enumerate(quality_factors, start=1):
		lines.append(f"{idx}. {factor}")

	report = "\n".join(lines)
	print(report)

	if args.output_path:
		output_path = (root / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(report, encoding="utf-8")
		print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
	main()
