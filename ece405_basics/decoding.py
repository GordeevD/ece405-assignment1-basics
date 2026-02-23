from __future__ import annotations

from typing import Any

import torch


def _sample_next_token(
	logits: torch.Tensor,
	temperature: float = 1.0,
	top_p: float = 1.0,
	generator: torch.Generator | None = None,
) -> int:
	if logits.ndim != 1:
		raise ValueError(f"Expected 1D logits tensor, got shape {tuple(logits.shape)}")

	if temperature <= 0:
		return int(torch.argmax(logits, dim=-1).item())

	if top_p <= 0 or top_p > 1:
		raise ValueError(f"top_p must be in (0, 1], got {top_p}")

	scaled_logits = logits / temperature
	probs = torch.softmax(scaled_logits, dim=-1)

	if top_p < 1.0:
		sorted_probs, sorted_indices = torch.sort(probs, descending=True)
		cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

		sorted_remove_mask = cumulative_probs > top_p
		sorted_remove_mask[0] = False
		sorted_probs = sorted_probs.masked_fill(sorted_remove_mask, 0.0)

		probs = torch.zeros_like(probs)
		probs.scatter_(0, sorted_indices, sorted_probs)

		total_prob = probs.sum()
		if total_prob <= 0:
			return int(torch.argmax(logits, dim=-1).item())
		probs = probs / total_prob

	next_token = torch.multinomial(probs, num_samples=1, generator=generator)
	return int(next_token.item())


@torch.no_grad()
def generate(
	model: torch.nn.Module,
	prompt_token_ids: list[int],
	max_new_tokens: int = 100,
	temperature: float = 1.0,
	top_p: float = 1.0,
	end_of_text_token_id: int | None = None,
	device: torch.device | None = None,
	generator: torch.Generator | None = None,
) -> list[int]:
	if max_new_tokens < 0:
		raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

	if len(prompt_token_ids) == 0:
		raise ValueError("prompt_token_ids must contain at least one token")

	if device is None:
		device = next(model.parameters()).device

	generated_token_ids = list(prompt_token_ids)
	context_length = getattr(model, "context_length", None)

	was_training = model.training
	model.eval()

	for _ in range(max_new_tokens):
		if context_length is None:
			context = generated_token_ids
		else:
			context = generated_token_ids[-context_length:]

		model_input = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
		logits = model(model_input)
		next_logits = logits[0, -1, :]

		next_token_id = _sample_next_token(
			logits=next_logits,
			temperature=temperature,
			top_p=top_p,
			generator=generator,
		)
		generated_token_ids.append(next_token_id)

		if end_of_text_token_id is not None and next_token_id == end_of_text_token_id:
			break

	if was_training:
		model.train()

	return generated_token_ids


def decode(
	model: torch.nn.Module,
	tokenizer: Any,
	prompt: str,
	max_new_tokens: int = 100,
	temperature: float = 1.0,
	top_p: float = 1.0,
	end_of_text_token: str = "<|endoftext|>",
	device: torch.device | None = None,
	generator: torch.Generator | None = None,
) -> str:
	prompt_token_ids = tokenizer.encode(prompt)
	eot_bytes = end_of_text_token.encode("utf-8")
	end_of_text_token_id = getattr(tokenizer, "bytes_to_id", {}).get(eot_bytes)

	generated_token_ids = generate(
		model=model,
		prompt_token_ids=prompt_token_ids,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		top_p=top_p,
		end_of_text_token_id=end_of_text_token_id,
		device=device,
		generator=generator,
	)

	return tokenizer.decode(generated_token_ids)
