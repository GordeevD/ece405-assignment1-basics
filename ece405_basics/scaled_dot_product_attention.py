import math

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def scaled_dot_product_attention(
	Q: Float[Tensor, " ... queries d_k"],
	K: Float[Tensor, " ... keys d_k"],
	V: Float[Tensor, " ... values d_v"],
	mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
	"""
	Compute scaled dot-product attention.

	Args:
		Q (Float[Tensor, "... queries d_k"]): Query tensor.
		K (Float[Tensor, "... keys d_k"]): Key tensor.
		V (Float[Tensor, "... values d_v"]): Value tensor.
		mask (Bool[Tensor, "... queries keys"] | None): Optional boolean mask where
			True entries are valid positions and False entries are masked out.

	Returns:
		Float[Tensor, "... queries d_v"]: Attention output tensor.
	"""
	d_k = Q.shape[-1]
	scale = 1.0 / math.sqrt(d_k)
	scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

	if mask is None:
		attention_probs = torch.softmax(scores, dim=-1)
	else:
		mask = mask.to(device=scores.device, dtype=torch.bool)
		if mask.dim() < scores.dim():
			mask = mask.reshape((1,) * (scores.dim() - mask.dim()) + tuple(mask.shape))
		mask = torch.broadcast_to(mask, scores.shape)

		neg_inf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
		masked_scores = torch.where(mask, scores, neg_inf)

		max_scores = torch.max(masked_scores, dim=-1, keepdim=True).values
		max_scores = torch.where(torch.isfinite(max_scores), max_scores, torch.zeros_like(max_scores))

		exp_scores = torch.exp(masked_scores - max_scores)
		exp_scores = torch.where(mask, exp_scores, torch.zeros_like(exp_scores))
		denom = exp_scores.sum(dim=-1, keepdim=True)

		attention_probs = torch.where(denom > 0, exp_scores / denom, torch.zeros_like(exp_scores))

	return torch.matmul(attention_probs, V)
