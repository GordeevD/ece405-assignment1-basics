import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
	inputs: Float[Tensor, " ... vocab_size"],
	targets: Int[Tensor, " ..."],
) -> Float[Tensor, ""]:
	max_logits = torch.max(inputs, dim=-1, keepdim=True).values
	shifted_logits = inputs - max_logits

	logsumexp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
	target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

	losses = logsumexp - target_logits
	return losses.mean()
