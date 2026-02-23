import torch


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
	max_vals = torch.max(in_features, dim=dim, keepdim=True).values
	shifted = in_features - max_vals
	exp_shifted = torch.exp(shifted)
	return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)
