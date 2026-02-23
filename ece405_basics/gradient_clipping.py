from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
	r"""Clip parameter gradients in-place to have global l2 norm at most ``max_l2_norm``.

	Uses :math:`\epsilon = 10^{-6}` in the denominator, matching PyTorch's default.
	"""
	eps = 1e-6
	parameters_with_grad = [p for p in parameters if p.grad is not None]
	if not parameters_with_grad:
		return

	total_norm_sq = 0.0
	for parameter in parameters_with_grad:
		grad = parameter.grad.detach()
		total_norm_sq += torch.sum(grad * grad).item()

	total_norm = total_norm_sq ** 0.5
	clip_coef = min(1.0, max_l2_norm / (total_norm + eps))

	if clip_coef < 1.0:
		for parameter in parameters_with_grad:
			parameter.grad.mul_(clip_coef)
