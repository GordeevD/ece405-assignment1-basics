from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional
import math

import torch


class AdamW(torch.optim.Optimizer):
	def __init__(
		self,
		params: Iterable[torch.nn.Parameter],
		lr: float = 1e-3,
		betas: tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-8,
		weight_decay: float = 1e-2,
	) -> None:
		if lr < 0.0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if eps < 0.0:
			raise ValueError(f"Invalid epsilon value: {eps}")
		beta1, beta2 = betas
		if not 0.0 <= beta1 < 1.0:
			raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
		if not 0.0 <= beta2 < 1.0:
			raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
		if weight_decay < 0.0:
			raise ValueError(f"Invalid weight_decay value: {weight_decay}")

		defaults = {
			"lr": lr,
			"betas": betas,
			"eps": eps,
			"weight_decay": weight_decay,
		}
		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()

		for group in self.param_groups:
			lr = group["lr"]
			beta1, beta2 = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]

			for parameter in group["params"]:
				if parameter.grad is None:
					continue
				grad = parameter.grad
				if grad.is_sparse:
					raise RuntimeError("AdamW does not support sparse gradients")

				state = self.state[parameter]
				if len(state) == 0:
					state["step"] = 0
					state["exp_avg"] = torch.zeros_like(parameter)
					state["exp_avg_sq"] = torch.zeros_like(parameter)

				exp_avg = state["exp_avg"]
				exp_avg_sq = state["exp_avg_sq"]

				state["step"] += 1
				step = state["step"]

				exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

				bias_correction1 = 1.0 - beta1**step
				bias_correction2 = 1.0 - beta2**step

				with torch.no_grad():
					parameter.mul_(1.0 - lr * weight_decay)
					denominator = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
					denominator.add_(eps)
					step_size = lr / bias_correction1
					parameter.addcdiv_(exp_avg, denominator, value=-step_size)

		return loss
