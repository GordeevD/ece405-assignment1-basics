from collections.abc import Callable, Iterable
from typing import Optional
import math

import torch


class SGD(torch.optim.Optimizer):
	def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr}
		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group["lr"]
			for p in group["params"]:
				if p.grad is None:
					continue
				state = self.state[p]
				t = state.get("t", 0)
				grad = p.grad.data
				p.data -= lr / math.sqrt(t + 1) * grad
				state["t"] = t + 1
		return loss


def run_experiment(lr: float, steps: int = 10, seed: int = 405) -> list[float]:
	torch.manual_seed(seed)
	weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
	opt = SGD([weights], lr=lr)

	losses: list[float] = []
	for _ in range(steps):
		opt.zero_grad()
		loss = (weights ** 2).mean()
		losses.append(loss.item())
		loss.backward()
		opt.step()

	return losses


def main() -> None:
	for lr in [1e1, 1e2, 1e3]:
		losses = run_experiment(lr=lr, steps=10)
		print(f"lr={lr:.0e}")
		for step, loss in enumerate(losses):
			print(f"  step {step:02d}: loss={loss:.6e}")


if __name__ == "__main__":
	main()
