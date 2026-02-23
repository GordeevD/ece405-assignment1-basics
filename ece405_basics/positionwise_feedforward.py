import torch
import torch.nn as nn

from .linear import Linear


def _default_d_ff(d_model: int) -> int:
	approx = int((8 * d_model) / 3)
	return ((approx + 63) // 64) * 64


class SwiGLU(nn.Module):
	def __init__(
		self,
		d_model: int,
		d_ff: int | None = None,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()

		if d_ff is None:
			d_ff = _default_d_ff(d_model)

		self.d_model = d_model
		self.d_ff = d_ff

		self.w1 = Linear(
			in_features=d_model,
			out_features=d_ff,
			device=device,
			dtype=dtype,
		)
		self.w2 = Linear(
			in_features=d_ff,
			out_features=d_model,
			device=device,
			dtype=dtype,
		)
		self.w3 = Linear(
			in_features=d_model,
			out_features=d_ff,
			device=device,
			dtype=dtype,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_w1 = self.w1(x)
		x_w3 = self.w3(x)
		gated = (x_w1 * torch.sigmoid(x_w1)) * x_w3
		return self.w2(gated)
