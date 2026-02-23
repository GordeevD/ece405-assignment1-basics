import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
	def __init__(
		self,
		theta: float,
		d_k: int,
		max_seq_len: int,
		device: torch.device | None = None,
	) -> None:
		super().__init__()

		if d_k % 2 != 0:
			raise ValueError("d_k must be even for RoPE.")

		self.theta = theta
		self.d_k = d_k
		self.max_seq_len = max_seq_len

		positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
		pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
		inv_freq = 1.0 / (theta ** (pair_indices / d_k))
		angles = torch.einsum("s,d->sd", positions, inv_freq)

		cos = torch.cos(angles)
		sin = torch.sin(angles)

		self.register_buffer("cos", cos, persistent=False)
		self.register_buffer("sin", sin, persistent=False)

	def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
		if x.shape[-1] != self.d_k:
			raise ValueError("Last dimension of x must match d_k.")

		cos = self.cos[token_positions].to(dtype=x.dtype)
		sin = self.sin[token_positions].to(dtype=x.dtype)

		x_even = x[..., 0::2]
		x_odd = x[..., 1::2]

		rot_even = x_even * cos - x_odd * sin
		rot_odd = x_even * sin + x_odd * cos

		return torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x)
