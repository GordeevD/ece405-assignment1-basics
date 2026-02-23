import torch
import torch.nn as nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		rope: RotaryPositionalEmbedding | None = None,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()

		if d_model % num_heads != 0:
			raise ValueError("d_model must be divisible by num_heads.")

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_head = d_model // num_heads
		self.rope = rope

		self.q_proj = Linear(
			in_features=d_model,
			out_features=d_model,
			device=device,
			dtype=dtype,
		)
		self.k_proj = Linear(
			in_features=d_model,
			out_features=d_model,
			device=device,
			dtype=dtype,
		)
		self.v_proj = Linear(
			in_features=d_model,
			out_features=d_model,
			device=device,
			dtype=dtype,
		)
		self.output_proj = Linear(
			in_features=d_model,
			out_features=d_model,
			device=device,
			dtype=dtype,
		)

	def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
		seq_len = x.shape[-2]
		x = x.reshape(*x.shape[:-1], self.num_heads, self.d_head)
		return x.transpose(-3, -2)

	def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
		x = x.transpose(-3, -2)
		return x.reshape(*x.shape[:-2], self.d_model)

	def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
		return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))

	def _expand_positions(self, token_positions: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
		return token_positions.unsqueeze(-2).expand(*q.shape[:-1])

	def forward(
		self,
		x: torch.Tensor,
		token_positions: torch.Tensor | None = None,
	) -> torch.Tensor:
		seq_len = x.shape[-2]

		q = self._split_heads(self.q_proj(x))
		k = self._split_heads(self.k_proj(x))
		v = self._split_heads(self.v_proj(x))

		if self.rope is not None:
			if token_positions is None:
				base_positions = torch.arange(seq_len, device=x.device)
				token_positions = base_positions.expand(*x.shape[:-2], seq_len)
			rope_positions = self._expand_positions(token_positions, q)
			q = self.rope(q, rope_positions)
			k = self.rope(k, rope_positions)

		mask = self._causal_mask(seq_len=seq_len, device=x.device)
		attention_out = scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
		attention_out = self._merge_heads(attention_out)

		return self.output_proj(attention_out)
