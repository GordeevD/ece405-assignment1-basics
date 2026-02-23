import torch
import torch.nn as nn

from .multihead_self_attention import MultiHeadSelfAttention
from .positionwise_feedforward import SwiGLU
from .rmsnorm import RMSNorm
from .rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		d_ff: int,
		max_seq_len: int,
		theta: float,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()

		d_head = d_model // num_heads
		rope = RotaryPositionalEmbedding(
			theta=theta,
			d_k=d_head,
			max_seq_len=max_seq_len,
			device=device,
		)

		self.attn = MultiHeadSelfAttention(
			d_model=d_model,
			num_heads=num_heads,
			rope=rope,
			device=device,
			dtype=dtype,
		)
		self.ffn = SwiGLU(
			d_model=d_model,
			d_ff=d_ff,
			device=device,
			dtype=dtype,
		)
		self.ln1 = RMSNorm(
			d_model=d_model,
			device=device,
			dtype=dtype,
		)
		self.ln2 = RMSNorm(
			d_model=d_model,
			device=device,
			dtype=dtype,
		)

	def forward(
		self,
		x: torch.Tensor,
		token_positions: torch.Tensor | None = None,
	) -> torch.Tensor:
		x = x + self.attn(self.ln1(x), token_positions=token_positions)
		x = x + self.ffn(self.ln2(x))
		return x
