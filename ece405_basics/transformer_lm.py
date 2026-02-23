import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		context_length: int,
		d_model: int,
		num_layers: int,
		num_heads: int,
		d_ff: int,
		rope_theta: float,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None,
	) -> None:
		super().__init__()

		self.context_length = context_length

		self.token_embeddings = Embedding(
			num_embeddings=vocab_size,
			embedding_dim=d_model,
			device=device,
			dtype=dtype,
		)

		self.layers = nn.ModuleList(
			[
				TransformerBlock(
					d_model=d_model,
					num_heads=num_heads,
					d_ff=d_ff,
					max_seq_len=context_length,
					theta=rope_theta,
					device=device,
					dtype=dtype,
				)
				for _ in range(num_layers)
			]
		)

		self.ln_final = RMSNorm(
			d_model=d_model,
			device=device,
			dtype=dtype,
		)
		self.lm_head = Linear(
			in_features=d_model,
			out_features=vocab_size,
			device=device,
			dtype=dtype,
		)

	def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
		seq_len = in_indices.shape[-1]
		if seq_len > self.context_length:
			raise ValueError("Input sequence length exceeds context_length.")

		x = self.token_embeddings(in_indices)
		token_positions = torch.arange(seq_len, device=in_indices.device)
		token_positions = token_positions.expand(*in_indices.shape[:-1], seq_len)

		for layer in self.layers:
			x = layer(x, token_positions=token_positions)

		x = self.ln_final(x)
		return self.lm_head(x)
