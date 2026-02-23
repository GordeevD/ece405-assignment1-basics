import numpy as np
import torch


def get_batch(
	x: np.ndarray,
	batch_size: int,
	context_length: int,
	device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
	max_start = len(x) - context_length
	starts = np.random.randint(0, max_start, size=batch_size)

	input_batch = np.stack([x[start : start + context_length] for start in starts])
	target_batch = np.stack([x[start + 1 : start + context_length + 1] for start in starts])

	inputs = torch.tensor(input_batch, dtype=torch.long, device=device)
	targets = torch.tensor(target_batch, dtype=torch.long, device=device)
	return inputs, targets
