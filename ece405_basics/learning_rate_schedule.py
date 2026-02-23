import math


def get_lr_cosine_schedule(
	t: int,
	alpha_max: float,
	alpha_min: float,
	T_w: int,
	T_c: int,
) -> float:
	"""Return the learning rate for a cosine schedule with linear warmup.

	The schedule is:
	- Linear warmup from 0 to ``alpha_max`` over iterations ``0..T_w``.
	- Cosine annealing from ``alpha_max`` to ``alpha_min`` over ``T_w..T_c``.
	- Constant ``alpha_min`` for iterations after ``T_c``.
	"""
	if t <= T_w:
		return alpha_max * t / T_w

	if t <= T_c:
		progress = (t - T_w) / (T_c - T_w)
		return alpha_min + 0.5 * (1 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)

	return alpha_min
