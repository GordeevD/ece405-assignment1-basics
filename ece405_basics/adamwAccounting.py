from dataclasses import dataclass


@dataclass(frozen=True)
class AdamWAccountingConfig:
	vocab_size: int
	context_length: int
	num_layers: int
	d_model: int
	num_heads: int

	@property
	def d_ff(self) -> int:
		return 4 * self.d_model


def _validate_config(config: AdamWAccountingConfig) -> None:
	if config.d_model % config.num_heads != 0:
		raise ValueError("d_model must be divisible by num_heads.")


def parameter_elements(config: AdamWAccountingConfig) -> int:
	_validate_config(config)

	V = config.vocab_size
	L = config.num_layers
	d = config.d_model
	d_ff = config.d_ff

	token_embedding = V * d
	per_layer = (4 * d * d) + (2 * d * d_ff) + (2 * d)
	final_rmsnorm = d
	output_embedding = V * d
	return token_embedding + L * per_layer + final_rmsnorm + output_embedding


def activation_elements(config: AdamWAccountingConfig, batch_size: int) -> int:
	_validate_config(config)
	if batch_size <= 0:
		raise ValueError("batch_size must be positive.")

	B = batch_size
	V = config.vocab_size
	T = config.context_length
	L = config.num_layers
	d = config.d_model
	H = config.num_heads
	d_ff = config.d_ff

	per_layer_rmsnorms = 2 * B * T * d
	per_layer_qkv = 3 * B * T * d
	per_layer_qk = B * H * T * T
	per_layer_softmax = B * H * T * T
	per_layer_weighted_values = B * T * d
	per_layer_attn_output_proj = B * T * d
	per_layer_ff_w1 = B * T * d_ff
	per_layer_silu = B * T * d_ff
	per_layer_ff_w2 = B * T * d

	per_layer_total = (
		per_layer_rmsnorms
		+ per_layer_qkv
		+ per_layer_qk
		+ per_layer_softmax
		+ per_layer_weighted_values
		+ per_layer_attn_output_proj
		+ per_layer_ff_w1
		+ per_layer_silu
		+ per_layer_ff_w2
	)

	final_rmsnorm = B * T * d
	output_embedding_logits = B * T * V
	cross_entropy_on_logits = B * T * V

	return L * per_layer_total + final_rmsnorm + output_embedding_logits + cross_entropy_on_logits


def gradient_elements(config: AdamWAccountingConfig) -> int:
	return parameter_elements(config)


def optimizer_state_elements(config: AdamWAccountingConfig) -> int:
	# AdamW keeps first and second moments (m, v) per parameter.
	return 2 * parameter_elements(config)


def peak_memory_bytes(config: AdamWAccountingConfig, batch_size: int) -> dict[str, int]:
	param_elems = parameter_elements(config)
	act_elems = activation_elements(config, batch_size)
	grad_elems = gradient_elements(config)
	opt_elems = optimizer_state_elements(config)

	bytes_per_float = 4
	result = {
		"parameters_bytes": bytes_per_float * param_elems,
		"activations_bytes": bytes_per_float * act_elems,
		"gradients_bytes": bytes_per_float * grad_elems,
		"optimizer_state_bytes": bytes_per_float * opt_elems,
	}
	result["total_peak_bytes"] = sum(result.values())
	return result


def bytes_to_gib(num_bytes: int) -> float:
	return num_bytes / (1024**3)


def bytes_to_gb(num_bytes: int) -> float:
	return num_bytes / (10**9)


def algebraic_expressions() -> dict[str, str]:
	return {
		"parameters_elements": (
			"P = 2Vd + L(4d^2 + 2dd_ff + 2d) + d, with d_ff = 4d; "
			"equivalently P = 2Vd + 12Ld^2 + (2L + 1)d"
		),
		"activations_elements": (
			"A = L[2BTd + 3BTd + BHT^2 + BHT^2 + BTd + BTd + BTd_ff + BTd_ff + BTd] + BTd + BTV + BTV, "
			"with d_ff = 4d; equivalently A = L(16BTd + 2BHT^2) + BTd + 2BTV"
		),
		"gradients_elements": "G = P",
		"optimizer_state_elements": "O = 2P (AdamW moments m and v)",
		"total_peak_bytes": (
			"M_peak = 4(P + A + G + O) = 4(A + 4P) bytes "
			"(float32, 4 bytes per element)"
		),
	}


def total_peak_memory_linear_bytes(config: AdamWAccountingConfig) -> tuple[int, int]:
	"""Return coefficients (a, b) for M_peak(B) = a * B + b in bytes."""
	_validate_config(config)

	V = config.vocab_size
	T = config.context_length
	L = config.num_layers
	d = config.d_model
	H = config.num_heads
	P = parameter_elements(config)

	activation_elements_per_batch = L * (16 * T * d + 2 * H * T * T) + T * d + 2 * T * V
	a = 4 * activation_elements_per_batch
	b = 16 * P
	return a, b


def max_batch_size_for_memory_limit(config: AdamWAccountingConfig, memory_limit_bytes: int) -> int:
	if memory_limit_bytes <= 0:
		raise ValueError("memory_limit_bytes must be positive.")

	a, b = total_peak_memory_linear_bytes(config)
	if memory_limit_bytes < b:
		return 0
	return (memory_limit_bytes - b) // a


def adamw_step_flops(config: AdamWAccountingConfig) -> int:
	"""Approximate FLOPs for one AdamW optimizer step over all parameters."""
	P = parameter_elements(config)
	return 16 * P


def adamw_step_flops_expression() -> str:
	return (
		"F_AdamW_step = 16P FLOPs, where P is the number of trainable parameters "
		"(ignoring O(1) scalar ops). Per parameter element: "
		"m update (3) + v update (4) + bias corrections m_hat/v_hat (2) + "
		"sqrt+eps (2) + normalization divide (1) + decoupled weight decay and "
		"parameter update (4) = 16 FLOPs."
	)


def training_time_days_from_mfu(
	forward_flops_batch1: int,
	batch_size: int,
	steps: int,
	peak_flops_per_second: float,
	mfu: float,
	backward_to_forward_ratio: float = 2.0,
) -> float:
	if forward_flops_batch1 <= 0:
		raise ValueError("forward_flops_batch1 must be positive.")
	if batch_size <= 0:
		raise ValueError("batch_size must be positive.")
	if steps <= 0:
		raise ValueError("steps must be positive.")
	if peak_flops_per_second <= 0:
		raise ValueError("peak_flops_per_second must be positive.")
	if not 0 < mfu <= 1:
		raise ValueError("mfu must be in (0, 1].")
	if backward_to_forward_ratio < 0:
		raise ValueError("backward_to_forward_ratio must be non-negative.")

	forward_flops_per_step = forward_flops_batch1 * batch_size
	train_flops_per_step = (1.0 + backward_to_forward_ratio) * forward_flops_per_step
	total_training_flops = train_flops_per_step * steps
	effective_flops_per_second = peak_flops_per_second * mfu
	seconds = total_training_flops / effective_flops_per_second
	return seconds / 86_400


def training_time_expression() -> str:
	return (
		"days = [steps * batch_size * F_fwd(B=1) * (1 + r_bw)] / [MFU * F_peak] / 86400, "
		"where r_bw is backward-to-forward FLOPs ratio (r_bw = 2 here)."
	)


if __name__ == "__main__":
	cfg = AdamWAccountingConfig(
		vocab_size=50_257,
		context_length=1_024,
		num_layers=12,
		d_model=768,
		num_heads=12,
	)
	batch_size = 8

	print("AdamW resource accounting (float32)")
	print(f"Config: {cfg}")
	print(f"Batch size: {batch_size}")

	formulas = algebraic_expressions()
	print("\nAlgebraic expressions:")
	for key, value in formulas.items():
		print(f"- {key}: {value}")

	breakdown = peak_memory_bytes(cfg, batch_size=batch_size)
	print("\nMemory breakdown:")
	for key, value in breakdown.items():
		if key.endswith("_bytes"):
			print(f"- {key}: {value:,} bytes ({bytes_to_gib(value):.4f} GiB)")

	gpt2_xl = AdamWAccountingConfig(
		vocab_size=50_257,
		context_length=1_024,
		num_layers=48,
		d_model=1_600,
		num_heads=25,
	)
	memory_limit_80gb = 80_000_000_000
	a, b = total_peak_memory_linear_bytes(gpt2_xl)
	max_batch_size = max_batch_size_for_memory_limit(gpt2_xl, memory_limit_80gb)

	print("\nGPT-2 XL instantiation (80 GB limit):")
	print(f"- M_peak(B) = {a:,} * B + {b:,} bytes")
	print(
		f"- M_peak(B) = {bytes_to_gb(a):.9f} * B + {bytes_to_gb(b):.9f} GB"
	)
	print(f"- Max batch size under 80 GB: {max_batch_size}")

	gpt2_xl_adamw_flops = adamw_step_flops(gpt2_xl)
	print("\nAdamW one-step FLOPs:")
	print(f"- {adamw_step_flops_expression()}")
	print(f"- GPT-2 XL instantiation: F_AdamW_step = {gpt2_xl_adamw_flops:,} FLOPs")

	gpt2_xl_forward_flops_batch1 = 4_513_336_524_800
	training_steps = 400_000
	training_batch_size = 1_024
	a100_fp32_peak_flops = 19.5e12
	mfu = 0.5
	train_days = training_time_days_from_mfu(
		forward_flops_batch1=gpt2_xl_forward_flops_batch1,
		batch_size=training_batch_size,
		steps=training_steps,
		peak_flops_per_second=a100_fp32_peak_flops,
		mfu=mfu,
		backward_to_forward_ratio=2.0,
	)

	print("\nMFU training-time estimate (single A100, GPT-2 XL):")
	print(f"- {training_time_expression()}")
	print(f"- Estimated training time for 400k steps, batch size 1024: {train_days:.2f} days")
