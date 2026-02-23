from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
	vocab_size: int
	context_length: int
	num_layers: int
	d_model: int
	num_heads: int
	d_ff: int


@dataclass(frozen=True)
class MatmulBreakdownItem:
	description: str
	m: int
	n: int
	p: int
	count: int
	flops_each: int
	total_flops: int


def count_parameters(config: TransformerConfig) -> int:
	if config.d_model % config.num_heads != 0:
		raise ValueError("d_model must be divisible by num_heads.")

	embedding_params = config.vocab_size * config.d_model

	per_layer_attention = 4 * config.d_model * config.d_model
	per_layer_ffn = 3 * config.d_model * config.d_ff
	per_layer_rmsnorm = 2 * config.d_model
	all_layers = config.num_layers * (per_layer_attention + per_layer_ffn + per_layer_rmsnorm)

	final_rmsnorm = config.d_model
	lm_head = config.vocab_size * config.d_model

	return embedding_params + all_layers + final_rmsnorm + lm_head


def model_memory_bytes(parameter_count: int, bytes_per_parameter: int = 4) -> int:
	if bytes_per_parameter <= 0:
		raise ValueError("bytes_per_parameter must be positive.")
	return parameter_count * bytes_per_parameter


def bytes_to_gib(num_bytes: int) -> float:
	return num_bytes / (1024 ** 3)


def bytes_to_gb(num_bytes: int) -> float:
	return num_bytes / (10 ** 9)


def _matmul_flops(m: int, n: int, p: int) -> int:
	return 2 * m * n * p


def forward_pass_matmul_flops_breakdown(
	config: TransformerConfig,
	seq_len: int | None = None,
	batch_size: int = 1,
) -> list[MatmulBreakdownItem]:
	if config.d_model % config.num_heads != 0:
		raise ValueError("d_model must be divisible by num_heads.")
	if batch_size <= 0:
		raise ValueError("batch_size must be positive.")

	T = config.context_length if seq_len is None else seq_len
	if T <= 0:
		raise ValueError("seq_len must be positive.")

	d = config.d_model
	H = config.num_heads
	d_h = d // H
	d_ff = config.d_ff
	V = config.vocab_size
	L = config.num_layers
	B = batch_size

	items: list[MatmulBreakdownItem] = []

	def add_item(description: str, m: int, n: int, p: int, count: int) -> None:
		flops_each = _matmul_flops(m, n, p)
		items.append(
			MatmulBreakdownItem(
				description=description,
				m=m,
				n=n,
				p=p,
				count=count,
				flops_each=flops_each,
				total_flops=flops_each * count,
			)
		)

	add_item(
		description="Q, K, V projections: (B*T x d) @ (d x d)",
		m=B * T,
		n=d,
		p=d,
		count=3 * L,
	)
	add_item(
		description="Attention output projection: (B*T x d) @ (d x d)",
		m=B * T,
		n=d,
		p=d,
		count=L,
	)
	add_item(
		description="Attention scores QK^T per head: (B*T x d_h) @ (d_h x T)",
		m=B * T,
		n=d_h,
		p=T,
		count=L * H,
	)
	add_item(
		description="Attention-weighted values per head: (B*T x T) @ (T x d_h)",
		m=B * T,
		n=T,
		p=d_h,
		count=L * H,
	)
	add_item(
		description="SwiGLU w1: (B*T x d) @ (d x d_ff)",
		m=B * T,
		n=d,
		p=d_ff,
		count=L,
	)
	add_item(
		description="SwiGLU w3: (B*T x d) @ (d x d_ff)",
		m=B * T,
		n=d,
		p=d_ff,
		count=L,
	)
	add_item(
		description="SwiGLU w2: (B*T x d_ff) @ (d_ff x d)",
		m=B * T,
		n=d_ff,
		p=d,
		count=L,
	)
	add_item(
		description="Final LM head: (B*T x d) @ (d x vocab_size)",
		m=B * T,
		n=d,
		p=V,
		count=1,
	)

	return items


def total_forward_pass_matmul_flops(
	config: TransformerConfig,
	seq_len: int | None = None,
	batch_size: int = 1,
) -> int:
	items = forward_pass_matmul_flops_breakdown(
		config=config,
		seq_len=seq_len,
		batch_size=batch_size,
	)
	return sum(item.total_flops for item in items)


def gpt2_reference_configs(context_length: int = 1024) -> dict[str, TransformerConfig]:
	return {
		"gpt2-small": TransformerConfig(
			vocab_size=50257,
			context_length=context_length,
			num_layers=12,
			d_model=768,
			num_heads=12,
			d_ff=3072,
		),
		"gpt2-medium": TransformerConfig(
			vocab_size=50257,
			context_length=context_length,
			num_layers=24,
			d_model=1024,
			num_heads=16,
			d_ff=4096,
		),
		"gpt2-large": TransformerConfig(
			vocab_size=50257,
			context_length=context_length,
			num_layers=36,
			d_model=1280,
			num_heads=20,
			d_ff=5120,
		),
		"gpt2-xl": TransformerConfig(
			vocab_size=50257,
			context_length=context_length,
			num_layers=48,
			d_model=1600,
			num_heads=25,
			d_ff=6400,
		),
	}


def forward_pass_matmul_flops_proportions(
	config: TransformerConfig,
	seq_len: int | None = None,
	batch_size: int = 1,
) -> list[tuple[str, int, float]]:
	items = forward_pass_matmul_flops_breakdown(
		config=config,
		seq_len=seq_len,
		batch_size=batch_size,
	)
	total = sum(item.total_flops for item in items)
	if total == 0:
		return []

	return [
		(item.description, item.total_flops, item.total_flops / total)
		for item in items
	]


def compare_context_lengths_flops(
	config: TransformerConfig,
	base_context_length: int,
	new_context_length: int,
	batch_size: int = 1,
) -> dict[str, object]:
	base_total = total_forward_pass_matmul_flops(
		config=config,
		seq_len=base_context_length,
		batch_size=batch_size,
	)
	new_total = total_forward_pass_matmul_flops(
		config=config,
		seq_len=new_context_length,
		batch_size=batch_size,
	)

	base_props = {
		description: proportion
		for description, _, proportion in forward_pass_matmul_flops_proportions(
			config=config,
			seq_len=base_context_length,
			batch_size=batch_size,
		)
	}
	new_props = {
		description: proportion
		for description, _, proportion in forward_pass_matmul_flops_proportions(
			config=config,
			seq_len=new_context_length,
			batch_size=batch_size,
		)
	}

	proportion_delta = {
		description: new_props[description] - base_props[description]
		for description in base_props
	}

	return {
		"base_context_length": base_context_length,
		"new_context_length": new_context_length,
		"base_total_flops": base_total,
		"new_total_flops": new_total,
		"flops_ratio": new_total / base_total,
		"base_proportions": base_props,
		"new_proportions": new_props,
		"proportion_delta": proportion_delta,
	}


if __name__ == "__main__":
	gpt2_configs = gpt2_reference_configs(context_length=1024)
	gpt2_xl = gpt2_configs["gpt2-xl"]

	params = count_parameters(gpt2_xl)
	memory_bytes = model_memory_bytes(params, bytes_per_parameter=4)
	flop_items = forward_pass_matmul_flops_breakdown(gpt2_xl)
	total_flops = sum(item.total_flops for item in flop_items)
	expected_total_flops = 4_513_336_524_800

	print(f"Parameters: {params:,}")
	print(f"FP32 memory (bytes): {memory_bytes:,}")
	print(f"FP32 memory (GiB): {bytes_to_gib(memory_bytes):.4f}")
	print(f"FP32 memory (GB): {bytes_to_gb(memory_bytes):.4f}")
	print("Forward matmul FLOPs breakdown:")
	for item in flop_items:
		print(f"- {item.description} | count={item.count:,} | FLOPs={item.total_flops:,}")
	print(f"Total forward matmul FLOPs: {total_flops:,}")
	print(f"Matches expected total: {total_flops == expected_total_flops}")

	print("\nGPT-2 small/medium/large FLOPs proportions:")
	for model_name in ("gpt2-small", "gpt2-medium", "gpt2-large"):
		cfg = gpt2_configs[model_name]
		total_model_flops = total_forward_pass_matmul_flops(cfg)
		print(f"{model_name} total FLOPs: {total_model_flops:,}")
		for description, flops, proportion in forward_pass_matmul_flops_proportions(cfg):
			print(f"- {description} | FLOPs={flops:,} | proportion={100 * proportion:.4f}%")

	print("\nGPT-2 XL context scaling (1024 -> 16384):")
	xl_context_report = compare_context_lengths_flops(
		config=gpt2_xl,
		base_context_length=1024,
		new_context_length=16384,
	)
	print(f"Base total FLOPs: {xl_context_report['base_total_flops']:,}")
	print(f"New total FLOPs: {xl_context_report['new_total_flops']:,}")
	print(f"Total FLOPs multiplier: {xl_context_report['flops_ratio']:.4f}x")

	base_props = xl_context_report["base_proportions"]
	new_props = xl_context_report["new_proportions"]
	for description in base_props:
		print(
			"- "
			f"{description} | "
			f"base={100 * base_props[description]:.4f}% | "
			f"new={100 * new_props[description]:.4f}%"
		)
