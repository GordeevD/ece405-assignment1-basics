from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from tokenizer import Tokenizer


END_OF_TEXT = "<|endoftext|>"


@dataclass
class CompressionStats:
	num_docs: int
	total_bytes: int
	total_tokens: int

	@property
	def bytes_per_token(self) -> float:
		if self.total_tokens == 0:
			return float("inf")
		return self.total_bytes / self.total_tokens


@dataclass
class ThroughputStats:
	total_bytes: int
	elapsed_seconds: float

	@property
	def bytes_per_second(self) -> float:
		if self.elapsed_seconds == 0.0:
			return float("inf")
		return self.total_bytes / self.elapsed_seconds

	@property
	def megabytes_per_second(self) -> float:
		return self.bytes_per_second / 1e6

	def estimated_hours_for_bytes(self, total_bytes: int) -> float:
		return (total_bytes / self.bytes_per_second) / 3600


def iter_documents(corpus_path: Path, end_token: str = END_OF_TEXT):
	with corpus_path.open("r", encoding="utf-8") as f:
		buffer = ""
		while True:
			chunk = f.read(1 << 20)
			if not chunk:
				break
			buffer += chunk
			parts = buffer.split(end_token)
			for doc in parts[:-1]:
				if doc.strip():
					yield doc
			buffer = parts[-1]

		if buffer.strip():
			yield buffer


def sample_documents(corpus_path: Path, sample_size: int, seed: int) -> list[str]:
	rng = random.Random(seed)
	sample: list[str] = []
	seen = 0

	for doc in iter_documents(corpus_path):
		seen += 1
		if len(sample) < sample_size:
			sample.append(doc)
		else:
			idx = rng.randrange(seen)
			if idx < sample_size:
				sample[idx] = doc

	if seen < sample_size:
		raise ValueError(f"Not enough documents to sample {sample_size}; found {seen}")

	return sample


def compute_compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> CompressionStats:
	total_bytes = 0
	total_tokens = 0
	for doc in documents:
		total_bytes += len(doc.encode("utf-8"))
		total_tokens += len(tokenizer.encode(doc))
	return CompressionStats(num_docs=len(documents), total_bytes=total_bytes, total_tokens=total_tokens)


def compare_openwebtext_with_tinystories_tokenizer(
	tiny_tokenizer: Tokenizer,
	owt_tokenizer: Tokenizer,
	owt_documents: list[str],
) -> tuple[CompressionStats, CompressionStats, float]:
	tiny_on_owt = compute_compression_ratio(tiny_tokenizer, owt_documents)
	owt_on_owt = compute_compression_ratio(owt_tokenizer, owt_documents)
	ratio = tiny_on_owt.bytes_per_token / owt_on_owt.bytes_per_token
	return tiny_on_owt, owt_on_owt, ratio


def measure_encode_throughput(
	tokenizer: Tokenizer,
	documents: list[str],
	repeats: int = 9,
) -> tuple[CompressionStats, ThroughputStats]:
	compute_compression_ratio(tokenizer, documents)

	times: list[float] = []
	stats = CompressionStats(num_docs=0, total_bytes=0, total_tokens=0)
	for _ in range(repeats):
		start = perf_counter()
		stats = compute_compression_ratio(tokenizer, documents)
		times.append(perf_counter() - start)

	elapsed_seconds = statistics.mean(times)
	throughput = ThroughputStats(total_bytes=stats.total_bytes, elapsed_seconds=elapsed_seconds)
	return stats, throughput


def build_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
	return Tokenizer.from_files(
		vocab_filepath=str(vocab_path),
		merges_filepath=str(merges_path),
		special_tokens=[END_OF_TEXT],
	)


def main() -> None:
	root = Path(__file__).resolve().parent.parent
	fixtures = root / "tests" / "fixtures"
	output_dir = Path(__file__).resolve().parent / "bpe_output"

	tiny_corpus = fixtures / "TinyStoriesV2-GPT4-train.txt"
	owt_corpus = fixtures / "owt_train.txt"

	tiny_tokenizer = build_tokenizer(output_dir / "vocab.json", output_dir / "merges.json")
	owt_tokenizer = build_tokenizer(output_dir / "vocab_owt.json", output_dir / "merges_owt.json")

	sample_size = 10
	tiny_seed = 405
	owt_seed = 406

	tiny_docs = sample_documents(tiny_corpus, sample_size=sample_size, seed=tiny_seed)
	owt_docs = sample_documents(owt_corpus, sample_size=sample_size, seed=owt_seed)

	tiny_stats, tiny_throughput = measure_encode_throughput(tiny_tokenizer, tiny_docs)
	owt_stats, owt_throughput = measure_encode_throughput(owt_tokenizer, owt_docs)
	tiny_on_owt_stats, owt_on_owt_stats, tiny_vs_owt_ratio = compare_openwebtext_with_tinystories_tokenizer(
		tiny_tokenizer=tiny_tokenizer,
		owt_tokenizer=owt_tokenizer,
		owt_documents=owt_docs,
	)
	pile_bytes = 825 * 10**9

	print(f"TinyStories tokenizer (10k vocab) on TinyStories sample ({sample_size} docs):")
	print(f"  total bytes   = {tiny_stats.total_bytes}")
	print(f"  total tokens  = {tiny_stats.total_tokens}")
	print(f"  bytes/token   = {tiny_stats.bytes_per_token:.4f}")

	print()
	print(f"OpenWebText tokenizer (32k vocab) on OpenWebText sample ({sample_size} docs):")
	print(f"  total bytes   = {owt_stats.total_bytes}")
	print(f"  total tokens  = {owt_stats.total_tokens}")
	print(f"  bytes/token   = {owt_stats.bytes_per_token:.4f}")

	print()
	print("Throughput estimates from measured encode elapsed time:")
	print("  TinyStories tokenizer on TinyStories sample:")
	print(f"    elapsed (avg over 9 runs) = {tiny_throughput.elapsed_seconds:.6f} s")
	print(f"    throughput                = {tiny_throughput.bytes_per_second:.2f} B/s ({tiny_throughput.megabytes_per_second:.3f} MB/s)")
	print(f"    est. time for 825GB Pile  = {tiny_throughput.estimated_hours_for_bytes(pile_bytes):.2f} hours")
	print("  OpenWebText tokenizer on OpenWebText sample:")
	print(f"    elapsed (avg over 9 runs) = {owt_throughput.elapsed_seconds:.6f} s")
	print(f"    throughput                = {owt_throughput.bytes_per_second:.2f} B/s ({owt_throughput.megabytes_per_second:.3f} MB/s)")
	print(f"    est. time for 825GB Pile  = {owt_throughput.estimated_hours_for_bytes(pile_bytes):.2f} hours")

	print()
	print(f"Cross-domain: OpenWebText sample ({sample_size} docs)")
	print("  TinyStories tokenizer on OWT:")
	print(f"    total bytes   = {tiny_on_owt_stats.total_bytes}")
	print(f"    total tokens  = {tiny_on_owt_stats.total_tokens}")
	print(f"    bytes/token   = {tiny_on_owt_stats.bytes_per_token:.4f}")
	print("  OpenWebText tokenizer on OWT:")
	print(f"    total bytes   = {owt_on_owt_stats.total_bytes}")
	print(f"    total tokens  = {owt_on_owt_stats.total_tokens}")
	print(f"    bytes/token   = {owt_on_owt_stats.bytes_per_token:.4f}")
	print(f"  relative compression (Tiny/OWT bytes-per-token) = {tiny_vs_owt_ratio:.4f}x")


if __name__ == "__main__":
	main()
