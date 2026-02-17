import pathlib
import json
import time
import psutil
import os
from ece405_basics.bpe import train_bpe

def train_bpe_tinystories():
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"
    input_path = FIXTURES_PATH / "TinyStoriesV2-GPT4-train.txt"
    
    # Track memory and time
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 ** 2)  # MB
    time_start = time.time()
    
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    
    # Calculate elapsed time and memory usage
    time_elapsed = time.time() - time_start
    memory_after = process.memory_info().rss / (1024 ** 2)  # MB
    memory_used = memory_after - memory_before
    hours_elapsed = time_elapsed / 3600
    
    print(f"Learned vocab size: {len(vocab)}")
    print(f"Learned merges: {len(merges)}")
    
    # Serialize vocabulary and merges to disk
    output_dir = pathlib.Path(__file__).resolve().parent / "bpe_output"
    output_dir.mkdir(exist_ok=True)
    
    # Convert vocab to serializable format (bytes -> hex string)
    vocab_serialized = {
        str(k): v.hex() for k, v in vocab.items()
    }
    merges_serialized = [
        (pair[0].hex(), pair[1].hex()) for pair in merges
    ]
    
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    
    with open(vocab_path, "w") as f:
        json.dump(vocab_serialized, f, indent=2)
    
    with open(merges_path, "w") as f:
        json.dump(merges_serialized, f, indent=2)
    
    print(f"\nVocabulary serialized to {vocab_path}")
    print(f"Merges serialized to {merges_path}")
    
    # Find longest token
    longest_token_id = max(vocab.keys(), key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    longest_token_length = len(longest_token_bytes)
    longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")
    
    print(f"\nLongest token length: {longest_token_length} bytes")
    print(f"Longest token: {longest_token_str}")
    
    # Print timing and memory info
    print(f"\nTraining time: {hours_elapsed:.4f} hours ({time_elapsed:.2f} seconds)")
    print(f"Memory usage: {memory_used:.2f} MB")

if __name__ == "__main__":    
    train_bpe_tinystories()
    