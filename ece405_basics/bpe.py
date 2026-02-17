import os
import regex as re
from multiprocessing import Pool
from typing import Dict, List, Tuple
from collections import defaultdict

def pretokenize(text):
    """Split text into pretokens using GPT-2 style pattern matching.
    
    This function tokenizes text by matching:
    - Contractions (e.g., 's, 't, 'm, 'll, 've, 're)
    - Letters (with optional leading space)
    - Numbers (with optional leading space)
    - Non-alphanumeric characters (with optional leading space)
    - Whitespace sequences
    
    Args:
        text: Input text string to pretokenize
        
    Yields:
        str: Individual pretokens extracted from the text
    """
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

def count_pairs(ids, weight=1, counts=None):
    """Count consecutive pairs in a sequence of IDs with optional weighting.
    
    Args:
        ids: Sequence of token IDs
        weight: Multiplier for the count (default 1)
        counts: Optional existing defaultdict to accumulate into
        
    Returns:
        defaultdict[tuple, int]: Dictionary mapping pairs to their weighted counts
    """
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts

def merge(ids, pair, new_id):
    """Replace all occurrences of a pair in an ID sequence with a new merged ID.
    
    Args:
        ids: Sequence of token IDs
        pair: Tuple of two consecutive IDs to merge
        new_id: The new ID to replace the pair with
        
    Returns:
        list: New sequence with all pair occurrences replaced by new_id
    """
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def find_chunk_boundaries(file_path, max_workers, end_token="<|endoftext|>"):
    """Divide a file into chunks aligned to end tokens for parallel processing.
    
    This function splits a large file into approximately equal chunks, but adjusts
    boundaries to align with end tokens to avoid splitting documents in the middle.
    
    Args:
        file_path: Path to the file to chunk
        max_workers: Number of worker processes (determines number of chunks)
        end_token: Token marking document boundaries (default "<|endoftext|>")
        
    Returns:
        list[int]: Sorted list of byte positions marking chunk boundaries
    """
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        num_chunks = max_workers or max(1, os.cpu_count() or 1)
        chunk_size = file_size // num_chunks
        

        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # Append the file end position

        buffer_size = 4096  # Size of buffer for reading ahead from boundary positions

        # Adjust boundary positions to align with end tokens
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)  # Start from estimated boundary position

            while True:
                buffer = file.read(buffer_size)  # Read one buffer's worth of data

                # If we've reached the end of the file
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Search for the end token in the buffer
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # If found, update the boundary to this position
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                # If not found, advance to the next buffer position
                chunk_position += buffer_size

    # Remove duplicates, sort, and return
    return sorted(set(chunk_boundaries))

def pretoken_chunk(args):
    """Process a file chunk and count pretokens (worker function for multiprocessing).
    
    Args:
        args: Tuple of (file_path, start, end, special_tokens)
            - file_path: Path to the file
            - start: Starting byte position
            - end: Ending byte position
            - special_tokens: List of special tokens to preserve
            
    Returns:
        defaultdict[str, int]: Dictionary mapping pretokens to their counts
    """
    file_path, start, end, special_tokens = args
    pretoken_counts = defaultdict(int)
    special_token_set = set(special_tokens or [])

    # Read the chunk from the file
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # Split text while preserving special tokens
        for part in split_keep_special_tokens(chunk_text, special_tokens):
            if part in special_token_set:
                pretoken_counts[part] += 1
                continue
            for pretoken in pretokenize(part):
                pretoken_counts[pretoken] += 1

    return pretoken_counts


def split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    Split text on special tokens so that no merging can occur across special token boundaries.
    
    Args:
        text: The input text to split
        special_tokens: List of special tokens to split on
    
    Returns:
        List of text segments, with special tokens removed as delimiters
    """
    if not special_tokens:
        return [text]
    
    # Escape special tokens and join with | for regex alternation
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escaped_tokens)
    
    # Split on the special tokens, which removes them from the result
    segments = re.split(pattern, text)
    
    # Filter out empty segments
    return [seg for seg in segments if seg]


def split_keep_special_tokens(text: str, special_tokens: List[str] | None) -> List[str]:
    """Split text on special tokens while keeping them in the result.
    
    Args:
        text: The input text to split
        special_tokens: List of special tokens to split on
        
    Returns:
        List of text segments alternating between regular text and special tokens
    """
    if not special_tokens:
        return [text]

    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = "(" + "|".join(escaped_tokens) + ")"
    parts = re.split(pattern, text)
    return [part for part in parts if part]

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str] | None = None,
    max_workers: int | None = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a Byte Pair Encoding (BPE) tokenizer from a text corpus.
    
    This function implements the BPE algorithm to learn a vocabulary of subword units
    by iteratively merging the most frequent pairs of bytes. It supports:
    - Parallel processing for large files
    - Special tokens that are never merged
    - Efficient incremental updates to avoid reprocessing entire corpus
    
    Args:
        input_path: Path to the training corpus file
        vocab_size: Target vocabulary size (must be >= 256 + len(special_tokens))
        special_tokens: Optional list of tokens to preserve (e.g., ["<|endoftext|>"])
        max_workers: Number of worker processes for parallel processing (default: CPU count)
        
    Returns:
        Tuple containing:
        - vocab: Dict mapping token IDs to their byte representations
        - merges: List of (bytes, bytes) pairs representing merge operations in order
    """
    
    # Initialize vocabulary with all single bytes (0-255)
    vocab: Dict[int, bytes] = {
        x: bytes([x]) for x in range(256)
    }
    merges: List[Tuple[bytes, bytes]] = []  # (bytes1, bytes2) pairs that were merged

    special_tokens = special_tokens or []
    end_token = special_tokens[0] if special_tokens else "<|endoftext|>"

    chunk_boundaries = find_chunk_boundaries(input_path, max_workers, end_token)
    total_chunks = len(chunk_boundaries) - 1

    chunk_info_list = []
    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        chunk_info_list.append((input_path, start, end, special_tokens))

    # Process chunks in parallel to count pretokens
    with Pool(processes=max_workers) as pool:
        all_results = list(pool.imap(pretoken_chunk, chunk_info_list))
    
    # Merge results from all chunks
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    # Add special tokens to vocabulary
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    # Convert pretokens to byte sequences for BPE training
    ids_counts = {
        tuple(pretoken.encode("utf-8")): count
        for pretoken, count in pretoken_counts.items()
        if pretoken not in special_tokens
    }
  
    # Calculate how many merges we need to reach target vocab size
    num_merges = max(vocab_size - len(vocab), 0)
    pair_to_ids = defaultdict(set)  # Cache mapping pairs to ID sequences containing them
    base_vocab_size = len(vocab)

    # Initialize pair counts from all byte sequences
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):  # Register pairs in cache for efficient lookup
            pair_to_ids[pair].add(ids)
    
    # Main BPE training loop: iteratively merge most frequent pairs
    for step in range(num_merges):
        if not pair_counts:  # Exit if no pairs exist to merge
            break

        # Find most frequent pair (tie-break by lexicographic order of bytes)
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        new_id = base_vocab_size + step

        # Add the merged pair to vocabulary and merge list
        pair_bytes = (vocab[best_pair[0]], vocab[best_pair[1]])
        vocab[new_id] = pair_bytes[0] + pair_bytes[1]
        merges.append(pair_bytes)

        # Update all sequences containing this pair
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]  # Remove from cache since this pair is now merged

        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)]  # Remove the old ID sequence
            ids_counts[tuple(new_ids)] = ids_count  # Add the new merged ID sequence

            # Update pair counts: subtract old pairs, add new pairs
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))

    return vocab, merges
