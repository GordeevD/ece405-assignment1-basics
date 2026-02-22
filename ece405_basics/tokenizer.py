from __future__ import annotations

import json
import string
from collections.abc import Iterable, Iterator

from ece405_basics.bpe import pretokenize, pretokenize_with_remainder, split_keep_special_tokens


def _gpt2_bytes_to_unicode() -> dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
                if b not in bs:
                        bs.append(b)
                        cs.append(2**8 + n)
                        n += 1
        return dict(zip(bs, [chr(n) for n in cs]))


def _is_hex_string(value: str) -> bool:
        return len(value) % 2 == 0 and all(char in string.hexdigits for char in value)


class Tokenizer:
        def __init__(
                self,
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None,
        ):
                self.vocab = dict(vocab)
                self.merges = list(merges)
                self.special_tokens = list(special_tokens or [])

                vocab_values = set(self.vocab.values())
                for token in self.special_tokens:
                        token_bytes = token.encode("utf-8")
                        if token_bytes not in vocab_values:
                                self.vocab[len(self.vocab)] = token_bytes
                                vocab_values.add(token_bytes)

                self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
                self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
                self.special_token_set = set(self.special_tokens)
                self._sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

                self._pretoken_cache: dict[bytes, tuple[int, ...]] = {}

        @classmethod
        def from_files(
                cls,
                vocab_filepath: str,
                merges_filepath: str,
                special_tokens: list[str] | None = None,
        ) -> Tokenizer:
                with open(vocab_filepath, "r", encoding="utf-8") as f:
                        vocab_obj = json.load(f)

                vocab: dict[int, bytes]
                merge_format: str
                if isinstance(vocab_obj, dict) and all(str(key).isdigit() for key in vocab_obj):
                        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in vocab_obj.items()}
                        merge_format = "hex"
                elif isinstance(vocab_obj, dict) and all(isinstance(value, int) for value in vocab_obj.values()):
                        byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
                        vocab = {
                                token_id: bytes([byte_decoder[ch] for ch in token_text])
                                for token_text, token_id in vocab_obj.items()
                        }
                        merge_format = "gpt2"
                else:
                        raise ValueError("Unsupported vocabulary serialization format.")

                merges: list[tuple[bytes, bytes]] = []
                if merges_filepath.endswith(".json"):
                        with open(merges_filepath, "r", encoding="utf-8") as f:
                                merges_obj = json.load(f)

                        if merge_format == "hex":
                                merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_obj]
                        else:
                                byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
                                merges = [
                                        (
                                                bytes([byte_decoder[ch] for ch in left]),
                                                bytes([byte_decoder[ch] for ch in right]),
                                        )
                                        for left, right in merges_obj
                                ]
                else:
                        byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
                        with open(merges_filepath, "r", encoding="utf-8") as f:
                                for raw_line in f:
                                        line = raw_line.strip()
                                        if not line or line.startswith("#"):
                                                continue
                                        parts = line.split()
                                        if len(parts) != 2:
                                                continue
                                        left, right = parts
                                        if merge_format == "hex" and _is_hex_string(left) and _is_hex_string(right):
                                                merges.append((bytes.fromhex(left), bytes.fromhex(right)))
                                        else:
                                                merges.append(
                                                        (
                                                                bytes([byte_decoder[ch] for ch in left]),
                                                                bytes([byte_decoder[ch] for ch in right]),
                                                        )
                                                )

                return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

        def _encode_pretoken(self, token: str) -> list[int]:
                token_bytes = token.encode("utf-8")
                cached = self._pretoken_cache.get(token_bytes)
                if cached is not None:
                        return list(cached)

                pieces = [bytes([byte]) for byte in token_bytes]
                while len(pieces) > 1:
                        best_rank: int | None = None
                        best_index = -1
                        for index in range(len(pieces) - 1):
                                rank = self.merge_ranks.get((pieces[index], pieces[index + 1]))
                                if rank is not None and (best_rank is None or rank < best_rank):
                                        best_rank = rank
                                        best_index = index

                        if best_index == -1:
                                break

                        merged_piece = pieces[best_index] + pieces[best_index + 1]
                        pieces = pieces[:best_index] + [merged_piece] + pieces[best_index + 2 :]

                token_ids = [self.bytes_to_id[piece] for piece in pieces]
                self._pretoken_cache[token_bytes] = tuple(token_ids)
                return token_ids

        def _find_next_special_token(self, text: str) -> tuple[int, str] | None:
                if not self._sorted_special_tokens:
                        return None

                best_index: int | None = None
                best_token = ""
                for token in self._sorted_special_tokens:
                        index = text.find(token)
                        if index == -1:
                                continue
                        if best_index is None or index < best_index or (index == best_index and len(token) > len(best_token)):
                                best_index = index
                                best_token = token

                if best_index is None:
                        return None
                return best_index, best_token

        def _encode_pretoken_buffer(self, text: str, final: bool) -> tuple[list[int], str]:
                token_ids: list[int] = []

                if final:
                        for token in pretokenize(text):
                                token_ids.extend(self._encode_pretoken(token))
                        return token_ids, ""

                tokens, remainder = pretokenize_with_remainder(text)
                for token in tokens:
                        token_ids.extend(self._encode_pretoken(token))
                return token_ids, remainder

        def _encode_text_with_special_tokens(self, text: str) -> list[int]:
                token_ids: list[int] = []
                for segment in split_keep_special_tokens(text, self._sorted_special_tokens):
                        if segment in self.special_token_set:
                                token_ids.append(self.bytes_to_id[segment.encode("utf-8")])
                        else:
                                segment_ids, _ = self._encode_pretoken_buffer(segment, final=True)
                                token_ids.extend(segment_ids)
                return token_ids

        def _special_prefix_suffix_length(self, text: str) -> int:
                if not self._sorted_special_tokens or not text:
                        return 0

                best_length = 0
                for token in self._sorted_special_tokens:
                        max_check_length = min(len(token) - 1, len(text))
                        for length in range(max_check_length, 0, -1):
                                if text.endswith(token[:length]):
                                        if length > best_length:
                                                best_length = length
                                        break
                return best_length

        def encode(self, text: str) -> list[int]:
                if not self._sorted_special_tokens:
                        token_ids, remainder = self._encode_pretoken_buffer(text, final=True)
                        if remainder:
                                token_ids.extend(self._encode_pretoken(remainder))
                        return token_ids
                return self._encode_text_with_special_tokens(text)

        def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
                buffered_text = ""
                for chunk in iterable:
                        buffered_text += chunk

                        while True:
                                next_special = self._find_next_special_token(buffered_text)
                                if next_special is None:
                                        break

                                special_start, special_token = next_special
                                if special_start > 0:
                                        prefix_ids, _ = self._encode_pretoken_buffer(buffered_text[:special_start], final=True)
                                        for token_id in prefix_ids:
                                                yield token_id

                                yield self.bytes_to_id[special_token.encode("utf-8")]
                                buffered_text = buffered_text[special_start + len(special_token) :]

                        keep_length = self._special_prefix_suffix_length(buffered_text)

                        cutoff = len(buffered_text) - keep_length
                        safe_prefix = buffered_text[:cutoff]
                        tail = buffered_text[cutoff:]

                        token_ids, remainder = self._encode_pretoken_buffer(safe_prefix, final=False)
                        for token_id in token_ids:
                                yield token_id
                        buffered_text = remainder + tail

                if buffered_text:
                        final_ids = self._encode_text_with_special_tokens(buffered_text)
                        for token_id in final_ids:
                                yield token_id

        def decode(self, ids: list[int]) -> str:
                decoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
                return decoded_bytes.decode("utf-8", errors="replace")