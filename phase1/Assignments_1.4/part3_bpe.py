from __future__ import annotations

import json
import re as stdlib_re
from pathlib import Path
from typing import Iterable

try:
    import regex as regex_re
except ImportError:
    regex_re = None


GPT2_SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| "
    r"?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
FALLBACK_SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|"
    r"\s+(?!\S)|\s+"
)


def _pair_counts(ids: list[int], counts: dict[tuple[int, int], int] | None = None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    result: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(ids[index])
            index += 1
    return result


def _recover_tiktoken_merges(mergeable_ranks: dict[bytes, int]):
    """Recover pair merges from tiktoken's merged-byte -> rank dictionary."""
    def recover_pair(token: bytes, max_rank: int | None):
        parts = [bytes([byte]) for byte in token]
        while True:
            best_index = None
            best_rank = None
            for index, (left, right) in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(left + right)
                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_index = index
                    best_rank = rank
            if best_rank is None or (max_rank is not None and best_rank >= max_rank):
                break
            assert best_index is not None
            parts = (
                parts[:best_index]
                + [parts[best_index] + parts[best_index + 1]]
                + parts[best_index + 2 :]
            )
        if len(parts) != 2:
            raise ValueError(f"Could not recover BPE merge for token {token!r}.")
        return parts[0], parts[1]

    merges: dict[tuple[int, int], int] = {}
    for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1]):
        if len(token) <= 1:
            continue
        left, right = recover_pair(token, max_rank=rank)
        merges[(mergeable_ranks[left], mergeable_ranks[right])] = rank
    return merges


class BytePairEncoding:
    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self.special_tokens: dict[str, int] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        self.byte_shuffle: dict[int, int] | None = None
        self.inverse_byte_shuffle: dict[int, int] | None = None
        self._compile_pattern()

    def _compile_pattern(self) -> None:
        if regex_re is not None:
            self._compiled_pattern = regex_re.compile(self.pattern)
            self._regex_module = regex_re
        else:
            if "\\p{" in self.pattern:
                pattern = FALLBACK_SPLIT_PATTERN
            else:
                pattern = self.pattern
            self._compiled_pattern = stdlib_re.compile(pattern)
            self._regex_module = stdlib_re

    def _split(self, text: str) -> list[str]:
        return self._regex_module.findall(self._compiled_pattern, text)

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256.")
        if self.byte_shuffle is not None:
            raise RuntimeError("A tokenizer loaded from tiktoken cannot be retrained.")

        chunks = self._split(text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]
        merges: dict[tuple[int, int], int] = {}
        vocab = {index: bytes([index]) for index in range(256)}
        for merge_index in range(vocab_size - 256):
            stats: dict[tuple[int, int], int] = {}
            for chunk_ids in ids:
                _pair_counts(chunk_ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + merge_index
            ids = [_merge_pair(chunk_ids, pair, new_id) for chunk_ids in ids]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {merge_index + 1}/{vocab_size - 256}: "
                    f"{pair} -> {new_id} ({vocab[new_id]!r}), "
                    f"count={stats[pair]}"
                )
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = dict(special_tokens)
        for token, token_id in self.special_tokens.items():
            self.vocab[token_id] = token.encode("utf-8")

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        if self.byte_shuffle is not None:
            text_bytes = bytes(self.byte_shuffle[byte] for byte in text_bytes)
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = _pair_counts(ids)
            candidates = [pair for pair in stats if pair in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda candidate: self.merges[candidate])
            ids = _merge_pair(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._split(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(
        self,
        text: str,
        allowed_special: str | set[str] = "none_raise",
    ) -> list[int]:
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            if any(token in text for token in self.special_tokens):
                raise ValueError("Input contains a special token; pass allowed_special.")
            special = {}
        elif isinstance(allowed_special, set):
            special = {
                token: token_id
                for token, token_id in self.special_tokens.items()
                if token in allowed_special
            }
        else:
            raise ValueError(f"Unsupported allowed_special={allowed_special!r}.")

        if not special:
            return self.encode_ordinary(text)

        escaped = "|".join(self._regex_module.escape(token) for token in special)
        parts = self._regex_module.split(
            self._regex_module.compile(f"({escaped})"), text
        )
        ids: list[int] = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token id {token_id}.")
            parts.append(self.vocab[token_id])
        text_bytes = b"".join(parts)
        if self.inverse_byte_shuffle is not None:
            text_bytes = bytes(
                self.inverse_byte_shuffle[byte] for byte in text_bytes
            )
        return text_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_tiktoken(cls, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Install tiktoken to run the reference comparison: "
                "python -m pip install tiktoken regex"
            ) from exc

        encoding = tiktoken.get_encoding(encoding_name)
        tokenizer = cls(pattern=encoding._pat_str)
        mergeable_ranks = dict(encoding._mergeable_ranks)
        tokenizer.merges = _recover_tiktoken_merges(mergeable_ranks)
        tokenizer.vocab = {index: bytes([index]) for index in range(256)}
        for (left, right), token_id in tokenizer.merges.items():
            tokenizer.vocab[token_id] = tokenizer.vocab[left] + tokenizer.vocab[right]
        tokenizer.byte_shuffle = {
            byte: mergeable_ranks[bytes([byte])] for byte in range(256)
        }
        tokenizer.inverse_byte_shuffle = {
            shuffled: byte for byte, shuffled in tokenizer.byte_shuffle.items()
        }
        tokenizer.register_special_tokens(dict(encoding._special_tokens))
        return tokenizer

    def save(self, path: str | Path) -> None:
        if self.byte_shuffle is not None:
            raise RuntimeError("Saving tiktoken-compatible permutation is not supported.")
        payload = {
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [
                [left, right, token_id]
                for (left, right), token_id in self.merges.items()
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(pattern=payload["pattern"])
        tokenizer.merges = {
            (int(left), int(right)): int(token_id)
            for left, right, token_id in payload["merges"]
        }
        tokenizer.vocab = {index: bytes([index]) for index in range(256)}
        for (left, right), token_id in tokenizer.merges.items():
            tokenizer.vocab[token_id] = tokenizer.vocab[left] + tokenizer.vocab[right]
        tokenizer.register_special_tokens(payload.get("special_tokens", {}))
        return tokenizer
