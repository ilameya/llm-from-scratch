"""Tokenizers: turning text into integers and back.

Two tokenizers live here:

* ``CharTokenizer`` - one token per character. Tiny vocabulary, long sequences.
* ``BPETokenizer``  - byte-level Byte Pair Encoding, the same family of algorithm
  used by GPT-2/3/4 and most modern LLMs. Starts from the 256 possible bytes and
  repeatedly merges the most frequent adjacent pair into a new token.

Both expose the same tiny interface: ``encode(str) -> list[int]``,
``decode(list[int]) -> str``, ``vocab_size``, ``save(path)`` and ``load(path)``.
"""

from __future__ import annotations

import json
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Maps every distinct character in the training text to an integer id."""

    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "chars": self.chars}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)


# ---------------------------------------------------------------------------
# Byte-level BPE tokenizer
# ---------------------------------------------------------------------------

# Split text into "chunks" before merging, so merges never cross word
# boundaries (we never want a token like "e. T"). This is a simplified version
# of the regex GPT-2 uses: optional leading space + letters, numbers,
# punctuation runs, or whitespace runs.
SPLIT_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def get_pair_counts(ids: list[int], counts: Counter | None = None, weight: int = 1) -> Counter:
    """Count how often each adjacent pair appears. [1,2,3,1,2] -> {(1,2):2, (2,3):1, (3,1):1}"""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A minimal byte-level BPE tokenizer (train / encode / decode)."""

    kind = "bpe"

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new token id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab must at least contain the 256 raw bytes"
        # Count unique chunks once; each chunk is a list of byte ids.
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        chunks = {c: list(c.encode("utf-8")) for c in chunk_counts}

        for new_id in range(256, vocab_size):
            stats: Counter = Counter()
            for c, ids in chunks.items():
                get_pair_counts(ids, stats, weight=chunk_counts[c])
            if not stats:
                break  # nothing left to merge
            pair = max(stats, key=stats.get)  # most frequent pair
            for c in chunks:
                chunks[c] = merge(chunks[c], pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                shown = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {new_id - 255:4d}: {pair} -> {new_id} {shown!r} (seen {stats[pair]}x)")

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the merge that was learned EARLIEST among the pairs present.
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "merges": [[a, b, i] for (a, b), i in self.merges.items()]}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls()
        for a, b, i in d["merges"]:  # merges are stored in the order they were learned
            tok.merges[(a, b)] = i
            tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        return tok

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)


def tokenizer_from_dict(d: dict):
    if d["kind"] == "char":
        return CharTokenizer(d["chars"])
    if d["kind"] == "bpe":
        return BPETokenizer.from_dict(d)
    raise ValueError(f"unknown tokenizer kind {d['kind']!r}")


def load_tokenizer(path: str):
    with open(path, encoding="utf-8") as f:
        return tokenizer_from_dict(json.load(f))
