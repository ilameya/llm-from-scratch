"""Loading text and cutting it into (input, target) training batches."""

from __future__ import annotations

import os
import urllib.request

import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare() -> str:
    """~1 MB of Shakespeare plays (public domain). Downloaded once, then cached."""
    path = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(ids: list[int], val_fraction: float = 0.1):
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu",
              generator: torch.Generator | None = None):
    """Sample ``batch_size`` random windows. y is x shifted one position to the left.

    For a window "hello" the model sees  x = h e l l   and must predict
                                         y = e l l o
    so every window gives ``block_size`` training examples at once.
    """
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)
