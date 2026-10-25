"""Small helpers shared by the scripts."""

from __future__ import annotations

import os
import random

import torch

from .model import GPT, GPTConfig
from .tokenizer import tokenizer_from_dict


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path: str, model: GPT, tokenizer, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(),
                "tokenizer": tokenizer.to_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, tokenizer_from_dict(ckpt["tokenizer"])
