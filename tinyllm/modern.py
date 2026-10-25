"""Building blocks used by modern LLMs (Llama, Mistral, Qwen), for the Part 8 exercises.

Drop-in replacements you can try in model.py:
    nn.LayerNorm(C)  ->  RMSNorm(C)
    MLP(cfg)         ->  SwiGLUMLP(cfg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction and bias: rescale by the root mean square."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLUMLP(nn.Module):
    """Gated MLP: silu(gate(x)) decides how much of up(x) passes through.

    A hidden size of 8/3 * C keeps the parameter count equal to our 4C GELU MLP (3 matrices instead of 2).
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):                       # three Linear layers instead of two
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
