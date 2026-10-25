"""A small GPT-style decoder-only transformer, written to be read.

Every piece is implemented by hand (attention, the MLP, the residual block, the
full model) so you can see exactly what happens to the tensors. Shapes are
written in comments using:

    B = batch size        T = sequence length (time)
    C = n_embd            H = n_head      D = head size = C // H
    V = vocab size
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65    # number of distinct tokens
    block_size: int = 128   # maximum context length (tokens the model can "see")
    n_layer: int = 4        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # width of the residual stream
    dropout: float = 0.1
    bias: bool = True       # bias terms in Linear / LayerNorm

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Each token builds a query ("what am I looking for?"), a key ("what do I
    contain?") and a value ("what do I hand over if chosen?"). A token's output
    is a weighted average of the values of the tokens before it, weighted by
    how well its query matches their keys.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        # One Linear produces q, k and v for all heads at once: C -> 3C
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask: position t may only look at positions <= t
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)
        self.last_attn: torch.Tensor | None = None  # saved for visualisation

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None, save_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                          # each (B, T, C)
        # Split the channels into H heads of size D and move heads next to batch
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, H, T, D)

        start = 0
        if kv_cache is not None:  # generation: reuse keys/values of earlier tokens
            if "k" in kv_cache:
                start = kv_cache["k"].size(2)
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        S = k.size(2)  # number of positions we can attend to

        # Scaled dot-product scores: how much should query i attend to key j?
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)    # (B, H, T, S)
        att = att.masked_fill(~self.mask[:, :, start:start + T, :S], float("-inf"))
        att = F.softmax(att, dim=-1)                                   # rows sum to 1
        if save_attn:
            self.last_attn = att.detach()
        att = self.attn_drop(att)
        y = att @ v                                                    # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)               # concat heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: each token is processed independently."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)   # expand
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # contract
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block (pre-norm): communicate (attention), then compute (MLP)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None, save_attn=False):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, save_attn=save_attn)  # residual 1
        x = x + self.mlp(self.ln2(x))                                           # residual 2
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id  -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position  -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the input embedding and the output projection share weights.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, save_attn=False):
        """idx: (B, T) token ids. Returns (logits (B, T, V), loss or None)."""
        B, T = idx.shape
        start = 0
        if kv_caches is not None and "k" in kv_caches[0]:
            start = kv_caches[0]["k"].size(2)
        assert start + T <= self.cfg.block_size, f"sequence too long: {start + T} > {self.cfg.block_size}"

        pos = torch.arange(start, start + T, device=idx.device)          # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)                         # (B, T, C)
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=None if kv_caches is None else kv_caches[i], save_attn=save_attn)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                          # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
        """AdamW with weight decay on matrices only (not on biases / LayerNorm gains)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
