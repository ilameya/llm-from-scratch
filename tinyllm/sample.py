"""Turning next-token probabilities into text.

The model only ever gives us a probability distribution over the next token.
*How* we pick from that distribution (greedy, temperature, top-k, top-p) is a
separate decision, and it changes the output dramatically.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def filter_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = None) -> torch.Tensor:
    """Apply temperature, top-k and top-p (nucleus) filtering to (B, V) logits."""
    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k).values[:, [-1]]                 # (B, 1)
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Drop tokens once the probability mass BEFORE them already exceeds top_p
        drop = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, greedy=False, generator=None):
    """Pick the next token id for each row of (B, V) logits."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             greedy=False, use_cache=False, generator=None):
    """Autoregressive generation: predict a token, append it, repeat.

    idx: (B, T) starting context. Returns (B, T + max_new_tokens).
    With ``use_cache=True`` the keys/values of past tokens are stored and only the
    newest token is pushed through the network each step (the "KV cache"). In this
    simple version the cache must fit inside the context window (block_size).
    """
    model.eval()
    block_size = model.cfg.block_size
    if use_cache:
        assert idx.size(1) + max_new_tokens <= block_size, "KV-cache demo needs everything to fit in block_size"
        caches = [dict() for _ in model.blocks]
        logits, _ = model(idx, kv_caches=caches)           # "prefill": process the prompt once
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)
            idx = torch.cat([idx, nxt], dim=1)
            logits, _ = model(nxt, kv_caches=caches)       # "decode": only the new token
        return idx

    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]                       # crop to the context window
        logits, _ = model(context)                           # (B, T, V)
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def main():
    from .utils import get_device, load_checkpoint

    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    p.add_argument("--ckpt", default="checkpoints/shakespeare_char.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = get_device()
    model, tok = load_checkpoint(args.ckpt, device)
    g = torch.Generator(device=device).manual_seed(args.seed)
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, args.tokens, args.temperature, args.top_k, args.top_p,
                   greedy=args.greedy, generator=g)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
