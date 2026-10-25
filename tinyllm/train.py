"""The training loop: predict the next token, measure how wrong we were, nudge the weights.

Run from the repo root, for example:

    python -m tinyllm.train                      # char-level Shakespeare, ~5 min on a laptop CPU
    python -m tinyllm.train --tokenizer bpe      # same model, BPE tokens
    python -m tinyllm.train --n_layer 6 --n_embd 384 --n_head 6 --block_size 256 --steps 5000  # GPU
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch

from .data import get_batch, load_shakespeare, train_val_split
from .model import GPT, GPTConfig
from .sample import generate
from .tokenizer import BPETokenizer, CharTokenizer
from .utils import get_device, save_checkpoint, set_seed


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    lr: float = 1e-3          # peak learning rate
    min_lr: float = 1e-4      # learning rate at the end of the cosine decay
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 20
    seed: int = 1337


def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr down to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = (step - tc.warmup_steps) / max(1, tc.steps - tc.warmup_steps)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def estimate_loss(model, splits: dict, block_size: int, tc: TrainConfig, device: str) -> dict:
    """Average loss over a few random batches per split (a single batch is too noisy)."""
    model.eval()
    g = torch.Generator().manual_seed(0)  # same batches every evaluation -> comparable numbers
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, block_size, tc.batch_size, device, g))[1].item()
                  for _ in range(tc.eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(model: GPT, train_data, val_data, tc: TrainConfig, device: str, log=print, on_eval=None) -> dict:
    """Train ``model`` in place and return the loss history. ``on_eval(step, model)`` runs at each evaluation."""
    model.to(device).train()
    opt = model.configure_optimizer(tc.lr, tc.weight_decay)
    history = {"step": [], "train": [], "val": [], "lr": [], "batch_loss": []}
    t0 = time.time()
    for step in range(tc.steps + 1):
        if step % tc.eval_interval == 0 or step == tc.steps:
            losses = estimate_loss(model, {"train": train_data, "val": val_data},
                                   model.cfg.block_size, tc, device)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            log(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                f"| lr {lr_at(step, tc):.2e} | {time.time() - t0:6.1f}s")
            if on_eval is not None:
                on_eval(step, model)
                model.train()
        if step == tc.steps:
            break

        for group in opt.param_groups:      # 1. set the learning rate for this step
            group["lr"] = lr_at(step, tc)
        x, y = get_batch(train_data, model.cfg.block_size, tc.batch_size, device)
        _, loss = model(x, y)               # 2. forward pass: how wrong are we?
        opt.zero_grad(set_to_none=True)
        loss.backward()                     # 3. backward pass: gradients of the loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)  # 4. avoid huge steps
        opt.step()                          # 5. update the weights
        history["batch_loss"].append(loss.item())
        history["lr"].append(lr_at(step, tc))
    history["seconds"] = time.time() - t0
    return history


def main():
    p = argparse.ArgumentParser(description="Train a tiny GPT on Tiny Shakespeare")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--bpe_vocab", type=int, default=512)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    for f, default in asdict(TrainConfig()).items():
        p.add_argument(f"--{f}", type=type(default), default=default)
    p.add_argument("--out", default=None, help="checkpoint path (default: checkpoints/shakespeare_<tok>.pt)")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    text = load_shakespeare()
    if args.tokenizer == "char":
        tok = CharTokenizer.from_text(text)
    else:
        print(f"training BPE tokenizer with vocab {args.bpe_vocab} ...")
        tok = BPETokenizer()
        tok.train(text, args.bpe_vocab)
    ids = tok.encode(text)
    train_data, val_data = train_val_split(ids)
    print(f"device {device} | vocab {tok.vocab_size} | {len(text):,} chars -> {len(ids):,} tokens")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters")
    tc = TrainConfig(**{f: getattr(args, f) for f in asdict(TrainConfig())})
    history = train(model, train_data, val_data, tc, device)

    out = args.out or f"checkpoints/shakespeare_{args.tokenizer}.pt"
    save_checkpoint(out, model, tok, extra={"train_config": asdict(tc)})
    with open(out.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f)
    print(f"saved {out}\n\n--- sample ---")
    idx = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long, device=device)
    print(tok.decode(generate(model, idx, 300, temperature=0.8)[0].tolist()))


if __name__ == "__main__":
    main()
