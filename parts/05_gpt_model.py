"""Part 5 - Assembling the full GPT and following one batch through it.

    python parts/05_gpt_model.py
"""

import math

import matplotlib.pyplot as plt
import torch

from _style import AQUA, BLUE, INK2, ORANGE, VIOLET, YELLOW_DARK, save
from tinyllm.data import load_shakespeare
from tinyllm.model import GPT, GPTConfig
from tinyllm.sample import generate
from tinyllm.tokenizer import CharTokenizer

torch.manual_seed(0)
text = load_shakespeare()
tok = CharTokenizer.from_text(text)
cfg = GPTConfig(vocab_size=tok.vocab_size)  # the default model trained in Part 6
model = GPT(cfg).eval()
print(cfg, "\n")

# --- 1. Shape trace: register hooks that print every module's output shape --------
trace = []
def hook(name):
    return lambda mod, inp, out: trace.append((name, tuple((out[0] if isinstance(out, tuple) else out).shape)))

for name in ["tok_emb", "pos_emb", "blocks.0.ln1", "blocks.0.attn", "blocks.0.mlp", "blocks.0",
             "blocks.3", "ln_f", "lm_head"]:
    model.get_submodule(name).register_forward_hook(hook(name))
idx = torch.tensor([tok.encode("ROMEO:\nWherefore art"), tok.encode("JULIET:\nO Romeo, Romeo")[:20]])
logits, _ = model(idx)
print(f"input ids: {tuple(idx.shape)}   (B=2 sequences, T=20 tokens)")
for name, shape in trace:
    print(f"  {name:14s} -> {shape}")
print()

# --- 2. Logits -> probabilities for the next token ---------------------------------
probs = torch.softmax(logits[0, -1], dim=-1)
print(f"next-token distribution after {tok.decode(idx[0].tolist())!r}: {probs.shape[0]} probabilities, sum={probs.sum():.3f}")
print(f"untrained model has no real preference yet: top probability {probs.max():.3f} (uniform would be {1 / cfg.vocab_size:.3f})")

# --- 3. Loss at initialisation should be about ln(V) -------------------------------
targets = torch.randint(0, cfg.vocab_size, idx.shape)
_, loss = model(idx, targets)
print(f"loss at init: {loss.item():.3f}   ln(V) = ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}\n")

# --- 4. Gibberish in, gibberish out (for now) --------------------------------------
out = generate(model, torch.tensor([tok.encode("ROMEO:")]), 80, generator=torch.Generator().manual_seed(1))
print("untrained sample:", repr(tok.decode(out[0].tolist())), "\n")

# --- 5. Parameter budget ------------------------------------------------------------
def count(prefix):
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))

L, C, V, T = cfg.n_layer, cfg.n_embd, cfg.vocab_size, cfg.block_size
parts = {
    "token embedding\n(shared with output)": count("tok_emb"),
    "position embedding": count("pos_emb"),
    "attention (all layers)": sum(count(f"blocks.{i}.attn") for i in range(L)),
    "MLP (all layers)": sum(count(f"blocks.{i}.mlp") for i in range(L)),
    "LayerNorms": sum(count(f"blocks.{i}.ln") for i in range(L)) + count("ln_f"),
}
total = sum(p.numel() for p in model.parameters())
for k, v in parts.items():
    print(f"  {k.replace(chr(10), ' '):38s} {v:9,d}  ({100 * v / total:4.1f}%)")
print(f"  {'total':38s} {total:9,d}")
print(f"  rule of thumb 12*L*C^2 = {12 * L * C * C:,}\n")

fig, ax = plt.subplots(figsize=(7.5, 3.4))
names, vals = list(parts)[::-1], list(parts.values())[::-1]
colors = [BLUE, VIOLET, ORANGE, AQUA, YELLOW_DARK][::-1]  # same colour roles as the diagrams
ax.barh(names, vals, color=colors, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + total * 0.01, i, f"{v:,}  ({100 * v / total:.1f}%)", va="center", color=INK2, fontsize=10)
ax.set_xlim(0, max(vals) * 1.35)
ax.set_xlabel("parameters")
ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
ax.grid(axis="y", visible=False)
ax.set_title(f"Where the {total / 1e6:.2f}M parameters of our tiny GPT live")
save(fig, "05_param_breakdown")

# --- 6. Same code, GPT-2 small size ------------------------------------------------
gpt2 = GPT(GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0))
print(f"GPT-2 small config -> {sum(p.numel() for p in gpt2.parameters()) / 1e6:.1f}M parameters (OpenAI reports 124M)")
