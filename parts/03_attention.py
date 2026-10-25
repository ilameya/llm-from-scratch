"""Part 3 - Self-attention, built up one step at a time.

    python parts/03_attention.py
The final heatmap needs a trained model (python -m tinyllm.train).
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from _style import BLUE, INK2, ORANGE, ROOT, SEQ, save
from tinyllm.utils import load_checkpoint

torch.manual_seed(42)
B, T, C = 1, 4, 8
x = torch.randn(B, T, C)  # 4 tokens, each an 8-dim vector

# --- Step 1: "average of everything before me", with a loop -----------------
avg_loop = torch.zeros_like(x)
for t in range(T):
    avg_loop[:, t] = x[:, : t + 1].mean(dim=1)

# --- Step 2: the same thing as ONE matrix multiply ----------------------------
w = torch.tril(torch.ones(T, T))
w = w / w.sum(dim=1, keepdim=True)
print("averaging weights (lower-triangular):\n", w.numpy().round(2))
avg_matmul = w @ x
print("loop == matmul?", torch.allclose(avg_loop, avg_matmul, atol=1e-6))

# --- Step 3: the same thing again, via a masked softmax -----------------------
scores = torch.zeros(T, T)
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
w_soft = F.softmax(scores, dim=-1)
print("softmax version == matmul?", torch.allclose(w_soft @ x, avg_matmul, atol=1e-6))
print("-> attention = replace the zeros in `scores` with LEARNED, data-dependent numbers\n")

# --- Step 4: a single attention head --------------------------------------------
head_size = 4
query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
q, k, v = query(x), key(x), value(x)                          # each (B, T, head_size)
scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)       # (B, T, T)
scores = scores.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ v                                             # (B, T, head_size)
print("q, k, v shapes:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
print("attention weights (each row sums to 1):\n", weights[0].detach().numpy().round(2))
print("output shape:", tuple(out.shape), "\n")

# --- Step 5: why divide by sqrt(head_size)? -----------------------------------
for d in [4, 64, 512]:
    qq, kk = torch.randn(10_000, d), torch.randn(10_000, d)
    dots = (qq * kk).sum(-1)
    print(f"d={d:4d}: variance of q.k = {dots.var():7.1f}   after / sqrt(d): {(dots / math.sqrt(d)).var():.2f}")

# The same five scores, once scaled and once not (for d=64 the unscaled ones are sqrt(64)=8x larger)
scaled_scores = torch.tensor([0.2, 0.5, 0.9, 1.2, 1.0])
p_scaled = F.softmax(scaled_scores, dim=-1)
p_raw = F.softmax(scaled_scores * math.sqrt(64), dim=-1)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for ax, p, title, color in [(axes[0], p_scaled, "scaled by 1/√d", BLUE),
                            (axes[1], p_raw, "not scaled", ORANGE)]:
    ax.bar(range(5), p, color=color, width=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.set_xticks(range(5), [f"tok {i}" for i in range(5)])
    for i, val in enumerate(p):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", color=INK2, fontsize=9)
axes[0].set_ylabel("attention weight")
axes[0].set_ylim(0, 1.1)
fig.suptitle("Without scaling, softmax piles almost all weight on one token", x=0.02, ha="left",
             fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "03_softmax_scaling")

# --- Step 6: real attention patterns from the trained model --------------------
ckpt = os.path.join(ROOT, "checkpoints/shakespeare_char.pt")
if not os.path.exists(ckpt):
    raise SystemExit("Train a model first (python -m tinyllm.train) to see real attention maps.")
model, tok = load_checkpoint(ckpt)
text = "To be, or not to be"
idx = torch.tensor([tok.encode(text)])
with torch.no_grad():
    model(idx, save_attn=True)
labels = [{" ": "␣"}.get(c, c) for c in text]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
layer = model.blocks[-1].attn.last_attn[0]  # (H, T, T) from the last block
for h, ax in enumerate(axes):
    ax.imshow(layer[h], cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(len(text)), labels, fontsize=8)
    ax.set_yticks(range(len(text)), labels, fontsize=8)
    ax.set_title(f"head {h + 1}", fontsize=11)
    ax.set_xlabel("attends to (key)")
axes[0].set_ylabel("token (query)")
fig.suptitle("Attention weights in the last layer: each row is one token looking back",
             x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "03_attention_heads")
