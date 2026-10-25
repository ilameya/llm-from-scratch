"""Part 2 - Embeddings: giving each token (and each position) a vector.

    python parts/02_embeddings.py
The last two figures need a trained model (python -m tinyllm.train).
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from _style import AQUA, BLUE, DIV, INK2, MUTED, ORANGE, ROOT, SEQ, save
from tinyllm.utils import load_checkpoint

torch.manual_seed(0)

# --- 1. An embedding is just a lookup table -------------------------------
vocab_size, n_embd = 10, 4
emb = nn.Embedding(vocab_size, n_embd)
token_id = torch.tensor([3])
one_hot = torch.zeros(1, vocab_size)
one_hot[0, 3] = 1.0
print("embedding table shape:", tuple(emb.weight.shape))
print("emb(3)            :", emb(token_id).detach().numpy().round(3))
print("one_hot(3) @ table:", (one_hot @ emb.weight).detach().numpy().round(3))
print("-> looking up row 3 == multiplying a one-hot vector by the table\n")

# --- 2. Why we need positions ---------------------------------------------
words = {"dog": 0, "bites": 1, "man": 2}
a = emb(torch.tensor([words[w] for w in "dog bites man".split()])).sum(0)
b = emb(torch.tensor([words[w] for w in "man bites dog".split()])).sum(0)
print("sum of embeddings, 'dog bites man':", a.detach().numpy().round(3))
print("sum of embeddings, 'man bites dog':", b.detach().numpy().round(3))
print("identical?", torch.allclose(a, b), "-> without position information, order is invisible\n")

# --- 3. Sinusoidal positional encodings (the original Transformer) -----------
def sinusoidal(T, C):
    pos = torch.arange(T).unsqueeze(1)                                  # (T, 1)
    freq = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))  # (C/2,)
    pe = torch.zeros(T, C)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe

pe = sinusoidal(128, 64)
fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(pe.T, aspect="auto", cmap=DIV, vmin=-1, vmax=1, interpolation="nearest")
ax.grid(False)
ax.set_xlabel("position in the sequence")
ax.set_ylabel("embedding dimension")
ax.set_title("Sinusoidal positional encoding: every position gets a unique pattern")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
save(fig, "02_sinusoidal_pe")

# --- 4. What a trained model learned ----------------------------------------
ckpt = os.path.join(ROOT, "checkpoints/shakespeare_char.pt")
if not os.path.exists(ckpt):
    raise SystemExit("Train a model first (python -m tinyllm.train) to see the learned embeddings.")
model, tok = load_checkpoint(ckpt)

# 4a. Token embeddings projected to 2D with PCA
W = model.tok_emb.weight.detach()
W = W - W.mean(0)
U, S, V = torch.pca_lowrank(W, q=2, center=False)
xy = W @ V[:, :2]

def group(ch):
    if ch.isupper():
        return "uppercase", AQUA
    if ch in "aeiou":
        return "lowercase vowel", ORANGE
    if ch.isalpha():
        return "lowercase consonant", BLUE
    return "space / punctuation / digit", MUTED

fig, ax = plt.subplots(figsize=(8.5, 6.5))
seen = set()
for i, ch in enumerate(tok.chars):
    name, color = group(ch)
    ax.scatter(xy[i, 0], xy[i, 1], s=70, color=color, edgecolor="white", linewidth=1.5,
               label=name if name not in seen else None, zorder=3)
    seen.add(name)
    label = {"\n": "\\n", " ": "␣"}.get(ch, ch)
    ax.annotate(label, (xy[i, 0], xy[i, 1]), textcoords="offset points", xytext=(5, 4),
                fontsize=9, color=INK2)
ax.set_title("Learned character embeddings (PCA to 2D)")
ax.set_xlabel("principal component 1")
ax.set_ylabel("principal component 2")
ax.legend(loc="best")
save(fig, "02_learned_token_embeddings")

# 4b. Learned positional embeddings: how similar is position i to position j?
P = model.pos_emb.weight.detach()
P = P / P.norm(dim=1, keepdim=True)
sim = P @ P.T
fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(sim, cmap=SEQ, vmin=max(0.0, sim.min().item()), vmax=1, interpolation="nearest")
ax.grid(False)
ax.set_xlabel("position j")
ax.set_ylabel("position i")
ax.set_title("Learned positions: neighbours look alike")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
save(fig, "02_learned_position_similarity")
