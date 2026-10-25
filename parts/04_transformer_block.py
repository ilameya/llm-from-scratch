"""Part 4 - The transformer block: multi-head attention + MLP + residuals + LayerNorm.

    python parts/04_transformer_block.py            # full run (~10 min on a laptop CPU)
    python parts/04_transformer_block.py --quick    # shorter ablation
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from _style import AQUA, BLUE, INK2, ORANGE, save
from tinyllm.data import load_shakespeare, train_val_split
from tinyllm.model import GPT, MLP, Block, CausalSelfAttention, GPTConfig
from tinyllm.tokenizer import CharTokenizer
from tinyllm.train import TrainConfig, train

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
torch.manual_seed(0)

# --- 1. A block maps (B, T, C) -> (B, T, C) --------------------------------------
cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.0)
block = Block(cfg)
x = torch.randn(2, 16, 32)
print("block input :", tuple(x.shape))
print("block output:", tuple(block(x).shape), "-> same shape, so blocks can be stacked\n")

# --- 2. Multi-head = several small heads side by side -----------------------------
attn = CausalSelfAttention(cfg)
print(f"n_embd={cfg.n_embd}, n_head={cfg.n_head} -> head_size={attn.head_size}")
q, k, v = attn.qkv(x).split(cfg.n_embd, dim=2)
print("q for all heads:", tuple(q.shape), "-> viewed as", (2, cfg.n_head, 16, attn.head_size), "\n")

# --- 3. LayerNorm normalises each token's vector ---------------------------------
ln = nn.LayerNorm(32)
y = ln(x * 5 + 3)
print(f"before LayerNorm: mean {(x * 5 + 3)[0, 0].mean():.2f}, std {(x * 5 + 3)[0, 0].std():.2f}")
print(f"after  LayerNorm: mean {y[0, 0].mean():.2f}, std {y[0, 0].std(unbiased=False):.2f}\n")

# --- 4. Where do the parameters live? ----------------------------------------------
def count(m):
    return sum(p.numel() for p in m.parameters())

C = 128
cfg128 = GPTConfig(n_embd=C, n_head=4)
b = Block(cfg128)
print(f"Parameters in one block with n_embd={C}:")
print(f"  attention (qkv + proj): {count(b.attn):7,d}   (~4*C^2 = {4 * C * C:,})")
print(f"  MLP (expand + contract): {count(b.mlp):7,d}   (~8*C^2 = {8 * C * C:,})")
print(f"  2 x LayerNorm          : {count(b.ln1) + count(b.ln2):7,d}")
print(f"  total                  : {count(b):7,d}   (~12*C^2 = {12 * C * C:,})\n")

# --- 5. Ablation: remove residuals or LayerNorm and watch training suffer ----------
class AblatedBlock(Block):
    def __init__(self, cfg, residual=True, layernorm=True):
        super().__init__(cfg)
        self.residual = residual
        if not layernorm:
            self.ln1, self.ln2 = nn.Identity(), nn.Identity()

    def forward(self, x, kv_cache=None, save_attn=False):
        a = self.attn(self.ln1(x))
        x = x + a if self.residual else a
        m = self.mlp(self.ln2(x))
        return x + m if self.residual else m


text = load_shakespeare()
tok = CharTokenizer.from_text(text)
train_data, val_data = train_val_split(tok.encode(text))
steps = 300 if args.quick else 800
deep = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=8, n_head=4, n_embd=64, dropout=0.0)
tc = TrainConfig(steps=steps, batch_size=32, lr=1e-3, warmup_steps=50, eval_interval=50, eval_iters=10)

variants = [("full block", dict(), BLUE),
            ("no LayerNorm", dict(layernorm=False), AQUA),
            ("no residual connections", dict(residual=False), ORANGE)]
results = {}
for name, kw, _ in variants:
    print(f"--- training 8-layer model: {name}")
    torch.manual_seed(0)
    model = GPT(deep)
    model.blocks = nn.ModuleList([AblatedBlock(deep, **kw) for _ in range(deep.n_layer)])
    model.apply(model._init_weights)
    results[name] = train(model, train_data, val_data, tc, "cpu", log=lambda s: None)
    print(f"    final val loss {results[name]['val'][-1]:.3f}")

# Baseline: a "model" that ignores context and only knows how often each character appears
counts = torch.bincount(train_data, minlength=tok.vocab_size).float()
unigram_loss = -torch.log(counts / counts.sum())[val_data].mean().item()
print(f"character-frequency baseline (no context at all): {unigram_loss:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.hlines(unigram_loss, 0, steps * 0.95, color=INK2, linestyles=":", linewidth=1.5)
ax.text(steps * 0.5, unigram_loss - 0.08, "character frequencies only", ha="center", color=INK2, fontsize=9)
for name, _, color in variants:
    h = results[name]
    ax.plot(h["step"], h["val"], color=color, label=name)
    ax.annotate(f"{h['val'][-1]:.2f}", (h["step"][-1], h["val"][-1]), textcoords="offset points",
                xytext=(6, 0), va="center", color=INK2)
ax.set_xlabel("training step")
ax.set_ylabel("validation loss")
ax.set_title("Same 8-layer model, three wirings")
ax.legend()
save(fig, "04_block_ablation")
