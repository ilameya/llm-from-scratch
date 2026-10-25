"""Part 8 - Scaling: what happens when the model gets bigger?

    python parts/08_scaling.py           # trains 4 model sizes (~15 min on a laptop CPU)
    python parts/08_scaling.py --quick   # fewer steps
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from _style import BLUE, FIG_DIR, INK2, save
from tinyllm.data import load_shakespeare, train_val_split
from tinyllm.model import GPT, GPTConfig
from tinyllm.tokenizer import CharTokenizer
from tinyllm.train import TrainConfig, train
from tinyllm.utils import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

device = get_device()
text = load_shakespeare()
tok = CharTokenizer.from_text(text)
train_data, val_data = train_val_split(tok.encode(text))
tc = TrainConfig(steps=400 if args.quick else 1500, eval_interval=250, eval_iters=20)
tokens_seen = tc.steps * tc.batch_size * 128

sizes = [(2, 32), (3, 64), (4, 128), (6, 192)]  # (n_layer, n_embd)
results = []
for n_layer, n_embd in sizes:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=n_layer, n_head=4,
                    n_embd=n_embd, dropout=0.0)
    model = GPT(cfg)
    n = model.num_params()
    h = train(model, train_data, val_data, tc, device, log=lambda s: None)
    results.append({"n_layer": n_layer, "n_embd": n_embd, "params": n, "val": h["val"][-1],
                    "train": h["train"][-1], "seconds": h["seconds"]})
    print(f"L={n_layer} C={n_embd:3d} | {n / 1e6:5.2f}M params | val {h['val'][-1]:.3f} "
          f"| {h['seconds']:.0f}s | tokens/param {tokens_seen / n:.0f}")

with open(os.path.join(FIG_DIR, "08_scaling_results.json"), "w") as f:
    json.dump(results, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 4))
xs, ys = [r["params"] for r in results], [r["val"] for r in results]
ax.plot(xs, ys, color=BLUE, marker="o", markersize=9)
for r in results:
    ax.annotate(f"L={r['n_layer']}, C={r['n_embd']}\n{r['val']:.2f}", (r["params"], r["val"]),
                textcoords="offset points", xytext=(8, 6), color=INK2, fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("parameters (log scale)")
ax.set_ylabel("validation loss")
ax.set_title(f"Bigger models reach lower loss (same {tc.steps} steps each)")
ax.margins(x=0.25, y=0.15)
save(fig, "08_scaling")
