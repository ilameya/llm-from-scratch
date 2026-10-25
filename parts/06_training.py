"""Part 6 - Training: watch the model go from noise to (almost) Shakespeare.

    python parts/06_training.py          # ~same as `python -m tinyllm.train`, plus figures
"""

import json
import os

import matplotlib.pyplot as plt
import torch

from _style import BLUE, INK2, MUTED, ORANGE, ROOT, save
from tinyllm.data import get_batch, load_shakespeare, train_val_split
from tinyllm.model import GPT, GPTConfig
from tinyllm.sample import generate
from tinyllm.tokenizer import CharTokenizer
from tinyllm.train import TrainConfig, lr_at, train
from tinyllm.utils import get_device, save_checkpoint, set_seed

set_seed(1337)
device = get_device()
text = load_shakespeare()
tok = CharTokenizer.from_text(text)
train_data, val_data = train_val_split(tok.encode(text))

# --- 1. What a training example looks like -------------------------------------
x, y = get_batch(train_data, block_size=8, batch_size=1)
print("input  x:", repr(tok.decode(x[0].tolist())))
print("target y:", repr(tok.decode(y[0].tolist())))
for t in range(4):
    print(f"  given {tok.decode(x[0, :t + 1].tolist())!r:12} -> predict {tok.decode([y[0, t].item()])!r}")
print("-> one window of 8 tokens = 8 training examples\n")

# --- 2. Train, sampling a little text at every evaluation --------------------------
cfg = GPTConfig(vocab_size=tok.vocab_size)
model = GPT(cfg)
tc = TrainConfig()
snapshots = {}
prompt = torch.tensor([tok.encode("ROMEO:")], device=device)

def take_snapshot(step, m):
    if step in (0, 250, 1000, tc.steps):
        g = torch.Generator(device=device).manual_seed(0)
        snapshots[step] = tok.decode(generate(m, prompt, 120, temperature=0.8, generator=g)[0].tolist())

history = train(model, train_data, val_data, tc, device, on_eval=take_snapshot)
for step, sample in snapshots.items():
    print(f"\n=== after {step} steps ===\n{sample}")

path = os.path.join(ROOT, "checkpoints/shakespeare_char.pt")
save_checkpoint(path, model, tok, extra={"snapshots": snapshots})
with open(path.replace(".pt", "_history.json"), "w") as f:
    json.dump(history, f)

# --- 3. Figures ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4))
steps_all = range(len(history["batch_loss"]))
ax.plot(steps_all, history["batch_loss"], color=MUTED, linewidth=1, alpha=0.5, label="single batch")
ax.plot(history["step"], history["train"], color=BLUE, marker="o", markersize=5, label="train (avg)")
ax.plot(history["step"], history["val"], color=ORANGE, marker="o", markersize=5, label="validation (avg)")
ax.axhline(torch.log(torch.tensor(float(cfg.vocab_size))).item(), color=INK2, linestyle=":", linewidth=1.5)
ax.text(tc.steps, 4.25, "random guessing = ln(65) ≈ 4.17", ha="right", color=INK2, fontsize=9)
ax.set_xlabel("training step")
ax.set_ylabel("cross-entropy loss")
ax.set_ylim(1.0, 4.5)
ax.set_title(f"Loss falls from 4.2 to {history['val'][-1]:.2f} in {history['seconds'] / 60:.0f} minutes on a CPU")
ax.legend()
save(fig, "06_loss_curve")

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot([lr_at(s, tc) for s in range(tc.steps)], color=BLUE)
ax.axvline(tc.warmup_steps, color=MUTED, linestyle="--", linewidth=1.2)
ax.text(tc.warmup_steps + 40, tc.lr * 0.95, "end of warmup", color=INK2, fontsize=9)
ax.set_xlabel("training step")
ax.set_ylabel("learning rate")
ax.set_title("Warmup, then cosine decay")
save(fig, "06_lr_schedule")
