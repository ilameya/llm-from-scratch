"""Part 7 - Generation: greedy, temperature, top-k, top-p, and the KV cache.

    python parts/07_sampling.py        (needs: python -m tinyllm.train)
"""

import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _style import BLUE, INK2, need_checkpoint, save
from tinyllm.sample import filter_logits, generate
from tinyllm.utils import load_checkpoint

model, tok = load_checkpoint(need_checkpoint())
prompt_text = "ROMEO:\nI will "
prompt = torch.tensor([tok.encode(prompt_text)])

# --- 1. The raw next-token distribution ----------------------------------------
with torch.no_grad():
    logits, _ = model(prompt)
last = logits[:, -1, :]
probs = F.softmax(last, dim=-1)[0]
top = torch.topk(probs, 8)
print(f"after {prompt_text!r} the model predicts:")
for p, i in zip(top.values, top.indices):
    print(f"  {tok.decode([i.item()])!r:6} {p:.3f}")

# --- 2. Temperature reshapes the distribution ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
order = torch.argsort(probs, descending=True)[:10]
labels = [{" ": "␣", "\n": "\\n"}.get(tok.decode([i.item()]), tok.decode([i.item()])) for i in order]
for ax, temp in zip(axes, [0.5, 1.0, 1.5]):
    p = F.softmax(filter_logits(last, temperature=temp), dim=-1)[0][order]
    ax.bar(range(10), p, color=BLUE, width=0.65)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(10), labels)
    ax.set_title(f"temperature {temp}")
    ax.text(9.4, 0.36, f"top token: {p[0]:.0%}", ha="right", color=INK2, fontsize=10)
axes[0].set_ylabel("probability")
axes[0].set_ylim(0, 0.4)
fig.suptitle(f"Next character after \"I will \" at three temperatures", x=0.01, ha="left",
             fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "07_temperature")

# --- 3. Same prompt, different decoding strategies ---------------------------------
settings = [
    ("greedy", dict(greedy=True)),
    ("temperature 0.5", dict(temperature=0.5)),
    ("temperature 1.0", dict(temperature=1.0)),
    ("temperature 1.5", dict(temperature=1.5)),
    ("top-k 10", dict(top_k=10)),
    ("top-p 0.9", dict(top_p=0.9)),
]
for name, kw in settings:
    g = torch.Generator().manual_seed(7)
    out = generate(model, prompt, 150, generator=g, **kw)
    print(f"\n===== {name} =====\n{tok.decode(out[0].tolist())}")

# --- 4. KV cache: same output, less work ----------------------------------------------
n_new = model.cfg.block_size - prompt.size(1)
timings = {}
for use_cache in (False, True):
    t0 = time.perf_counter()
    for _ in range(3):
        out = generate(model, prompt, n_new, greedy=True, use_cache=use_cache)
    timings[use_cache] = (time.perf_counter() - t0) / 3
    print(f"\nuse_cache={use_cache}: {n_new} tokens in {timings[use_cache] * 1000:.0f} ms")
a = generate(model, prompt, n_new, greedy=True, use_cache=False)
b = generate(model, prompt, n_new, greedy=True, use_cache=True)
print("identical outputs:", torch.equal(a, b), f"| speed-up {timings[False] / timings[True]:.1f}x")
