"""Part 1 - From text to numbers: tokenization.

    python parts/01_tokenization.py
"""

import time

import matplotlib.pyplot as plt

from _style import BLUE, INK2, ORANGE, save
from tinyllm.data import load_shakespeare
from tinyllm.tokenizer import BPETokenizer, CharTokenizer

text = load_shakespeare()
print(f"Dataset: {len(text):,} characters\n")
print(text[:200], "\n...")

# --- 1. Character-level tokenizer ------------------------------------------
char_tok = CharTokenizer.from_text(text)
print(f"\n[char] vocabulary size: {char_tok.vocab_size}")
print(f"[char] vocabulary: {''.join(char_tok.chars)!r}")
ids = char_tok.encode("Hello there!")
print(f"[char] encode('Hello there!') -> {ids}")
print(f"[char] decode(...)            -> {char_tok.decode(ids)!r}")

# --- 2. Bytes: what every string really is ---------------------------------
s = "café 🙂"
print(f"\n{s!r} has {len(s)} characters but {len(s.encode('utf-8'))} UTF-8 bytes: {list(s.encode('utf-8'))}")

# --- 3. Byte Pair Encoding, one merge at a time -----------------------------
print("\n[bpe] first 12 merges learned from Shakespeare:")
bpe = BPETokenizer()
bpe.train(text, vocab_size=256 + 12, verbose=True)

print("\n[bpe] training a 512-token vocabulary ...")
t0 = time.time()
bpe = BPETokenizer()
bpe.train(text, vocab_size=512)
print(f"[bpe] done in {time.time() - t0:.1f}s")

sentence = "ROMEO: But, soft! what light through yonder window breaks?"
pieces = [bpe.decode([i]) for i in bpe.encode(sentence)]
print(f"\n[bpe] {sentence!r}")
print(f"[bpe] -> {len(pieces)} tokens: " + "|".join(pieces))
print(f"[char] -> {len(char_tok.encode(sentence))} tokens")
assert bpe.decode(bpe.encode(sentence)) == sentence

longest = sorted(bpe.vocab.values(), key=len, reverse=True)[:10]
print("[bpe] longest tokens:", [t.decode("utf-8", "replace") for t in longest])

# --- 4. Figure: vocabulary size vs. sequence length -------------------------
sample = text[:100_000]
n_bytes = len(sample.encode("utf-8"))
sizes = [256, 320, 384, 512, 768, 1024]
ratios = []
for v in sizes:
    tok = BPETokenizer()
    tok.train(text, vocab_size=v)
    ratios.append(n_bytes / len(tok.encode(sample)))
    print(f"[bpe] vocab {v:5d}: {ratios[-1]:.2f} characters per token")

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(sizes, ratios, color=BLUE, marker="o", markersize=8)
ax.axhline(1.0, color=ORANGE, linestyle="--", linewidth=2)
ax.text(1024, 1.06, "character-level = 1.0", color=INK2, ha="right")
for v, r in zip(sizes, ratios):
    ax.annotate(f"{r:.2f}", (v, r), textcoords="offset points", xytext=(0, 10), ha="center", color=INK2)
ax.set_xlabel("BPE vocabulary size")
ax.set_ylabel("characters per token")
ax.set_title("Bigger vocabularies pack more text into each token")
ax.set_ylim(0.8, max(ratios) + 0.5)
save(fig, "01_bpe_compression")
