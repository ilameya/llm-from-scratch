# LLMs from Scratch

**Build a small GPT, one piece at a time, and understand every line.**

This repository is the companion code for the blog series
[*LLMs from Scratch*](https://ilameya.github.io/blog/llm-from-scratch/) by
[Lameya Islam](https://ilameya.github.io). It is written for students: you need
some Python and a little linear algebra (matrix multiplication), and nothing else.
By the end you will have trained a language model on Shakespeare on your own laptop
and you will know what happens inside it, from raw text to generated text.

```
ROMEO:
Why more know self it one means.

PAULINA:
And My pourty his treath? hear think the seem.
```
*Our 0.8M-parameter model after 15 minutes of training on a small CPU. Not Shakespeare yet, but it learned
character names, line breaks, spelling and the shape of a play from nothing but characters.*

## The series

| # | Post | What you build | Script |
|---|------|----------------|--------|
| 1 | [From text to numbers: tokenization](https://ilameya.github.io/blog/llm-from-scratch/part-1-tokenization/) | character tokenizer, byte-level BPE | `parts/01_tokenization.py` |
| 2 | [Embeddings: giving tokens meaning and position](https://ilameya.github.io/blog/llm-from-scratch/part-2-embeddings/) | token + position embeddings | `parts/02_embeddings.py` |
| 3 | [Self-attention, step by step](https://ilameya.github.io/blog/llm-from-scratch/part-3-attention/) | causal self-attention | `parts/03_attention.py` |
| 4 | [The transformer block](https://ilameya.github.io/blog/llm-from-scratch/part-4-transformer-block/) | multi-head attention, MLP, residuals, LayerNorm | `parts/04_transformer_block.py` |
| 5 | [Assembling GPT](https://ilameya.github.io/blog/llm-from-scratch/part-5-gpt-model/) | the full model, shape by shape | `parts/05_gpt_model.py` |
| 6 | [Training: from noise to Shakespeare](https://ilameya.github.io/blog/llm-from-scratch/part-6-training/) | the training loop, AdamW, LR schedule | `parts/06_training.py` |
| 7 | [Generating text: sampling and the KV cache](https://ilameya.github.io/blog/llm-from-scratch/part-7-sampling/) | greedy, temperature, top-k, top-p, KV cache | `parts/07_sampling.py` |
| 8 | [Scaling up, and where to go next](https://ilameya.github.io/blog/llm-from-scratch/part-8-scaling/) | a mini scaling experiment | `parts/08_scaling.py` |

## Quick start

```bash
git clone https://github.com/ilameya/llm-from-scratch.git
cd llm-from-scratch
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                  # 8 sanity checks, a few seconds
python -m tinyllm.train              # train the default model (0.8M params, ~15 min on a 2-core CPU)
python -m tinyllm.sample --prompt "JULIET:" --temperature 0.8
```

The dataset (Tiny Shakespeare, ~1 MB, public domain) is already in `data/`. Everything
runs on a laptop CPU. With an NVIDIA GPU or Apple Silicon the code picks `cuda` or
`mps` automatically.

**No GPU?** Open [`notebooks/train_on_colab.ipynb`](notebooks/train_on_colab.ipynb) in
Google Colab (free GPU) to train a ~10.7M-parameter model in about half an hour.

## Repository layout

```
tinyllm/              the library, under 700 lines including comments
  tokenizer.py        CharTokenizer and a byte-level BPETokenizer
  model.py            GPTConfig, CausalSelfAttention, MLP, Block, GPT
  data.py             loading text, train/val split, random batches
  train.py            the training loop (python -m tinyllm.train)
  sample.py           temperature / top-k / top-p sampling, KV cache (python -m tinyllm.sample)
  utils.py            device selection, checkpoints
  modern.py           RMSNorm and SwiGLU, for the Part 8 exercises
parts/                one runnable script per blog post; each prints its results and saves its figures
tests/                sanity checks: round-trips, causality, overfitting, KV cache == no cache
figures/              figures produced by the scripts (used in the posts)
diagrams/             the hand-drawn SVG diagrams from the posts
notebooks/            Colab notebook for GPU training
```

## Run the parts in order

```bash
python parts/01_tokenization.py      # ~1 min
python parts/06_training.py          # ~15 min on a 2-core CPU (faster on most laptops); saves checkpoints/shakespeare_char.pt
python parts/02_embeddings.py        # needs the checkpoint for its last two figures
python parts/03_attention.py         # needs the checkpoint for the attention heatmap
python parts/04_transformer_block.py # ablation: ~10 min (use --quick for ~4 min)
python parts/05_gpt_model.py
python parts/07_sampling.py          # needs the checkpoint
python parts/08_scaling.py           # ~15 min (use --quick)
```

Training first is optional for Parts 1, 4, 5 and 8; the others load the trained model.

## Bigger runs

```bash
# char-level, ~10.7M params, block 256: use a GPU (Colab) for this one
python -m tinyllm.train --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 \
    --batch_size 64 --lr 6e-4 --steps 5000 --dropout 0.2

# BPE tokens instead of characters
python -m tinyllm.train --tokenizer bpe --bpe_vocab 512
python -m tinyllm.sample --ckpt checkpoints/shakespeare_bpe.pt
```

## Design choices (and why)

- **Hand-written attention.** `model.py` computes `softmax(QKᵀ/√d)V` explicitly so you can
  print and plot every intermediate tensor. In production you would call
  `torch.nn.functional.scaled_dot_product_attention`, which is the same maths, faster.
- **PyTorch for autograd only.** We use `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and the
  AdamW optimizer, but every architectural decision is written out.
- **GPT-2 conventions.** Pre-norm blocks, learned position embeddings, GELU, weight tying and
  GPT-2's initialisation. Part 8 lists what newer models (Llama, Mistral, Qwen) change.

## Credits and further reading

This project stands on the shoulders of excellent teaching material, especially
Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
"Let's build GPT" lecture, Sebastian Raschka's *Build a Large Language Model (From Scratch)*,
Jay Alammar's *The Illustrated Transformer*, and the original paper
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).
Tiny Shakespeare comes from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## License

MIT, see [LICENSE](LICENSE). Use it for your courses, study groups and projects.
If you spot a mistake or have a question, please open an issue.
