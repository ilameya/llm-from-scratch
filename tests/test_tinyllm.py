"""Sanity checks. Run with:  python -m pytest -q"""

import math

import torch

from tinyllm import GPT, BPETokenizer, CharTokenizer, GPTConfig
from tinyllm.sample import filter_logits, generate
from tinyllm.tokenizer import tokenizer_from_dict

TEXT = "To be, or not to be, that is the question.\nWhether 'tis nobler in the mind to suffer"


def tiny_model(vocab=40, block=16):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=32, dropout=0.0))


def test_char_roundtrip():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT
    assert tok.vocab_size == len(set(TEXT))


def test_bpe_roundtrip_and_compression():
    tok = BPETokenizer()
    tok.train(TEXT * 5, vocab_size=280)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8"))           # merges must compress
    unseen = "Héllo wörld 🙂"                                # bytes cover any text
    assert tok.decode(tok.encode(unseen)) == unseen
    again = tokenizer_from_dict(tok.to_dict())
    assert again.encode(TEXT) == ids


def test_shapes_and_initial_loss():
    model = tiny_model()
    x = torch.randint(0, 40, (3, 16))
    y = torch.randint(0, 40, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 40)
    # An untrained model should be close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - math.log(40)) < 0.3


def test_causality():
    """Changing a future token must not change predictions for earlier positions."""
    model = tiny_model().eval()
    x = torch.randint(0, 40, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 40
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_can_overfit_one_batch():
    model = tiny_model()
    x = torch.randint(0, 40, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2


def test_kv_cache_matches_full_recompute():
    model = tiny_model(block=32).eval()
    prompt = torch.randint(0, 40, (2, 5))
    a = generate(model, prompt, 20, greedy=True, use_cache=False)
    b = generate(model, prompt, 20, greedy=True, use_cache=True)
    assert torch.equal(a, b)


def test_top_k_and_top_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    k = filter_logits(logits, top_k=2)
    assert torch.isinf(k[0, 2:]).all() and not torch.isinf(k[0, :2]).any()
    p = filter_logits(logits, top_p=0.7)            # 0.5 + 0.3 reaches 0.7 -> keep 2
    assert torch.isinf(p[0, 2:]).all() and not torch.isinf(p[0, :2]).any()


def test_modern_blocks():
    from tinyllm.modern import RMSNorm, SwiGLUMLP
    x = torch.randn(2, 5, 32) * 7
    y = RMSNorm(32)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 5), atol=1e-3)
    cfg = GPTConfig(n_embd=32, dropout=0.0)
    assert SwiGLUMLP(cfg)(x).shape == x.shape
