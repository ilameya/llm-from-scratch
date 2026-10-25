"""Generate every diagram used in the blog series.

    python diagrams/make_diagrams.py      # writes diagrams/*.svg (+ .png if cairosvg is installed)
"""

from svgkit import FONT, INK, INK2, KINDS, LINE, MUTED, SVG

ORANGE_S, BLUE_S, AQUA_S, VIOLET_S, YELLOW_S = (KINDS[k][1] for k in ("orange", "blue", "aqua", "violet", "yellow"))


# ---------------------------------------------------------------------------- Part 1
def big_picture():
    s = SVG(1060, 272, "The big picture: text goes in, a probability for every possible next token comes out")
    cy = 115
    s.box(20, cy - 35, 140, 70, "Text", '"To be or no"')
    s.line(160, cy, 188, cy)
    s.box(190, cy - 35, 130, 70, "Tokenizer", "Part 1", mono_sub=False)
    s.line(320, cy, 348, cy)
    s.box(350, cy - 35, 170, 70, "Token ids", "[32, 53, 1, 40, ...]")
    s.line(520, cy, 548, cy)
    # the model
    s.rect(550, 20, 220, 192, "#fbfbfa", MUTED, dash="5 4")
    s.text(660, 38, "GPT", 15, 700)
    s.box(565, 54, 190, 38, "Embeddings", kind="blue", size=14)
    s.line(660, 92, 660, 102, arrow=True)
    s.box(565, 104, 190, 38, "Transformer blocks × N", kind="orange", size=14)
    s.line(660, 142, 660, 152, arrow=True)
    s.box(565, 154, 190, 38, "Output layer", kind="blue", size=14)
    s.text(660, 202, "Parts 2-5", 11.5, 400, INK2)
    s.line(770, cy, 798, cy)
    # probabilities mini bar chart
    s.text(858, 44, "next-token", 13, 600)
    s.text(858, 61, "probabilities", 13, 600)
    bars = [("t", 0.62), ("w", 0.16), ("b", 0.09), ("…", 0.05)]
    for i, (ch, p) in enumerate(bars):
        x = 810 + i * 26
        h = p * 90
        s.rect(x, 160 - h, 18, h, KINDS["blue"][1], KINDS["blue"][1], rx=3, sw=0)
        s.text(x + 9, 173, ch, 12, 400, INK2, mono=True)
        s.text(x + 9, 160 - h - 9, f"{p:.2f}".lstrip("0"), 10, 400, INK2)
    s.line(806, 160, 912, 160, LINE, 1, arrow=False)
    s.line(918, cy, 953, cy)
    s.text(935, cy - 14, "sample", 11, 400, INK2)
    s.box(955, cy - 35, 85, 70, "Next", '"t"')
    s.path(f"M 997 {cy + 35} L 997 236 L 90 236 L 90 {cy + 37}", MUTED, 1.6, dash="6 4")
    s.text(543, 254, "append the new token to the text and repeat (Part 7)", 13, 400, INK2)
    s.save("01_big_picture")


def bpe_merges():
    s = SVG(900, 330, "Byte Pair Encoding repeatedly merges the most frequent neighbouring pair")
    words = [["t", "h", "e"], ["t", "h", "i", "n"], ["t", "h", "e", "n"]]
    rows = [
        ("Start: one token per byte", "most frequent pair: (t, h), 3 times", words, ("t", "h")),
        ("After merge 1: t + h → th", "most frequent pair: (th, e), 2 times",
         [["th", "e"], ["th", "i", "n"], ["th", "e", "n"]], ("th", "e")),
        ("After merge 2: th + e → the", "11 tokens became 6",
         [["the"], ["th", "i", "n"], ["the", "n"]], None),
    ]
    cw, ch, gap = 40, 40, 26
    for r, (title, sub, ws, pair) in enumerate(rows):
        y = 40 + r * 100
        s.text(20, y + 12, title, 15, 600, anchor="start")
        s.text(20, y + 34, sub, 13, 400, INK2, anchor="start")
        x = 330
        for w in ws:
            for i, tok in enumerate(w):
                is_pair = pair is not None and i + 1 < len(w) and (tok, w[i + 1]) == pair
                was_pair = pair is not None and i > 0 and (w[i - 1], tok) == pair
                kind = "orange" if (is_pair or was_pair) else "blue"
                width = cw + 12 * (len(tok) - 1)
                s.cell(x, y, width, ch, tok, kind, size=16, weight=600 if len(tok) > 1 else 400)
                x += width + 4
            x += gap
        if r < 2:
            s.line(350, y + ch + 8, 350, y + 92, MUTED, 1.4)
    s.text(880, 322, "orange = the pair that gets merged next", 12, 400, INK2, anchor="end")
    s.save("01_bpe_merges")


# ---------------------------------------------------------------------------- Part 2
def embeddings():
    s = SVG(1000, 470, "Each token id selects a row of the token table; each position selects a row of the position table; the two are added")
    toks, ids = ["T", "o", "␣", "b", "e"], [32, 53, 1, 40, 43]
    labels = [(60, "token", None), (110, "token id", None),
              (160, "token embedding", "row [id] of a 65 × C table"),
              (262, "position embedding", "row [t] of a 128 × C table"),
              (380, "input vectors x", "x = token + position")]
    for y, t, sub in labels:
        s.text(30, y if not sub else y - 8, t, 15, 600, anchor="start")
        if sub:
            s.text(30, y + 12, sub, 12.5, 400, INK2, anchor="start", mono=True)
    n, cw, chh = 6, 16, 24
    for i, (tk, tid) in enumerate(zip(toks, ids)):
        cx = 380 + i * 120
        s.cell(cx - 22, 40, 44, 40, tk, "blue", 18, weight=600)
        s.text(cx, 110, str(tid), 15, 400, INK2, mono=True)
        s.line(cx, 122, cx, 144, MUTED, 1.4)
        x0 = cx - n * cw / 2
        for j in range(n):
            s.rect(x0 + j * cw, 148, cw, chh, KINDS["blue"][0], BLUE_S, rx=0, sw=1)
        s.plus(cx, 205)
        s.text(cx, 236, f"pos {i}", 12, 400, INK2, mono=True)
        for j in range(n):
            s.rect(x0 + j * cw, 250, cw, chh, KINDS["violet"][0], VIOLET_S, rx=0, sw=1)
        s.line(cx, 280, cx, 360, MUTED, 1.4)
        s.text(cx + 14, 320, "=", 16, 600, INK2)
        for j in range(n):
            s.rect(x0 + j * cw, 366, cw, chh, "#d6e6fa", BLUE_S, rx=0, sw=1.4)
    # shape bracket
    s.path("M 330 406 L 330 416 L 900 416 L 900 406", INK2, 1.4, arrow=False)
    s.text(615, 436, "x has shape (T, C): one C-dimensional vector per position, ready for the transformer",
           13, 400, INK2)
    s.save("02_embeddings")


# ---------------------------------------------------------------------------- Part 3
def _tri_grid(s, x, y, n, c, values=None, masked_text="−∞", show_tokens=None):
    for r in range(n):
        for k in range(n):
            if k > r:
                s.rect(x + k * c, y + r * c, c, c, "#f4f4f2", LINE, rx=0, sw=1)
                s.text(x + k * c + c / 2, y + r * c + c / 2, masked_text, 10, 400, MUTED)
            else:
                v = values[r][k] if values else 0.55
                s.rect(x + k * c, y + r * c, c, c, ORANGE_S, "#ffffff", rx=0, sw=1, opacity=0.18 + 0.75 * v)
                if values:
                    s.text(x + k * c + c / 2, y + r * c + c / 2, f"{v:.2f}".lstrip("0") if v < 1 else "1", 10.5,
                           600 if v > 0.45 else 400, "#ffffff" if v > 0.45 else INK)
    if show_tokens:
        for i, t in enumerate(show_tokens):
            s.text(x - 12, y + i * c + c / 2, t, 12, 600, INK2, mono=True)
            s.text(x + i * c + c / 2, y - 11, t, 12, 600, INK2, mono=True)


def _mat(s, x, y, rows, cols, cw, ch, kind):
    fill, stroke = KINDS[kind]
    s.rect(x, y, cols * cw, rows * ch, fill, stroke, rx=4, sw=1.5)
    for r in range(1, rows):
        s.line(x, y + r * ch, x + cols * cw, y + r * ch, stroke, 0.6, arrow=False)


def attention_flow():
    s = SVG(1110, 400, "Causal self-attention: queries meet keys, the mask hides the future, softmax turns scores into weights, weights mix the values")
    toks = ["T", "o", "␣", "b", "e"]
    # x
    _mat(s, 30, 160, 5, 6, 16, 18, "blue")
    s.text(78, 140, "x", 17, 700)
    s.text(78, 270, "(T, C)", 12.5, 400, INK2, mono=True)
    # Q K V
    for name, y, lab in [("Q", 40, "queries: what am I looking for?"), ("K", 165, "keys: what do I contain?"),
                         ("V", 290, "values: what I pass on")]:
        _mat(s, 210, y, 5, 4, 16, 18, "blue")
        s.text(291, y + 30, name, 18, 700, anchor="start")
        s.text(291, y + 52, lab, 11.5, 400, INK2, anchor="start")
        s.text(291, y + 70, "(T, d)", 11.5, 400, INK2, anchor="start", mono=True)
        s.path(f"M 126 205 C 165 205, 170 {y + 45}, 206 {y + 45}", MUTED, 1.4)
    s.text(168, 74, "× W_q", 11.5, 400, INK2, mono=True)
    s.text(166, 192, "× W_k", 11.5, 400, INK2, mono=True)
    s.text(186, 352, "× W_v", 11.5, 400, INK2, mono=True)
    # scores
    c = 30
    sx, sy = 570, 120
    s.path(f"M 310 70 L 480 70 C 530 70, 510 {sy + 45}, {sx - 22} {sy + 45}", MUTED, 1.4)
    s.path(f"M 310 195 L 480 195 C 530 195, 510 {sy + 105}, {sx - 22} {sy + 105}", MUTED, 1.4)
    _tri_grid(s, sx, sy, 5, c, None, show_tokens=toks)
    s.items.append(f'<text x="{sx + 75}" y="{sy - 45}" font-family="{FONT}" font-size="13" font-weight="600" fill="{INK}" text-anchor="middle" dominant-baseline="middle">scores = Q · K<tspan dy="-6" font-size="9">T</tspan><tspan dy="6"> / √d, masked</tspan></text>')
    s.text(sx + 75, sy + 172, "(T, T): how much each query", 11.5, 400, INK2)
    s.text(sx + 75, sy + 188, "matches each earlier key", 11.5, 400, INK2)
    # softmax
    s.line(sx + 158, sy + 75, sx + 205, sy + 75)
    s.text(sx + 182, sy + 60, "softmax", 12, 600, INK2)
    s.text(sx + 182, sy + 92, "per row", 11, 400, INK2)
    w = [[1.0, 0, 0, 0, 0], [0.62, 0.38, 0, 0, 0], [0.2, 0.5, 0.3, 0, 0],
         [0.1, 0.15, 0.45, 0.3, 0], [0.05, 0.25, 0.1, 0.2, 0.4]]
    wx = sx + 225
    _tri_grid(s, wx, sy, 5, c, w, masked_text="0", show_tokens=toks)
    s.text(wx + 75, sy - 45, "attention weights", 13, 600)
    s.text(wx + 75, sy + 172, "each row sums to 1", 11.5, 400, INK2)
    # output
    ox = wx + 205
    s.line(wx + 158, sy + 75, ox - 6, sy + 75)
    s.text(wx + 182, sy + 60, "× V", 12, 600, INK2, mono=True)
    _mat(s, ox, sy + 30, 5, 4, 16, 18, "orange")
    s.text(ox + 32, sy + 12, "output", 13, 600)
    s.text(ox + 32, sy + 140, "(T, d)", 11.5, 400, INK2, mono=True)
    s.path(f"M 310 320 L {ox + 32} 320 L {ox + 32} {sy + 158}", MUTED, 1.4, dash="5 4")
    s.text(700, 336, "the values V are mixed using the weights", 11.5, 400, INK2)
    s.save("03_attention_flow")


def lines(s, x, y, rows, size=12.5, color=INK2, anchor="middle", lh=18, weight=400, mono=False):
    for i, r in enumerate(rows):
        s.text(x, y + i * lh, r, size, weight, color, anchor=anchor, mono=mono)


# ---------------------------------------------------------------------------- Part 4
def transformer_block():
    s = SVG(820, 690, "A pre-norm transformer block: two sub-layers, each wrapped in a residual connection")
    X = 150
    s.text(X, 28, "input x   (B, T, C)", 14, 600, mono=True)
    s.line(X, 44, X, 640, BLUE_S, 4, arrow=False)
    s.path(f"M {X - 9} 636 L {X + 9} 636 L {X} 652 Z", BLUE_S, 1, arrow=False, fill=BLUE_S)
    s.text(X - 16, 380, "residual stream", 12, 600, BLUE_S, anchor="end")
    # sub-layer 1: attention
    s.path(f"M {X} 90 L 430 90 L 430 106", INK2, 1.6)
    s.box(300, 108, 260, 44, "LayerNorm", kind="yellow", size=15)
    s.line(430, 152, 430, 176)
    s.box(300, 178, 260, 64, "Masked multi-head", "self-attention", kind="orange", size=15, sub_size=15, mono_sub=False)
    s.path(f"M 430 242 L 430 272 L {X + 16} 272", INK2, 1.6)
    s.plus(X, 272)
    s.text(232, 258, "x + attn(ln1(x))", 12, 400, INK2, mono=True)
    lines(s, 585, 196, ["communicate: every token", "gathers information from", "the tokens before it"], 13, anchor="start")
    # sub-layer 2: MLP
    s.path(f"M {X} 330 L 430 330 L 430 346", INK2, 1.6)
    s.box(300, 348, 260, 44, "LayerNorm", kind="yellow", size=15)
    s.line(430, 392, 430, 414)
    s.rect(300, 416, 260, 150, KINDS["aqua"][0], AQUA_S)
    s.text(430, 434, "MLP (feed-forward)", 15, 600)
    for i, (lab, sub) in enumerate([("Linear", "C → 4C"), ("GELU", None), ("Linear", "4C → C")]):
        y = 450 + i * 38
        s.rect(330, y, 200, 30, "#ffffff", AQUA_S, rx=6, sw=1.2)
        s.text(430 if not sub else 400, y + 15, lab, 13, 600)
        if sub:
            s.text(480, y + 15, sub, 12, 400, INK2, mono=True)
    s.path(f"M 430 566 L 430 596 L {X + 16} 596", INK2, 1.6)
    s.plus(X, 596)
    s.text(232, 582, "x + mlp(ln2(x))", 12, 400, INK2, mono=True)
    lines(s, 585, 470, ["compute: every token is", "processed on its own,", "with the same weights"], 13, anchor="start")
    s.text(X, 672, "output   (B, T, C), same shape: blocks stack", 14, 600, mono=True, anchor="start")
    s.save("04_transformer_block")


def multihead():
    s = SVG(1000, 330, "Multi-head attention: several small heads look at the sequence in parallel, then their outputs are concatenated and mixed")
    s.box(20, 120, 110, 80, "x", "(T, 128)", size=18)
    for i in range(4):
        y = 30 + i * 70
        s.path(f"M 130 160 C 165 160, 170 {y + 25}, 206 {y + 25}", MUTED, 1.4)
        s.box(210, y, 220, 50, f"head {i + 1}", "attention on 32 dims", kind="orange", size=14, sub_size=12, mono_sub=False)
        s.path(f"M 430 {y + 25} C 465 {y + 25}, 470 160, 506 160", MUTED, 1.4)
    s.box(510, 115, 150, 90, "concatenate", "(T, 4 × 32 = 128)", kind="gray", size=14, sub_size=11.5)
    s.line(660, 160, 698, 160)
    s.box(700, 115, 140, 90, "Linear proj", "128 → 128", size=14)
    s.line(840, 160, 868, 160)
    s.box(870, 120, 110, 80, "output", "(T, 128)", kind="orange", size=15)
    s.text(320, 316, "each head has its own W_q, W_k, W_v and is free to look for a different pattern", 12.5, 400, INK2)
    s.save("04_multihead")


# ---------------------------------------------------------------------------- Part 5
def gpt_architecture():
    s = SVG(860, 800, "The full GPT: embeddings, N transformer blocks, a final LayerNorm and a linear layer that scores every token in the vocabulary")
    cx = 330
    shapes = []
    s.box(cx - 120, 20, 240, 52, "token ids", "e.g. [32, 53, 1, 40, 43]", size=15)
    shapes.append((46, "(B, T)"))
    s.path(f"M {cx - 40} 72 L {cx - 90} 102", INK2, 1.5)
    s.path(f"M {cx + 40} 72 L {cx + 90} 102", INK2, 1.5)
    s.box(cx - 205, 104, 190, 56, "token embedding", "table: V × C", size=14)
    s.box(cx + 15, 104, 190, 56, "position embedding", "table: block_size × C", kind="violet", size=14)
    s.path(f"M {cx - 110} 160 L {cx - 12} 190", INK2, 1.5)
    s.path(f"M {cx + 110} 160 L {cx + 12} 190", INK2, 1.5)
    s.plus(cx, 200)
    shapes.append((200, "(B, T, C)"))
    s.line(cx, 213, cx, 236)
    s.box(cx - 70, 238, 140, 34, "dropout", kind="gray", size=13)
    s.line(cx, 272, cx, 296)
    # blocks
    s.rect(cx - 170, 298, 340, 188, "#fbfbfa", MUTED, dash="5 4")
    s.text(cx + 160, 316, "× N", 16, 700, anchor="end")
    s.text(cx - 155, 316, "transformer block (Part 4)", 12.5, 600, INK2, anchor="start")
    s.box(cx - 140, 332, 280, 34, "LayerNorm → masked multi-head attention", kind="orange", size=13, weight=600)
    s.text(cx, 382, "+ residual", 12, 400, INK2)
    s.box(cx - 140, 398, 280, 34, "LayerNorm → MLP", kind="aqua", size=13, weight=600)
    s.text(cx, 448, "+ residual", 12, 400, INK2)
    shapes.append((392, "(B, T, C)"))
    s.line(cx, 486, cx, 512)
    s.box(cx - 100, 514, 200, 40, "final LayerNorm", kind="yellow", size=14)
    shapes.append((534, "(B, T, C)"))
    s.line(cx, 554, cx, 580)
    s.box(cx - 100, 582, 200, 52, "lm_head (Linear)", "C → V", size=14)
    shapes.append((608, "(B, T, V)  logits"))
    s.line(cx, 634, cx, 660)
    s.box(cx - 100, 662, 200, 52, "softmax", "probabilities", kind="gray", size=14, mono_sub=False)
    shapes.append((688, "(B, T, V)"))
    # weight tying
    s.path(f"M {cx - 205} 132 C {cx - 300} 132, {cx - 300} 608, {cx - 104} 608", VIOLET_S, 1.6, dash="6 4")
    lines(s, 76, 366, ["weight tying:", "the same", "matrix"], 12, VIOLET_S, anchor="start", lh=17, weight=600)
    # shape column
    s.text(640, 22, "tensor shape", 13, 600, INK2, anchor="start")
    for y, t in shapes:
        s.line(560, y, 630, y, LINE, 1, dash="3 3", arrow=False)
        s.text(640, y, t, 13, 400, INK, anchor="start", mono=True)
    lines(s, 640, 742, ["B = batch, T = tokens,", "C = width, V = vocab size"], 12, INK2, anchor="start", lh=17)
    s.text(cx, 760, "every position predicts the token that comes after it", 13, 400, INK2)
    s.save("05_gpt_architecture")


# ---------------------------------------------------------------------------- Part 6
def next_token_targets():
    s = SVG(900, 270, "Targets are the inputs shifted by one: every position learns to predict the next character")
    xs = list("First Ci")
    ys = list("irst Cit")
    cw = 62
    x0 = 190
    s.text(30, 70, "input x", 15, 600, anchor="start")
    s.text(30, 170, "target y", 15, 600, anchor="start")
    for i, (a, b) in enumerate(zip(xs, ys)):
        x = x0 + i * cw
        ctx = i <= 3
        s.cell(x, 50, 54, 40, {" ": "␣"}.get(a, a), "orange" if ctx else "blue", 17)
        s.line(x + 27, 92, x + 27, 146, MUTED, 1.3)
        s.cell(x, 150, 54, 40, {" ": "␣"}.get(b, b), "orange" if i == 3 else "gray", 17)
        s.text(x + 27, 30, str(i), 11.5, 400, INK2, mono=True)
    s.text(x0 - 6, 30, "position", 11.5, 400, INK2, anchor="end")
    s.text(450, 230, "at position 3 the model sees \"Firs\" (orange) and must predict \"t\"; one window = 8 examples",
           13, 400, INK2)
    s.save("06_next_token_targets")


def training_loop():
    s = SVG(960, 270, "One training step, repeated thousands of times")
    steps = [("1. sample a batch", "x, y = get_batch()", "blue"),
             ("2. forward pass", "logits = model(x)", "blue"),
             ("3. loss", "cross_entropy(logits, y)", "violet"),
             ("4. backward pass", "loss.backward()", "violet"),
             ("5. update weights", "opt.step()", "violet")]
    w, gap = 160, 30
    for i, (t, code, kind) in enumerate(steps):
        x = 20 + i * (w + gap)
        s.box(x, 50, w, 70, t, code, kind=kind, size=14, sub_size=11)
        if i < 4:
            s.line(x + w, 85, x + w + gap - 2, 85)
    notes = ["random windows of text", "predict every next token", "how surprised were we?",
             "gradient for every weight", "AdamW + LR schedule"]
    for i, n in enumerate(notes):
        s.text(20 + i * (w + gap) + w / 2, 140, n, 12, 400, INK2)
    x_last = 20 + 4 * (w + gap) + w / 2
    s.path(f"M {x_last} 158 L {x_last} 205 L 8 205 L 8 85 L 18 85", MUTED, 1.6, dash="6 4")
    s.text(480, 230, "repeat (3,000 steps for our model): the loss goes down, the samples get better", 13, 400, INK2)
    s.save("06_training_loop")


# ---------------------------------------------------------------------------- Part 7
def kv_cache():
    s = SVG(1000, 470, "Without a KV cache every step re-processes the whole sequence; with a cache only the newest token is processed")
    toks = list("ROMEO")
    for panel, (x0, title, sub, cached) in enumerate([
            (30, "without KV cache", "every step recomputes all tokens", False),
            (530, "with KV cache", "earlier keys and values are reused", True)]):
        s.text(x0, 32, title, 17, 700, anchor="start")
        s.text(x0, 56, sub, 13, 400, INK2, anchor="start")
        for step in range(5):
            y = 84 + step * 58
            s.text(x0, y + 22, f"step {step + 1}", 12.5, 400, INK2, anchor="start", mono=True)
            for j in range(step + 1):
                new = j == step
                kind = "orange" if (new or not cached) else "gray"
                s.cell(x0 + 80 + j * 52, y, 46, 44, toks[j], kind, 16, weight=600 if new else 400)
            s.line(x0 + 80 + (step + 1) * 52 + 2, y + 22, x0 + 80 + (step + 1) * 52 + 26, y + 22, MUTED, 1.3)
            s.text(x0 + 80 + (step + 1) * 52 + 42, y + 22, "?", 16, 700, INK2)
        total = "1 + 2 + 3 + 4 + 5 = 15 token passes" if not cached else "1 + 1 + 1 + 1 + 1 = 5 token passes"
        s.text(x0, 392, total, 13.5, 600, anchor="start")
        s.text(x0, 414, "grows like T² for long texts" if not cached else "grows like T, at the cost of memory for K, V",
               12.5, 400, INK2, anchor="start")
    s.cell(30, 440, 20, 18, "", "orange")
    s.text(56, 449, "computed this step", 12, 400, INK2, anchor="start")
    s.cell(210, 440, 20, 18, "", "gray")
    s.text(236, 449, "read from the cache", 12, 400, INK2, anchor="start")
    s.save("07_kv_cache")


# ---------------------------------------------------------------------------- Part 8
def roadmap():
    s = SVG(1040, 420, "From our tiny GPT to a modern chat assistant")
    stages = [
        ("Our tiny GPT", ["0.8M parameters", "1 MB of Shakespeare", "characters as tokens"], "blue"),
        ("Pretraining at scale", ["billions of parameters", "trillions of tokens", "BPE vocab 50k-200k"], "blue"),
        ("Instruction tuning", ["supervised fine-tuning", "on example conversations"], "aqua"),
        ("Preference tuning", ["RLHF or DPO: learn which", "answers people prefer"], "violet"),
        ("Assistant", ["answers questions,", "follows instructions"], "orange"),
    ]
    w, gap = 172, 33
    for i, (t, rows, kind) in enumerate(stages):
        x = 20 + i * (w + gap)
        fill, stroke = KINDS[kind]
        s.rect(x, 40, w, 130, fill, stroke)
        s.text(x + w / 2, 66, t, 15, 700)
        lines(s, x + w / 2, 98, rows, 12.5, INK2, lh=19)
        if i < 4:
            s.line(x + w, 105, x + w + gap - 2, 105)
    s.text(20 + w / 2, 188, "you are here", 12.5, 700, BLUE_S)
    s.text(20 + 1.5 * w + gap, 188, "same architecture, same loss", 12.5, 400, INK2)
    # architecture upgrades
    s.rect(20, 220, 1000, 180, "#fbfbfa", MUTED, dash="5 4")
    s.text(40, 246, "What modern LLMs change inside the architecture", 15, 700, anchor="start")
    chips = [("RoPE", "rotary positions"), ("RMSNorm", "simpler normalisation"), ("SwiGLU", "gated MLP"),
             ("Grouped-query attention", "smaller KV cache"), ("FlashAttention", "faster, same maths"),
             ("Mixture of Experts", "route tokens to experts")]
    for i, (a, b) in enumerate(chips):
        col, row = i % 3, i // 3
        x, y = 40 + col * 325, 270 + row * 62
        s.box(x, y, 305, 50, a, b, kind="white", size=14, sub_size=12, mono_sub=False)
    s.save("08_roadmap")


if __name__ == "__main__":
    big_picture()
    bpe_merges()
    embeddings()
    attention_flow()
    transformer_block()
    multihead()
    gpt_architecture()
    next_token_targets()
    training_loop()
    kv_cache()
    roadmap()
