"""A tiny SVG helper so all diagrams in the series share one visual language.

Colour roles (used consistently across every post):
    blue   = tokens, ids, embeddings, data flowing through the model
    orange = attention
    aqua   = MLP / feed-forward
    yellow = normalisation
    violet = loss, gradients, optimisation
    gray   = neutral / inactive / cached
"""

from __future__ import annotations

import os
from html import escape

FONT = "Helvetica, Arial, 'DejaVu Sans', sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2"

KINDS = {  # (fill, stroke)
    "blue": ("#e8f1fc", "#2a78d6"),
    "orange": ("#fdebe3", "#eb6834"),
    "aqua": ("#e0f5ed", "#1baf7a"),
    "yellow": ("#fdf3dc", "#c98500"),
    "violet": ("#ecebfb", "#6250d6"),
    "gray": ("#f4f4f2", "#8a8984"),
    "white": ("#ffffff", "#8a8984"),
}
HERE = os.path.dirname(os.path.abspath(__file__))


class SVG:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title = w, h, title
        self.items: list[str] = []
        self.markers: set[str] = set()

    # -- primitives -------------------------------------------------------------
    def rect(self, x, y, w, h, fill="#fff", stroke=MUTED, rx=10, sw=1.5, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def text(self, x, y, s, size=15, weight=400, color=INK, anchor="middle", mono=False, italic=False):
        fam = MONO if mono else FONT
        it = ' font-style="italic"' if italic else ""
        self.items.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{color}" text-anchor="{anchor}" dominant-baseline="middle"{it}>{escape(s)}</text>')

    def _marker(self, color):
        mid = "arrow" + color.strip("#")
        self.markers.add(color)
        return mid

    def line(self, x1, y1, x2, y2, color=INK2, sw=1.8, dash=None, arrow=True):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{self._marker(color)})"' if arrow else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d}{m}/>')

    def path(self, d, color=INK2, sw=1.8, dash=None, arrow=True, fill="none"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{self._marker(color)})"' if arrow else ""
        self.items.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{da}{m}/>')

    def circle(self, cx, cy, r, fill="#fff", stroke=INK2, sw=1.8):
        self.items.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def plus(self, cx, cy, r=13, color=INK2):
        self.circle(cx, cy, r, "#fff", color)
        self.line(cx - r * 0.55, cy, cx + r * 0.55, cy, color, 2, arrow=False)
        self.line(cx, cy - r * 0.55, cx, cy + r * 0.55, color, 2, arrow=False)

    # -- composites -------------------------------------------------------------
    def box(self, x, y, w, h, label, sub=None, kind="blue", size=16, sub_size=12.5, weight=600, mono_sub=True):
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke)
        if sub:
            self.text(x + w / 2, y + h / 2 - 9, label, size, weight)
            self.text(x + w / 2, y + h / 2 + 11, sub, sub_size, 400, INK2, mono=mono_sub)
        else:
            self.text(x + w / 2, y + h / 2, label, size, weight)

    def cell(self, x, y, w, h, label="", kind="blue", size=14, mono=True, weight=400, color=INK):
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke, rx=4, sw=1.2)
        if label != "":
            self.text(x + w / 2, y + h / 2 + 0.5, label, size, weight, color, mono=mono)

    def grid(self, x, y, rows, cols, cw, ch, kind="blue", highlight_rows=(), hkind="blue", labels=None):
        fill, stroke = KINDS[kind]
        for r in range(rows):
            for c in range(cols):
                k = hkind if r in highlight_rows else kind
                f, s = KINDS[k]
                if r not in highlight_rows:
                    f = "#fafaf8"
                    s = LINE
                self.rect(x + c * cw, y + r * ch, cw, ch, f, s, rx=0, sw=1)
                if labels is not None and labels[r][c] != "":
                    self.text(x + c * cw + cw / 2, y + r * ch + ch / 2, labels[r][c], 11, 400, INK2, mono=True)

    # -- output -----------------------------------------------------------------
    def svg(self) -> str:
        defs = "".join(
            f'<marker id="arrow{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in sorted(self.markers))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{escape(self.title)}">'
                f'<title>{escape(self.title)}</title><defs>{defs}</defs>'
                f'<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.items) + "</svg>")

    def save(self, name: str, png: bool = True):
        out = os.path.join(HERE, f"{name}.svg")
        with open(out, "w", encoding="utf-8") as f:
            f.write(self.svg())
        if png:
            try:
                import cairosvg
                cairosvg.svg2png(bytestring=self.svg().encode(), write_to=out.replace(".svg", ".png"), scale=2)
            except ImportError:
                pass
        print("saved", out)
