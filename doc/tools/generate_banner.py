#!/usr/bin/env python3
"""Generate the banner background for the NEST GPU documentation header.

Create an SVG of the banner showing a random network blending into a field of GPUs.

Usage:
    python3 doc/tools/generate_banner.py [OUTPUT.svg]

with the default output being doc/htmldoc/static/img/banner.svg.

The final figure is committed to the repository for convenience;
rerun this script only when changes are needed.
"""
import math
import random
import sys
from pathlib import Path

DEFAULT_OUT = (Path(__file__).resolve().parent.parent
               / "htmldoc" / "static" / "img" / "banner.svg")

W_DESIGN = 1848     # reference header width: what is seen at normal zoom
W_CANVAS = 3600     # total canvas: the reserve revealed when zooming out

# Height of the whole header zone, which this banner spans as one image: the
# 180px of #header plus the 2.4rem tab strip below it (48px at the theme's
# base font-size of 125%). Must stay in step with --banner-h in custom.css.
H = 228

ORANGE = "#ff6633"  # the logo orange
INK = "#12222c"     # deep slate ground

BLEND_X0, BLEND_X1 = 900, 1290   # overlap zone: network out, cards in

# Change NET_SEED for a different draw of neuron positions and connections;
# CARD_SEED only affects the brightness of individual GPU cards.
NET_SEED = 23
CARD_SEED = 4


def ramp(x, x0, x1):
    """0 below x0, 1 above x1, linear between."""
    return min(1.0, max(0.0, (x - x0) / (x1 - x0)))


def network(seed=NET_SEED, col=ORANGE, edge_col="#ffffff", edge_op=0.22):
    """Irregular network of neurons; density and opacity fall away across the
    blend zone so it dissolves into the card field instead of stopping."""
    r = random.Random(seed)
    edges, nodes, pts = [], [], []
    # rows scale with the band height, so density stays put if H changes
    cols, rows = 22, max(2, round(H / 26))
    for j in range(rows):
        for i in range(cols):
            x = 40 + i * (BLEND_X1 - 60) / (cols - 1) + r.uniform(-24, 24)
            y = 22 + j * (H - 44) / (rows - 1) + r.uniform(-24, 24)
            w = 1.0 - ramp(x, BLEND_X0, BLEND_X1)
            if w <= 0.02 or r.random() > 0.35 + 0.65 * w:
                continue
            pts.append((x, y, w))
    for a, (x1, y1, w1) in enumerate(pts):
        for x2, y2, w2 in pts[a + 1:]:
            if math.hypot(x2 - x1, y2 - y1) < 62 and r.random() < 0.30:
                op = edge_op * min(w1, w2)
                if op >= 0.02:
                    edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}"'
                                 f' x2="{x2:.1f}" y2="{y2:.1f}"'
                                 f' opacity="{op:.2f}"/>')
    for x, y, w in pts:
        nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}"'
                     f' r="{r.uniform(2.0, 4.4):.1f}"'
                     f' opacity="{r.uniform(0.35, 0.95) * w:.2f}"/>')
    # The colours are the same for every edge and every node, so they are
    # carried by a group instead of being repeated on each element; likewise
    # stroke-width, whose SVG default is already 1.
    return ([f'<g stroke="{edge_col}">'] + edges + ['</g>']
            + [f'<g fill="{col}">'] + nodes + ['</g>'])


def iso_cards(seed=CARD_SEED, col=ORANGE):
    """Isometric field of GPU cards: fades in across the blend zone, then
    continues to the right edge of the canvas."""
    r = random.Random(seed)
    out = []
    cw, ch, dep = 58, 20, 26
    cols = int((W_CANVAS - BLEND_X0) / (cw + 10)) + 8
    # enough rows to fill the band, plus one running off the bottom edge
    rows = int((H - 14) / 25) + 1
    for row in range(rows):
        for c in range(cols):
            x = BLEND_X0 + c * (cw + 10) - row * 30
            y = 14 + row * 25
            if x < BLEND_X0 - 120 or x > W_CANVAS + cw:
                continue
            w = ramp(x, BLEND_X0, BLEND_X1)
            if w <= 0.02:
                continue
            op = r.uniform(0.18, 0.55) * w
            faces = (
                (f'M{x:.0f},{y:.0f} l{cw},0 l{dep},-{ch} l-{cw},0 Z', 0.50),
                (f'M{x:.0f},{y:.0f} l{cw},0 l0,{ch} l-{cw},0 Z', 0.18),
                (f'M{x + cw:.0f},{y:.0f} l{dep},-{ch} l0,{ch} l-{dep},{ch} Z',
                 0.30),
            )
            for d, shade in faces:
                out.append(f'<path d="{d}" opacity="{op * shade:.2f}"'
                           f' stroke-opacity="{op:.2f}"/>')
    # Every face shares fill, stroke and stroke-width, so the group carries
    # them once rather than all paths carrying them each.
    return [f'<g fill="{col}" stroke="{col}">'] + out + ['</g>']


def banner():
    body = [f'<rect width="{W_CANVAS}" height="{H}" fill="{INK}"/>']
    body += network()
    body += iso_cards()
    body += [
        f'<defs><linearGradient id="fade" gradientUnits="userSpaceOnUse"'
        f' x1="0" y1="0" x2="{W_DESIGN}" y2="0">'
        f'<stop offset="0" stop-color="{INK}" stop-opacity="0.94"/>'
        f'<stop offset="0.30" stop-color="{INK}" stop-opacity="0.30"/>'
        f'<stop offset="1" stop-color="{INK}" stop-opacity="0"/>'
        f'</linearGradient></defs>',
        f'<rect width="{W_CANVAS}" height="{H}" fill="url(#fade)"/>',
    ]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg"'
            f' viewBox="0 0 {W_CANVAS} {H}"'
            f' width="{W_CANVAS}" height="{H}"'
            f' preserveAspectRatio="xMinYMid slice">\n'
            + "\n".join(body) + "\n</svg>\n")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.write_text(banner())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
