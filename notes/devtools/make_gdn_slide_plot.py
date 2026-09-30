#!/usr/bin/env python3
"""Render the GDN stage-latency chart for pto_einsum_slide.pptx.

Matches the deck's visual language (see make_slide_plots.py): flat horizontal
bars on the panel's own #EBEBEB ground, no frame, no gridlines, monospace
labels, the value printed at the bar end and the ratio at the right margin.

Values come from devtools/gdn_bench/*.json, so re-running the benchmark and then
this script refreshes the slide.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RED, GREY, RULE = "#C00000", "#929292", "#C5C5C5"
INK, BODY, MUTED = "#1D1D1A", "#3F3F3D", "#666666"
MONO = "DejaVu Sans Mono"
DPI = 300

#: PNG geometry in inches — must match the placement in build_slides.py.
FIG_W, FIG_H = 5.59, 2.24

T = 8192
STAGES = [("chunk_cumsum", "chunk_cumsum"), ("scaled_dot_kkt", "scaled_dot_kkt"),
          ("solve_tril", "solve_tril"), ("wy_fast", "wy_fast"),
          ("chunk_h", "chunk_h"), ("chunk_o", "chunk_o")]


def _load(path, metric="median_us"):
    raw = json.loads(Path(path).read_text())
    return {(r["stage"], r["t"]): r.get(metric) for r in raw["records"] if r.get("ok")}


def main() -> int:
    ours = _load(HERE / "gdn_bench" / "gdn_bench_pypto.json")
    theirs = _load(HERE / "gdn_bench" / "gdn_bench_mega.json")
    rows = [(label, ours[(key, T)], theirs[(key, T)]) for key, label in STAGES]
    xmax = max(v for _l, a, b in rows for v in (a, b)) * 1.24

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("#EBEBEB")
    ax = fig.add_axes([0.235, 0.135, 0.645, 0.775])
    ax.set_facecolor("#EBEBEB")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=2)

    h = 0.34
    ypos = list(range(len(rows)))[::-1]
    for y, (_lab, mine, ref) in zip(ypos, rows):
        ax.barh(y - h / 2, ref, height=h, color=GREY, zorder=2)
        ax.text(ref + xmax * 0.012, y - h / 2, f"{ref:,.0f}", va="center", ha="left",
                fontfamily=MONO, fontsize=5.2, color=MUTED)
        ax.barh(y + h / 2, mine, height=h, color=RED, zorder=2)
        ax.text(mine + xmax * 0.012, y + h / 2, f"{mine:,.0f}", va="center", ha="left",
                fontfamily=MONO, fontsize=5.2, color=RED, fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontfamily=MONO, fontsize=5.4, color=BODY)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="x", labelsize=5.0, colors=MUTED)
    for lab in ax.get_xticklabels():
        lab.set_fontfamily(MONO)
    for y, (_lab, mine, ref) in zip(ypos, rows):
        ax.text(xmax * 0.995, y, f"{ref / mine:.2f}×", va="center", ha="right",
                fontfamily=MONO, fontsize=5.6, fontweight="bold",
                color=RED if ref >= mine else MUTED)
    ax.axvline(0, color=RULE, lw=0.6, zorder=1)

    fig.text(0.012, 0.965, "MICROSECONDS PER CALL, LOWER IS BETTER", fontfamily=MONO,
             fontsize=5.6, fontweight="bold", color=MUTED, va="top")
    fig.text(0.585, 0.965, "■", fontsize=5.4, color=RED, va="top", ha="left")
    fig.text(0.607, 0.965, "generated", fontfamily=MONO, fontsize=5.2, color=MUTED,
             va="top", ha="left")
    fig.text(0.775, 0.965, "■", fontsize=5.4, color=GREY, va="top", ha="left")
    fig.text(0.797, 0.965, "hand-written", fontfamily=MONO, fontsize=5.2, color=MUTED,
             va="top", ha="left")

    out = HERE / "slide_plot_gdn.png"
    fig.savefig(out, facecolor="#EBEBEB")
    total_o = sum(r[1] for r in rows)
    total_t = sum(r[2] for r in rows)
    print(f"wrote {out}  (pipeline {total_o:.0f} vs {total_t:.0f} us, "
          f"{total_t / total_o:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
