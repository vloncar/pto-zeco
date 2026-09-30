#!/usr/bin/env python3
"""Render the two ZeCO/AllScan result charts as PNGs for pto_einsum_slide.pptx.

Matches the deck's existing visual language exactly (see the PTO-EINSUM half):
flat horizontal bars, no frame, no gridlines, monospace labels, value printed at
the bar end. Palette is lifted from the slide's own shapes:

    C00000  the favoured series (compiler-generated) / accent
    929292  the comparison series (hand-written) / muted
    C5C5C5  rules
    1D1D1A  headings      3F3F3D  body      666666  labels + captions

Consolas is not installed on Linux; DejaVu Sans Mono is the visual stand-in and is
close enough at 6-7pt. Run with the slide venv, not the system python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, GREY, RULE = "#C00000", "#929292", "#C5C5C5"
INK, BODY, MUTED = "#1D1D1A", "#3F3F3D", "#666666"
MONO = "DejaVu Sans Mono"
DPI = 300

#: PNG geometry in inches — must match the placement in add_plots_to_slide.py.
FIG_W, FIG_H = 5.59, 1.58


def _canvas(title, note):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("#EBEBEB")          # the bottom-half panel colour
    ax = fig.add_axes([0.255, 0.215, 0.61, 0.60])
    ax.set_facecolor("#EBEBEB")
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=2)
    fig.text(0.012, 0.945, title, fontfamily=MONO, fontsize=6.2, fontweight="bold",
             color=MUTED, va="top")
    fig.text(0.012, 0.028, note, fontfamily=MONO, fontsize=4.9, color=MUTED, va="bottom")
    return fig, ax


def _draw(ax, rows, xmax, ratios=True):
    """rows: list of (label, generated_value, handwritten_value | None)."""
    h = 0.34
    ypos = list(range(len(rows)))[::-1]
    for y, (_lab, gen, hand) in zip(ypos, rows):
        if hand is not None:
            ax.barh(y - h / 2, hand, height=h, color=GREY, zorder=2)
            ax.text(hand + xmax * 0.012, y - h / 2, f"{hand:,.1f}".rstrip("0").rstrip("."),
                    va="center", ha="left", fontfamily=MONO, fontsize=5.2, color=MUTED)
        if gen is not None:
            ax.barh(y + h / 2, gen, height=h, color=RED, zorder=2)
            ax.text(gen + xmax * 0.012, y + h / 2, f"{gen:,.1f}".rstrip("0").rstrip("."),
                    va="center", ha="left", fontfamily=MONO, fontsize=5.2, color=RED,
                    fontweight="bold")
        else:
            ax.text(xmax * 0.012, y + h / 2, "no result — shape ceiling", va="center",
                    ha="left", fontfamily=MONO, fontsize=4.8, color=MUTED, style="italic")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontfamily=MONO, fontsize=5.2, color=BODY)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="x", labelsize=5.0, colors=MUTED)
    for lab in ax.get_xticklabels():
        lab.set_fontfamily(MONO)
    if ratios:
        for y, (_lab, gen, hand) in zip(ypos, rows):
            if gen and hand:
                ax.text(xmax * 0.995, y, f"{hand / gen:.1f}\u00d7", va="center", ha="right",
                        fontfamily=MONO, fontsize=5.6, fontweight="bold", color=RED)
    ax.axvline(0, color=RULE, lw=0.6, zorder=1)


def _legend(fig, left_label, right_label):
    fig.text(0.585, 0.945, "■", fontsize=5.4, color=RED, va="top", ha="left")
    fig.text(0.607, 0.945, left_label, fontfamily=MONO, fontsize=5.2, color=MUTED,
             va="top", ha="left")
    fig.text(0.815, 0.945, "■", fontsize=5.4, color=GREY, va="top", ha="left")
    fig.text(0.837, 0.945, right_label, fontfamily=MONO, fontsize=5.2, color=MUTED,
             va="top", ha="left")


def plot_single_chip(p1_json: Path, out: Path):
    """Job 1 — one chip, whole operator, end to end."""
    raw = json.load(open(p1_json))
    by = {}
    for r in raw:
        by.setdefault((r.get("direction") or r.get("dir"), r["L"], r["D"]), {})[r["impl"]] = r
    order = [("fwd", 128, 32), ("fwd", 256, 32), ("fwd", 128, 64),
             ("bwd", 128, 32), ("bwd", 256, 32), ("bwd", 128, 64)]
    rows = []
    for k in order:
        v = by.get(k)
        if not v or "simpler" not in v:
            continue
        gen = v["pypto"]["p50_ms"] if "pypto" in v else None
        rows.append((f"{k[0]}  L={k[1]:<4} D={k[2]}", gen, v["simpler"]["p50_ms"]))
    xmax = max(h for _l, _g, h in rows) * 1.42
    fig, ax = _canvas("ONE CHIP · WHOLE OPERATOR, END TO END",
                      "L = tokens per chip, D = head width  ·  milliseconds per call, lower is better  ·  "
                      "median of 10, steady state, all verified correct")
    _legend(fig, "pypto", "simpler")
    _draw(ax, rows, xmax)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} rows)")
    return rows


def plot_collective(fwd_json: Path, bwd_json: Path, out: Path):
    """Job 2 — the hand-off routine on its own, both directions, cost amortized."""
    rows, present = [], []
    for tag, path in (("fwd", fwd_json), ("bwd", bwd_json)):
        if not path.exists():
            continue
        present.append(tag)
        by = {}
        for r in json.load(open(path)):
            by.setdefault((r["P"], r["dk"], r["dv"], r["K"]), {})[r["impl"]] = r
        for k in sorted(by, key=lambda k: (k[0], k[1], k[3])):
            v = by[k]
            if "pypto" not in v or "simpler" not in v:
                continue
            rows.append((f"{tag}  P={k[0]} {k[1]}×{k[2]} K={k[3]}",
                         v["pypto"]["p50_ms"], v["simpler"]["p50_ms"]))
    if not rows:
        print("collective: no comparable rows", file=sys.stderr)
        return []
    xmax = max(max(g or 0, h) for _l, g, h in rows) * 1.42
    fig, ax = _canvas("THE HAND-OFF ROUTINE ALONE · 2 AND 4 CHIPS",
                      "milliseconds per exchange, lower is better  ·  fixed setup amortized identically on both sides")
    _legend(fig, "pypto", "simpler")
    _draw(ax, rows, xmax)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} rows)")
    return rows


def plot_pending(out: Path, title: str, headline: str, note: str):
    """A styled stand-in that holds the second chart's exact footprint."""
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("#EBEBEB")
    ax = fig.add_axes([0.012, 0.10, 0.976, 0.72])
    ax.set_facecolor("#EBEBEB")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linestyle((0, (4, 4)))
        ax.spines[side].set_linewidth(0.7)
    ax.text(0.5, 0.56, headline, ha="center", va="center", fontfamily=MONO,
            fontsize=8.0, fontweight="bold", color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.30, note, ha="center", va="center", fontfamily=MONO,
            fontsize=5.0, color=MUTED, transform=ax.transAxes, wrap=True)
    fig.text(0.012, 0.945, title, fontfamily=MONO, fontsize=6.2, fontweight="bold",
             color=MUTED, va="top")
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}  (pending stand-in)")


if __name__ == "__main__":
    d = Path("/root/workspace/allscan/devtools")
    plot_single_chip(d / "p1_results.json", d / "slide_plot_single_chip.png")
    rows = plot_collective(d / "allscan_fair_forward.json", d / "allscan_fair_backward.json",
                           d / "slide_plot_collective.png")
    if not rows:
        plot_pending(d / "slide_plot_collective.png",
                     "THE HAND-OFF ROUTINE ALONE \u00b7 2 AND 4 CHIPS",
                     "measurement pending",
                     "pypto side measured; simpler does not currently launch on this stack,\n"
                     "so there is no like-for-like pair yet")
