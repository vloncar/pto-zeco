"""Grouped bar chart: every stage of the Qwen3.8-27B GDN block, three implementations.

PyPTO (this port), the hand-written PTO-ISA reference kernels, and the CANN
operator library through torch_npu. All at the model's own shape -- T = 8192,
H = 48 value heads, Hg = 16 key heads, D = 128, chunk = 128 -- on a2a3.

A bar is present only where that implementation exists and was measured. Three
kinds of absence, all different:
  * the CANN library has no operator for the delta rule on this generation
    (`npu_recurrent_gated_delta_rule` is A5-only), and none for the qk-norm;
  * no hand-written counterpart exists for the projections, the norms or the
    quantiser -- megagdn-pto is the delta rule only;
  * the conv's hand-written counterpart comes from pto-kernels, not megagdn-pto.

PROVENANCE, which matters because there is a ~5% spread BETWEEN task-submit
grants and under 1% within one:
  * delta rule, both series: one grant, 2026-09-16, back to back on one card.
    PyPTO from models/qwen3_8_27b/bench.py (mean of the per-round effective
    window, 50 rounds); megagdn-pto from devtools/gdn_bench_mega.py (median of
    50 x 20-launch batches). Its per-batch MINIMA are a harness artifact -- some
    read near zero -- so medians only.
  * each PyPTO-vs-CANN pair: measured against each other on one card in the
    grant that built that kernel (Q4b-Q4d), quoted from the docs page.
So read within a group freely; across groups, the sums are indicative.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dd"
PYPTO, HAND, CANN = "#2a78d6", "#eb6834", "#7a6cc4"

# stage, PyPTO us, hand-written PTO-ISA us, CANN us
# None = that implementation does not exist / was not measured.
ROWS = [
    ("quant_x",         191.0,  None,   105.0),
    ("in_proj_qkv",    1813.0,  None,  1660.0),
    ("in_proj_z",      1014.0,  None,   993.0),
    ("in_proj_ab",      115.0,  None,    47.0),
    ("short_conv",      648.0, 3069.2, 2597.4),
    ("qk_norm_gate",    122.0,  None,    None),
    ("chunk_cumsum",     22.1,   22.5,   None),
    ("scaled_dot_kkt",  279.6,  418.0,   None),
    ("solve_tril",     1268.5, 1217.7,   None),
    ("wy_fast",         598.3,  641.6,   None),
    ("chunk_h",         835.4,  941.2,   None),
    ("chunk_o",         907.6, 1047.5,   None),
    ("gated_rmsnorm",   643.0,  None,  2584.1),
    ("out_proj",       1048.0,  None,   977.0),
]

DELTA = {"chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast", "chunk_h", "chunk_o"}
# The composed block, measured as one program on hardware (14 operators, one
# launch): the number the port actually delivers.
BLOCK_COMPOSED = 9329.8


def totals():
    d_py = sum(r[1] for r in ROWS if r[0] in DELTA)
    d_hw = sum(r[2] for r in ROWS if r[0] in DELTA)
    # the operators that have a CANN counterpart, so the pair is like for like
    pair = [r for r in ROWS if r[3] is not None]
    return d_py, d_hw, sum(r[1] for r in pair), sum(r[3] for r in pair)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK_2, labelsize=8)


def label(ax, bars, fs=6.4, dy=2.5):
    """Value on top of every bar. `dy` lifts one series so near-equal
    neighbours (chunk_cumsum's 22.1 against 22.5) do not print on top of
    each other."""
    for b in bars:
        h = b.get_height()
        if not h:
            continue
        ax.annotate(f"{h:,.0f}" if h >= 100 else f"{h:.1f}",
                    (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=fs, color=INK_2)


fig = plt.figure(figsize=(15.0, 6.6), dpi=200, facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.5], wspace=0.10,
                      left=0.045, right=0.99, top=0.785, bottom=0.235)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

names = [r[0] for r in ROWS]
py = np.array([r[1] for r in ROWS], dtype=float)
hw = np.array([r[2] if r[2] is not None else 0.0 for r in ROWS])
cn = np.array([r[3] if r[3] is not None else 0.0 for r in ROWS])

W, x = 0.27, np.arange(len(ROWS))
style(axL)
label(axL, axL.bar(x - W, py, W, color=PYPTO, label="PyPTO (this port)"))
label(axL, axL.bar(x, hw, W, color=HAND, label="hand-written PTO-ISA"), dy=11.0)
label(axL, axL.bar(x + W, cn, W, color=CANN, label="CANN library (torch_npu)"))
axL.set_xticks(x)
axL.set_xticklabels(names, rotation=38, ha="right", fontsize=8.4, color=INK)
axL.set_ylabel("latency, us  (lower is better)", fontsize=8.5, color=INK_2)
axL.set_ylim(0, 3450)
axL.set_xlim(-0.72, len(ROWS) - 0.28)

# mark where a missing bar is a missing implementation, not a missing measurement
for i, r in enumerate(ROWS):
    if r[2] is None:
        axL.annotate("\u2013", (i, 8), ha="center", va="bottom", fontsize=9, color="#a8a6a0")
    if r[3] is None:
        axL.annotate("\u2013", (i + W, 8), ha="center", va="bottom", fontsize=9, color="#a8a6a0")

# the delta rule occupies one contiguous run of stages; say so on the plot
lo = min(i for i, r in enumerate(ROWS) if r[0] in DELTA) - 0.5
hi = max(i for i, r in enumerate(ROWS) if r[0] in DELTA) + 0.5
axL.axvspan(lo, hi, color="#f3f2ee", zorder=0)
axL.annotate("the delta rule \u2014 the CANN library has no operator for it on this generation",
             ((lo + hi) / 2, 3320), ha="center", fontsize=8.2, color=INK_2)

d_py, d_hw, p_py, p_cn = totals()
style(axR)
H = W / 2                     # two-bar groups straddle their tick, one-bar groups sit on it
label(axR, axR.bar([0 - H, 1 - H, 2.0], [d_py, p_py, BLOCK_COMPOSED], W, color=PYPTO), fs=7.2)
label(axR, axR.bar([0 + H], [d_hw], W, color=HAND), fs=7.2)
label(axR, axR.bar([1 + H], [p_cn], W, color=CANN), fs=7.2)
axR.set_xticks([0, 1, 2])
axR.set_xticklabels(["delta rule\n(6 stages)",
                     "CANN-comparable\n(7 ops)",
                     "whole block\n(14, one program)"],
                    fontsize=7.6, color=INK)
axR.set_ylim(0, 10400)
axR.set_xlim(-0.55, 2.55)
axR.set_title("summed", fontsize=8.5, color=INK_2, pad=6)

fig.text(0.045, 0.945, "Qwen3.8-27B Gated DeltaNet: every stage, three implementations",
         fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.045, 0.905,
         "a2a3 (910B2), T = 8192, H = 48 value / 16 key heads, D = 128, chunk = 128.   "
         "A bar is absent, and marked \u2013, where that implementation does not exist \u2014 not where it was left unmeasured.",
         fontsize=8.8, color=INK_2)
axL.legend(frameon=False, fontsize=8.8, loc="upper left", bbox_to_anchor=(0.0, 1.105),
           ncols=3, handlelength=1.1, labelcolor=INK)

FOOT = [
    "Delta rule, both series: measured back to back inside ONE task-submit grant on one card, 2026-09-16 \u2014 PyPTO is the mean of 50 per-round effective windows, PTO-ISA the median of 50 batches of 20 launches",
    "(its per-batch minima are a harness artifact and are not used).  Each PyPTO-vs-CANN pair was measured against itself on one card in one grant; between grants the spread is ~5%, so read within a group and treat the sums as indicative.",
    "short_conv's hand-written counterpart is pto-kernels' causal_conv1d, not megagdn-pto, and its CANN bar charges the transpose our token-major layout needs (channels-first, uncharged, it is 1532).",
]
for i, line in enumerate(FOOT):
    fig.text(0.045, 0.075 - i * 0.027, line, fontsize=7.2, color=INK_2)

out = "/root/workspace/allscan/devtools/gdn_bench/gdn_three_way.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)

data = dict(
    shape=dict(platform="a2a3", t=8192, h=48, hg=16, d=128, chunk=128),
    units="us", rows=[dict(stage=n, pypto=p, pto_isa=h, cann=c) for n, p, h, c in ROWS],
    totals=dict(delta_pypto=d_py, delta_pto_isa=d_hw,
                cann_pairs_pypto=p_py, cann_pairs_cann=p_cn,
                block_composed_pypto=BLOCK_COMPOSED),
)
with open("/root/workspace/allscan/devtools/gdn_bench/gdn_three_way.json", "w") as f:
    json.dump(data, f, indent=1)
print("delta:", d_py, d_hw, " cann-pairs:", p_py, p_cn)
