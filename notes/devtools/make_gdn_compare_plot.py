"""Grouped bar chart: PyPTO's GDN stages against megagdn-pto's.

Numbers are the measured medians already in devtools/gdn_bench/gdn_bench.md
(T = 8192, H = 16, D = 128, chunk = 128) plus the solve_tril improvement from
the doubling-block change, which was measured against that same 512.0 baseline.
Nothing here re-runs.

The total sits in its own panel: at ~1.5 ms it is three times the largest stage,
and on a shared axis it would flatten chunk_cumsum (17 us) to nothing. Two
panels, one measure, one unit -- never two y-scales on one plot.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e2e1dd"
PYPTO, MEGA, MEGA_IMP = "#2a78d6", "#eb6834", "#1baf7a"

STAGES = ["chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast", "chunk_h", "chunk_o"]
PY   = [17.0, 112.8, 466.3, 129.8, 416.6, 320.9]
MG   = [17.7, 215.2, 512.0, 123.4, 420.2, 283.3]
MGI  = [None, None,  418.0, None,  None,  None]
TOT_PY, TOT_MG, TOT_MGI = sum(PY), sum(MG), sum(MG) - 512.0 + 418.0

fig = plt.figure(figsize=(11.5, 4.6), dpi=200, facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[6.4, 1.35], wspace=0.16,
                      left=0.055, right=0.985, top=0.80, bottom=0.13)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

W, x = 0.26, np.arange(len(STAGES))


def style(ax, ymax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9, colors=INK_2)
    ax.set_ylim(0, ymax)


def label(ax, bars, size=8):
    for b in bars:
        h = b.get_height()
        if h:
            ax.text(b.get_x() + b.get_width() / 2, h + ax.get_ylim()[1] * 0.018,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=size, color=INK_2)


style(axL, 600)
# a stage with two bars centres them on its tick; only solve_tril has a third
off = [(-W, 0.0, None) if m is None else (-W * 1.5, -W * 0.5, W * 0.5)
       for m in MGI]
off = [(-W / 2, W / 2, None) if m is None else (-W, 0.0, W) for m in MGI]
b1 = axL.bar([xi + o[0] for xi, o in zip(x, off)], PY, W, color=PYPTO,
             label="PyPTO (ours)", zorder=3)
b2 = axL.bar([xi + o[1] for xi, o in zip(x, off)], MG, W, color=MEGA,
             label="megagdn-pto, as shipped", zorder=3)
b3 = axL.bar([xi + (o[2] if o[2] is not None else 0) for xi, o in zip(x, off)],
             [v if v else 0 for v in MGI], W, color=MEGA_IMP,
             label="megagdn-pto, with our solve_tril change", zorder=3)
label(axL, b1); label(axL, b2); label(axL, b3)
axL.set_xticks(x)
axL.set_xticklabels(STAGES, fontsize=9.5, color=INK)
axL.set_ylabel("latency per call (µs)", fontsize=9.5, color=INK_2)

style(axR, 1750)
xr = np.array([0.0])
t1 = axR.bar(xr - W, [TOT_PY], W, color=PYPTO, zorder=3)
t2 = axR.bar(xr, [TOT_MG], W, color=MEGA, zorder=3)
t3 = axR.bar(xr + W, [TOT_MGI], W, color=MEGA_IMP, zorder=3)
label(axR, t1); label(axR, t2); label(axR, t3)
axR.set_xticks(xr)
axR.set_xticklabels(["all stages"], fontsize=9.5, color=INK)
axR.set_xlim(-0.55, 0.55)
axR.set_ylabel("sum of all stages (µs)", fontsize=9.5, color=INK_2)

fig.text(0.055, 0.935, "Gated DeltaNet: PyPTO against the hand-written PTO-ISA kernels",
         fontsize=13, color=INK, weight="bold")
fig.text(0.055, 0.878,
         "Ascend 910B4  ·  T = 8192, H = 16, D = 128, chunk = 128  ·  median per-call latency, lower is better",
         fontsize=9, color=INK_2)
axL.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncols=3,
           loc="upper left", bbox_to_anchor=(0.0, 1.10), handlelength=1.1)

out = "devtools/gdn_bench/gdn_pypto_vs_megagdn.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
print(f"  totals: pypto {TOT_PY:.1f}  megagdn {TOT_MG:.1f}  megagdn+ours {TOT_MGI:.1f}")
