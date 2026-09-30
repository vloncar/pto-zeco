# B5.4 — steady-state forward vs backward, simpler vs pypto

Run 2026-08-18 on cards 0-3 (comm canary green, all six pairs). `--iters 10 --warmup 3`,
every row correctness-verified against the torch golden, every row `SS=Y` (steady state:
one-time setup excluded). Raw data: `b54_results_merged.json`.

simpler rows are new. pypto rows are carried from the 2026-08-17 sweep — its backend is
untouched since (empty diff over `gla/implementations/pypto`) and the harness is identical.

## Forward, mean ms/call

| P | L | D | pypto | simpler |
|---|---|---|---|---|
| 2 | 128 | 32 | 45.89 | 30 922 |
| 4 | 128 | 32 | 226.38 | 34 280 |
| 2 | 256 | 32 | 38.83 | 29 453 |
| 4 | 256 | 32 | 232.26 | 32 318 |
| 2 | 128 | 64 | 47.94 | 29 725 |
| 4 | 128 | 64 | 168.81 | 33 118 |

## Backward, mean ms/call

| P | L | D | pypto | simpler |
|---|---|---|---|---|
| 2 | 128 | 32 | 115.42 | 57 537 |
| 4 | 128 | 32 | 227.91 | 63 764 |
| 2 | 256 | 32 | 92.87 | 60 289 |
| 4 | 256 | 32 | 226.78 | 66 904 |
| 2 | 128 | 64 | — shape ceiling | not run |
| 4 | 128 | 64 | — shape ceiling | not run |

pypto's backward at `D=64` exceeds the 184 KB vector buffer (the B4 backward is roughly
twice the forward's working set). simpler was not run there because there would be no
counterpart to compare against.

## What the numbers actually say

**Do not read the cross-backend ratio as a kernel-quality result.** Two facts rule it out.

**1. simpler's per-call cost does not depend on the workload.** Its forward is
29 453–34 280 ms across all six configs — a 16% spread while `L` doubles, `D` doubles and
`P` doubles. Real compute would scale with those. This is a fixed cost per call.

**2. simpler's backward/forward ratio is flat at ~2, at every P.**

| config | ratio |
|---|---|
| P=2 L=128 D=32 | 1.86x |
| P=4 L=128 D=32 | 1.86x |
| P=2 L=256 D=32 | 2.05x |
| P=4 L=256 D=32 | 2.07x |

The forward builds **one** boundary AllScan worker per call; the backward builds **two**
(forward ring, then reverse ring). The ratio is a build counter. The GLA compute is inside
the noise of that fixed cost — consistent with F6.1–F6.4, which measured simpler's warm
compute kernels at ~6.0 ms, below pypto's entire 12.15 ms forward call.

So the gap here is **architecture, not kernels**: a device hosts one worker at a time, so
comm and compute cannot co-reside, so every call pays a full HCCL distributed-worker build
and teardown. That is the last of the three workarounds in this backend; the other two are
gone (see `../allscan/issues/simpler-l3-callable-redispatch/`).

Note that closing F4.1 — holding one compute worker across dispatches instead of forking a
fresh one per kernel — **did not move these numbers**, and that is itself the finding. It was
never the dominant term at `P>=2`. It was worth doing anyway: it is what makes the backward
measurable in steady state at all (this is the first B5.4 run to produce simpler backward
numbers), and it deleted a code path plus a silent fallback.

**pypto's own ratio is the other half of the story:**

| config | ratio |
|---|---|
| P=2 L=128 D=32 | 2.52x |
| P=2 L=256 D=32 | 2.39x |
| P=4 L=128 D=32 | 1.01x |
| P=4 L=256 D=32 | 0.98x |

At P=2 the backward costs ~2.5x the forward, which is what recompute plus the extra matmuls
should cost. At P=4 it costs **the same as the forward** — pypto's forward has already
become orchestration-bound by four ranks, so the extra backward compute is free. Whatever
dominates pypto at P=4 is not arithmetic either.

## Gate

**F6.6 is still open**, so none of this may be turned into a compute-vs-comm or
kernel-vs-kernel split: simpler runs `_S_total`, `_shift_snaps` and `_gammas` on the host
while pypto does all of it on device. These are end-to-end per-call latencies, which is the
one thing that is comparable without that parity.

---

## Caveat added 2026-08-19 (F6.6 step 1-2): the pypto P=2 means above are not representative

pypto's per-call latency at **P=2 is bimodal**, and the table reports means. From this
file's own archived samples:

| row | mean | median | min | p95 |
|---|---|---|---|---|
| fwd P=2 L=128 D=32 | 45.89 | 21.15 | 19.33 | 122.06 |
| fwd P=2 L=256 D=32 | 38.83 | 19.29 | 17.75 | 120.06 |
| fwd P=2 L=128 D=64 | 47.94 | 18.14 | 16.50 | 119.62 |
| bwd P=2 L=128 D=32 | 115.42 | 125.49 | 23.28 | 126.81 |
| fwd P=4 L=128 D=32 | 226.38 | 225.52 | 224.31 | 230.71 |

Most P=2 forward calls take ~20 ms with occasional ~120 ms ones pulling the mean to 46.
P=4 is tight. **The P=2 pypto rows and every ratio derived from them (including "backward
is 2.52x the forward at P=2" — 5.9x by medians) need a re-run with medians and more
samples.** The P=4 rows and all simpler rows are unaffected. Separately, pypto's
`measure()` has since been fixed to time a whole `forward()` call (it previously excluded
input staging while `measure_backward()` included it), so the pypto forward rows above were
produced under a stopwatch the code no longer uses; that correction is small (0.1-3 ms).

The B5.4 attribution argument itself is **strengthened**, not weakened: the per-phase split
in `F66-STEP2-RESULTS.md` measures simpler's boundary collective at 26.6-27.1 s per phase,
flat in P/L/D, with the backward paying exactly two of them — so "the number is a build
counter" is now a measurement rather than an inference.
