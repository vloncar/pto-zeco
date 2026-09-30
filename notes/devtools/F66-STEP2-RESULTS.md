# F6.6 steps 1-2 — stopwatch parity, and where a call's time actually goes

Measured 2026-08-19 on cards 0-3 (comm canary green on all six pairs), simpler `3165cc89`,
pypto main + ptoas 0.57, pto-isa pin `83d01313` + the two carried fixes.

* **Step 1** — pypto's `measure()` now times a whole `forward()` call, matching
  `measure_backward()` and matching simpler. Committed.
* **Step 2** — `devtools/host_device_split.py`: wraps both backends' methods with
  accumulating timers (no backend code changed) and splits one steady-state call into
  buckets. Runner `devtools/split_run.sh`. Raw: `split_{simpler,pypto}_{forward,backward}.json`.
* Cross-check — `devtools/host_glue_micro.py`: the same host arithmetic timed on its own,
  CPU only, so the residual can be sanity-checked against a direct measurement.

Three samples per config (the point is the *split*, not a headline latency), one warm call
first. pypto totals below are therefore noisy and **must not be read as revised B5.4
numbers** — see "Two measurement defects" for why pypto means need many more samples.

## pypto — already effectively all on-device

| direction | config | total ms | chip | staging | host arith | off-chip % |
|---|---|---|---|---|---|---|
| fwd | P=2 L=128 D=32 | 89.04 | 88.88 | 0.098 | 0.052 | 0.18 |
| fwd | P=4 L=128 D=32 | 224.99 | 224.73 | 0.171 | 0.107 | 0.11 |
| fwd | P=2 L=256 D=32 | 99.58 | 99.39 | 0.114 | 0.057 | 0.19 |
| fwd | P=4 L=256 D=32 | 193.41 | 190.21 | 3.006 | 2.596 | 1.65 |
| fwd | P=2 L=128 D=64 | 50.96 | 50.75 | 0.126 | 0.064 | 0.41 |
| fwd | P=4 L=128 D=64 | 195.41 | 191.41 | 3.756 | 3.533 | 2.04 |
| bwd | P=2 L=128 D=32 | 72.65 | 72.31 | 0.158 | 0.101 | 0.47 |
| bwd | P=4 L=128 D=32 | 147.42 | 146.97 | 0.177 | 0.094 | 0.30 |
| bwd | P=2 L=256 D=32 | 71.33 | 70.86 | 0.159 | 0.081 | 0.65 |
| bwd | P=4 L=256 D=32 | 216.23 | 213.48 | 2.205 | 1.501 | 1.27 |

pypto's only off-device arithmetic is the per-rank total decay (`A.prod`). Normally
0.05-0.1 ms, but it **spiked to 2.6-3.5 ms on three of the four P=4 rows** — one slow
iteration in three is enough to do that. Small, but not zero, and it is on the list for a
fully-on-device implementation.

## simpler — the leftover arithmetic is real but is not what dominates

`chip` is a compute dispatch end to end (shared-memory staging + submit + wait + kernel),
minus the lazy worker `open()` nested inside it. `host` is the residual: total minus every
other bucket, so it counts the leftover torch arithmetic **plus** its tensor allocations and
Python overhead.

| direction | config | total s | boundary s | open s | release s | chip ms | host ms | boundary % | host/(host+chip) |
|---|---|---|---|---|---|---|---|---|---|
| fwd | P=2 L=128 D=32 | 29.36 | 26.63 | 2.13 | 0.40 | 198.4 | 7.63 | 90.7% | 3.7% |
| fwd | P=4 L=128 D=32 | 32.18 | 26.68 | 4.27 | 0.82 | 396.7 | 16.20 | 82.9% | 3.9% |
| fwd | P=2 L=256 D=32 | 29.51 | 26.76 | 2.13 | 0.40 | 200.5 | 9.21 | 90.7% | 4.4% |
| fwd | P=4 L=256 D=32 | 32.28 | 26.88 | 4.18 | 0.81 | 393.2 | 19.07 | 83.2% | 4.6% |
| fwd | P=2 L=128 D=64 | 29.59 | 26.81 | 2.16 | 0.39 | 232.0 | 10.97 | 90.6% | 4.5% |
| fwd | P=4 L=128 D=64 | 32.41 | 27.06 | 4.17 | 0.79 | 382.4 | 19.52 | 83.5% | 4.9% |
| bwd | P=2 L=128 D=32 | 58.70 | 53.18 | 4.29 | 0.79 | 395.6 | 46.37 | 90.6% | 10.5% |
| bwd | P=4 L=128 D=32 | 65.15 | 54.22 | 8.53 | 1.55 | 769.8 | 84.54 | 83.2% | 9.9% |
| bwd | P=2 L=256 D=32 | 58.68 | 53.24 | 4.25 | 0.76 | 383.5 | 42.37 | 90.7% | 9.9% |

P=4 L=256 backward FAILED with a 507018 AICPU drain timeout on card 0 (a device wedge, not
a code defect); the other three backward configs are enough to establish the split.

The named, portable pieces inside `host`: `_S_total` 0.52-1.40 ms, `_shift_snaps`
0.55-2.10 ms, `_gammas` 0.32-0.71 ms (the last is inside `boundary`, and is host work in
**both** backends — not a parity gap).

## Host arithmetic timed on its own (CPU, 20 reps, no cards)

Per call, summed over ranks:

| config | forward glue | backward glue |
|---|---|---|
| P=2 L=128 D=32 | 0.52 | 3.55 |
| P=4 L=128 D=32 | 0.99 | 7.08 |
| P=2 L=256 D=32 | 0.52 | 6.37 |
| P=4 L=256 D=32 | 1.05 | 12.73 |
| P=2 L=128 D=64 | 0.62 | 4.22 |
| P=4 L=128 D=64 | 1.23 | 8.50 |

The backward is 7-12x the forward, and one piece dominates: the per-chunk `gate_h`
correction loop is **53-58% on its own**, and the three per-chunk Python loops
(`gate_h`, `reverse_recurrence`, `dcp`) are **78-87%** together. Those are the expensive
pieces to move and the hard ones.

These are 3-6x lower than the on-device `host` residual (e.g. 0.52 vs 7.63 ms at
P=2 L=128). Two reasons, both expected: the residual includes tensor allocation and Python
overhead beyond the named arithmetic, and the CPU on the measurement box is contended by
the forked chip children during a real run.

## Four findings, three of which change the plan

**1. A round trip to the chip costs ~33 ms (forward) / ~39 ms (backward), regardless of
how much work it carries.** Dividing `chip` by the dispatch count (3 kernels x P ranks
forward, 5 x P backward) gives 33.1 / 33.1 / 33.4 / 32.8 / 38.7 / 31.9 ms forward and
39.6 / 38.5 / 38.4 ms backward — flat while `L` doubles and `D` doubles. The dispatch is
round-trip-bound, not compute-bound.

> **Consequence: a port that ADDS a dispatch is a net loss.** A standalone `shift_snaps`
> kernel would cost ~33 ms x P to remove 0.5-2.1 ms of arithmetic. Everything moved
> on-device must fold into a dispatch that already happens. This kills the "separate
> `shift_snaps` orchestration" option from the original plan.

**2. The boundary collective is 26.6-27.1 s per phase and does not depend on P, L or D.**
It is 83-91% of every call in both directions. The backward's is 53.2-54.2 s = exactly
twice the forward's, which *proves* (rather than infers) B5.4's reading that the
backward/forward ratio of ~2 is a boundary-build counter. The forward builds one AllScan
worker per call; the backward builds two.

**3. Work placement is a 4-5% effect in the forward and ~10% in the backward** — measured
as host / (host + chip). Real, and enough to matter for a kernel-level claim, but nowhere
near enough to explain any cross-backend gap. Against the whole call it is 0.03-0.13%.

**4. The compute-worker stand-up after a boundary release is 2.1 s per call at P=2 and
4.2 s at P=4 (double that in the backward)** — 8-16% of the call. This is the cost of the
one surviving runtime constraint: a device hosts one worker at a time, so the held compute
workers must be dropped for the boundary and stood back up afterwards.

## Two measurement defects found

**Stopwatch asymmetry (fixed).** pypto's `measure()` staged inputs *outside* the timing
loop and timed `_dispatch()` alone, while `measure_backward()` timed a full `backward()`
with staging inside. Same backend, two rules. Fixed in step 1; the correction itself is
small (0.1-3 ms) but the rule now matches simpler in both directions.

**pypto's P=2 latency is bimodal, and B5.4 reported means.** From B5.4's own archived
samples:

| B5.4 pypto row | mean | median | min | p95 |
|---|---|---|---|---|
| fwd P=2 L=128 D=32 | 45.89 | 21.15 | 19.33 | 122.06 |
| fwd P=2 L=256 D=32 | 38.83 | 19.29 | 17.75 | 120.06 |
| fwd P=2 L=128 D=64 | 47.94 | 18.14 | 16.50 | 119.62 |
| bwd P=2 L=128 D=32 | 115.42 | 125.49 | 23.28 | 126.81 |
| fwd P=4 L=128 D=32 | 226.38 | 225.52 | 224.31 | 230.71 |

Most P=2 forward calls take ~20 ms with occasional ~120 ms ones dragging the mean to 46.
P=4 is tight (226 vs 225). So **B5.4's P=2 pypto means are not representative**, and the
"backward is 2.5x the forward at P=2" figure is built on them — by medians it would be
5.9x. This also explains why this run's 3-sample pypto totals do not match B5.4's.
Independent of the stopwatch fix, and it needs its own re-run with medians and more
samples before any pypto ratio is quoted again.

## Revised plan

Unchanged in direction — the goal is a fully on-device implementation of both paths — but
the sequencing and the mechanism change:

1. **`_S_total` first, and it is free.** `chunk_h_orch.cpp` allocates the carried state `S`
   as a scope temporary; after the last chunk `S` **is** `S_total`, and the host then
   recomputes it. Point the carried state at a slot of a widened `[N+1,dk,dv]` `s_snap`
   output instead. No new kernel, **no new dispatch**, one changed allocation.
2. **`_shift_snaps` and `log(A)` fold into existing dispatches** — `chunk_o`/`grad_o`
   prologue and `gate_cumsum`'s prep respectively. Not standalone kernels (finding 1). The
   per-chunk gate product `_shift_snaps` rebuilds from `A` is already the last row of each
   chunk's `g_cs`, so recomputing it is pure waste.
3. **The backward's three per-chunk loops are 78-87% of the leftover** and are the real
   work. The reverse recurrence is structurally the same scan `chunk_h` already runs.
   Merge the duplicated copy in `forward_multihead`/`backward_multihead` first, so the
   port lands once.
4. **Honest labelling applies now, not at the end.** With the numbers above, F6.6's
   original warning can be replaced by a quantity: work placement is a 4-5% (fwd) /
   ~10% (bwd) effect on simpler's compute, 0.03-0.13% of its call.

**And a note on expectations.** Completing all of this removes 8-20 ms (fwd) / 42-85 ms
(bwd) from a 29-65 **second** call. It is worth doing because a fully on-device
implementation is the goal and because it makes the comparison legitimate — not because it
is a speed win. The speed target is finding 2: 26.7 s of boundary worker build+teardown per
phase, which F7.4 already showed can be amortized across heads (17.3 -> 8.85 s/head).
