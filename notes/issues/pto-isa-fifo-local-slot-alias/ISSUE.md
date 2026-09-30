# pto-isa: cross-core FIFO local slots alias when consecutive tiles differ in size

**Status:** ROOT-CAUSED and FIXED 2026-08-11, one line. NOT yet filed upstream.
**Layer:** pto-isa (`include/pto/npu/a2a3/TPush.hpp`). **Live on current upstream master.**
**Severity:** silent wrong results from `pl.matmul` — no error, no warning, no diagnostic.

## Root cause

`popMatTileFromGMFiFo` strides the consumer's **local (L1) ring** by the popped tile's own
size, while the **GM ring three lines above** strides by the ring's fixed `SLOT_SIZE`:

```cpp
size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;   // GM: fixed  ✔
...
uint64_t localTileBase = fifo.V2C_CONSUMER_BUF
                       + (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);  // L1: payload-dependent  ✘
```

Slots of a ring must be at fixed positions. Striding by the payload means slot `i+1` of a
*smaller* tile lands **inside** slot `i` of a larger one, so two tiles that the FIFO
considers to be in different slots occupy overlapping L1. Consecutive tiles of different
sizes then corrupt each other.

In the a2a3 source the GM line still carries the old formula as a trailing comment
(`// ConsM * ConsN * sizeof(T);`) — the GM ring was migrated to `SLOT_SIZE` and the local
ring was not. a5 has the same payload-dependent local stride, so this is not a2a3-specific.

## Fix

```cpp
uint64_t localTileBase = fifo.V2C_CONSUMER_BUF
                       + (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * RingFiFo::SLOT_SIZE;
```

## Why the failure predicate is exactly `N < M`, with `K` irrelevant

For `[M,K] @ [K,N]` the operands cross the pipe as two Mat tiles:

* A occupies `M*K*sizeof(T)` bytes at local offset 0
* B occupies `K*N*sizeof(T)` bytes at local offset `1 * K*N*sizeof(T)`

They overlap iff `K*N < M*K`, i.e. **`N < M`** — `K` cancels. That predicate matches all 19
measured fp32 shapes, including the controls chosen to break it (`M=32,K=64,N=32` has
`N == M` but `N < K` and passes; `M=128,K=64,N=64` has `N=64` and fails because `M=128`).

## Evidence (a2a3 hardware, card 0)

| | before | after (one-line fix) |
|---|---|---|
| bare `pl.matmul` probe, 18 configs | 8/18 | **18/18** |
| GLA shape matrix, 7 configs | 3/7 | **7/7** |
| original F3.1c case `C=64, dv=32` | max diff **72** | **2.67e-05** |

**No regressions.** Full a2a3 ST pipe suite against the patched header — `tpushpop_vc` (13),
`tpushpop_cv` (10), `tpushpop_vc_nosplit` (6), `tpushpop_cv_nosplit` (3), `tpushpop_fixpipe`
(4), `tpushpop_dir_both` (2), `tpushpop_dir_both_concurrent` (2), `tpushpop_subtile` (1),
`tmatmul` (16), plus the new case (6) — **10/10 testcases, all green.**

## Why it survived

Not because `N < M` was untested — **four existing `tmatmul` cases already run it**
(`TMATMULBIASTest` 101x288x67, 55x127x29, 150x89x50, 135x64x88), and they pass, because the
cube `TLOAD`s its own operands and never touches a ring. The untested thing is narrower: no
existing case pushes tiles of **different sizes** through one ring and holds both at once.
`st/tpushpop_mixed_tile_size` adds exactly that.

> Corrected 2026-08-12. The first draft of this write-up claimed every existing `tmatmul`
> shape had `N >= M`; that was checked against the eight non-bias cases and assumed of the
> bias ones. It is wrong, and it was the stated justification for shipping a cube-only
> companion testcase — which has accordingly been dropped from the MR. The root cause,
> predicate and fix are unaffected: they never depended on it.

## What this is NOT

* **Not PTOAS.** With disjoint slots there is no hazard between `TPOP` into slot `i+1` and
  `TMOV` out of slot `i`, so the absence of an `MTE1 -> MTE2` anti-dependency in generated
  code is correct, not a codegen omission.
* **Not `TMATMUL` and not `TSTORE`.** A cube-only kernel doing the same tall matmul and
  store is correct to 3.3e-06.
* **Not the merged `DIR_BOTH` ring-offset fix** (`69a81f3b`): reproduces identically on a
  stock tree without it, digit for digit.
* **Not numerical conditioning.** fp32-vs-fp64 divergence is ~2e-5 at every shape, failing
  and passing alike.

---

# Investigation history

The elimination sequence below is kept because it records what was ruled out and how; the
sections after this point predate the root cause above and some of their conclusions were
superseded.

## Independent of our local pto-isa patch (A/B'd against stock)

We carry one local pto-isa change (the `DIR_BOTH` V2C ring offset, upstream MR !1438), so
every number above was measured on a patched tree. Re-running the whole probe against a
**pristine** checkout (`PTO_ISA_ROOT` pointed at a fresh clone with `TPush.hpp` reverted;
`V2C_ENTRY_OFFSET` occurrences: 0 vs 3) reproduces it **identically**:

| M | K | N | patched | stock |
|---|---|---|---|---|
| 64 | 64 | 64 | 3.43e-07 | 3.43e-07 |
| 64 | 64 | 32 | 1.23 | 1.23 |
| 64 | 64 | 16 | 9.05e-01 | 9.05e-01 |
| 64 | 32 | 32 | 1.14 | 1.14 |
| 32 | 32 | 32 | 2.04e-07 | 2.04e-07 |
| 32 | 32 | 16 | 8.84e-01 | 8.84e-01 |
| 32 | 32 | 64 | 1.71e-07 | 1.71e-07 |
| 64 | 64 | 128 | 3.68e-07 | 3.68e-07 |
| 128 | 64 | 64 | 1.18 | 1.18 |

8/18 both ways, same shapes, same values. Structurally this was expected — the generated ISA
uses `Direction::DIR_V2C` and our patch body is inside `if constexpr (is_both)`, so it is not
instantiated — but the measurement is what makes the report safe to file.

Shim: `devtools/stock_isa_run.sh`, chained after `tq_env.sh`.

## ISA-level reproducer: it is the CROSS-CORE TRANSPORT, not the matmul and not the store

Two PTO-ISA ST testcases (`st/` here; drop into `tests/npu/a2a3/src/st/testcase/` and add to
`ALL_TESTCASES`). Same maths, same tile types, same `TSTORE`, same shapes — the only
difference is how the operands reach the cube:

| testcase | operands reach the cube via | tall result |
|---|---|---|
| cube-only variant (`pto_cube_st`) | cube `TLOAD` from GM | **correct** (3.3e-06) |
| `tpushpop_mixed_tile_size` (`pto_mix_st`) | vector -> cross-core pipe -> cube | **wrong** |

(The cube-only variant did its job as a discriminator here and was then dropped from the MR:
the existing `TMATMULBIASTest` cases already cover cube-only `N < M`, so shipping it would
have duplicated coverage — and its host file — for no gain.)

and within the mixed case, both pipe directions fail identically:

| case | pipe | shape | result |
|---|---|---|---|
| case1 | `DIR_V2C` | 64x64x32 | **FAIL** max diff 9.25, 2047/2048 elements wrong |
| case2 | `DIR_V2C` | 64x64x64 | ok 2.86e-06 |
| case3 | `DIR_V2C` | 32x32x16 | **FAIL** max diff 7.47, 512/512 wrong |
| case4 | `DIR_BOTH` | 64x64x32 | **FAIL** max diff 9.94, 2048/2048 wrong |
| case5 | `DIR_BOTH` | 64x64x64 | ok 2.86e-06 |
| case6 | `DIR_BOTH` | 32x32x16 | **FAIL** max diff 5.63, 512/512 wrong |

So `TMATMUL` and `TSTORE` are exonerated by the cube-only case, and the defect is **not**
direction-specific — it is in `TPUSH`/`TPOP` of a tile through a cross-core ring.

**The pattern that fits every result:** the failing operand's tile is *smaller than the ring
slot*. `SLOT_SIZE` is set by the larger operand (`A`, `[64,64]` = 16384 B), so `B` at
`[64,32]` = 8192 B occupies half a slot. In the square control every tile fills its slot
exactly, which would hide any slot-vs-tile stride confusion. The cube-only case has no slot
at all. That is consistent with all six results plus the two cube-only ones.

This also settles that the defect is unrelated to the merged `DIR_BOTH` ring-offset fix
(pto-isa `69a81f3b`): case4/case6 fail *with* that fix applied, and they fail for a different
reason (an under-filled tile, not two rings aliasing).

### Unresolved

`gla_stage2` pushes `la` `[64,32]` — equally under-filled — through a `DIR_BOTH` pipe with the
same 16384 B slot, and `C=64, dk=32, dv=64` is correct end to end. That contradicts the rule
above, and the remaining structural difference is that PyPTO declares the popped `Mat` tile
with **dynamic** valid dims (`-1, -1`, bound at runtime) where the ST case uses static ones.
Worth testing before quoting a mechanism.

## Reproducer

`repro.py` (~40 lines of DSL, no distribution, no collectives, no loop). Run:

```bash
python3 repro.py <device_id> a2a3       # hardware: 8/18 pass
python3 repro.py 0 a2a3sim              # simulator: all pass
```

It builds one `@pl.program` per shape with a single InCore `matmul` and compares against
torch. Note the program needs an `Orchestration` wrapper around the InCore function — a bare
InCore program compiles but emits no kernels, and the runtime then fails with a
`FileNotFoundError` about a missing `kernel_config.py` rather than anything about codegen.

## How it was found

The ZeCO/GLA fused forward (`gla/implementations/pypto/`) was wrong on hardware only, at
some shapes and not others. Sweeping `dk` and `dv` **independently** — which no earlier run
had done — showed the trigger was `dv < C`, with `dk` irrelevant. Every `[C, dv]` tile in
that kernel's output path is taller than wide in exactly the failing configs, which led to
the matmul test above. See ROADMAP F3.1c.

Hypotheses eliminated before getting here, each with data:

* **Numerical conditioning** — the chunk math divides by a cumulative decay, so large `C`
  was a plausible fp32 story. Evaluating the identical math in fp32 vs fp64 gives ~2e-5
  divergence at *every* shape, failing and passing alike (`devtools/f31c_numerics.py`).
* **Tile sharing / allocator aliasing** — the `gamma` chain is allocated fully in place
  (source and destination at the same address) in the **passing** shapes too.
* **Cross-core pipe geometry** — `slot_size`, slot count and reservations are identical
  across failing and passing shapes, with no tile exceeding a slot.
* **Loop-carried state** — `N = 1` (a single chunk, no carry at all) fails just as hard.

## The rule is exactly `N < M` (fp32), confirmed on 10 further shapes

| M | K | N | result | |
|---|---|---|---|---|
| 16 | 16 | 16 | 6.64e-08 | square |
| 32 | 32 | 32 | 2.04e-07 | square |
| 128 | 128 | 128 | 5.80e-07 | square |
| 32 | 64 | 32 | 4.75e-07 | **N == M but N < K** |
| 64 | 128 | 64 | 3.52e-07 | **N == M, N < K** |
| 16 | 32 | 32 | 1.93e-07 | wide |
| 32 | 32 | 16 | **8.84e-01** | tall |
| 128 | 128 | 64 | **1.12** | tall |
| 128 | 128 | 32 | **8.59e-01** | tall |
| 32 | 16 | 16 | **1.20** | tall |

Every prediction of the `N < M` rule held, including the two `N == M, N < K` controls chosen
to break it. `K` genuinely does not participate.

## fp16 is affected too, but with a different boundary

| M | K | N | fp16 | fp32 (same shape) |
|---|---|---|---|---|
| 64 | 64 | 64 | 3.43e-07 | ok |
| 64 | 64 | 32 | **1.23** | wrong |
| 32 | 32 | 16 | 2.98e-07 | **wrong** |

So it is **not** fp32-only — but `32x32x16` is correct in fp16 and wrong in fp32, so the
threshold is dtype-dependent and NOT simply `N < M` for fp16. fp16 doubles the elements per
C0 (16 vs 8 for fp32), which is the obvious suspect, but three data points do not pin it
down and this write-up does not claim a fp16 rule.

## The destination tensor's width is not the trigger

Storing the same `[64, 32]` result into a `[64, 64]` destination instead of a `[64, 32]` one
— changing only the partition view and store stride — gives the identical wrong answer
(1.23 both ways). So this is not the destination stride.

## Not pypto, and not PTOAS

Comparing the generated code for `64x64x32` against `64x64x64`:

* the **PTO IR** is correct — `pto.tmatmul` with `rows=64, cols=32`, matching
  `pto.alloc_tile ... valid_col = 32` and a `pto.tstore` into a `partition_tensor_view<64x32xf32>`,
  strides `[32, 1]`;
* the **PTOAS-generated ISA** is structurally identical to the square case, differing only
  in the tile dimensions, which are right:
  `Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, ..., 1024, ...>` then a plain
  `TMATMUL(acc, left, right)` and `TSTORE(view, acc)`.

Both layers emit what they should, which points at **pto-isa's `TMATMUL`/`TSTORE` handling of
an accumulator tile with more rows than columns**, or an undocumented hardware constraint
that neither layer enforces.

## Padding N up to M is a working fix; transposing is not

The same product, three ways (`devtools/f31c_matmul_localise.py`):

| M | K | N | A: store `[M,N]` | B: pad N->M, store `[M,M]` | C: transpose, store `[N,M]` |
|---|---|---|---|---|---|
| 64 | 64 | 32 | **1.23** | 2.59e-07 | **1.23** |
| 32 | 32 | 16 | **8.84e-01** | 1.99e-07 | **8.84e-01** |
| 64 | 64 | 16 | **9.05e-01** | 2.30e-07 | **9.05e-01** |
| 64 | 64 | 64 | 3.43e-07 | 3.43e-07 | 3.43e-07 |

**B is a usable workaround**: zero-padding the B operand to `[K, M]` so the product is
square gives the right answer at ~2e-07, on every affected shape.

## Unresolved: the same Acc tile is fine when TPUSHed and wrong when TSTOREd

One observation does not fit "a tall matmul computes garbage", and it should be reconciled
before anyone acts on a mechanism.

In `gla_stage2` at `C=64, dk=32`, `b = tril[64,64] @ la[64,32]` is a tall matmul with an Acc
tile that the generated ISA declares **identically** to the failing probe:

```
Tile<Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, ...>
```

The only difference is what happens next — the GLA pushes it to the vector core
(`TPUSH<TPipe<0, DIR_BOTH, ...>>`), while the probe stores it (`TSTORE`). And the GLA config
`C=64, dk=32, dv=64` is **correct** end to end on hardware; since `b` feeds `qt = q*b` and
`kb = k/b`, a corrupt `b` could not produce a correct output. So that tall matmul's result
is good.

That means the defect is **not** in `TMATMUL` itself, and the current best reading is that
it is in reading a tall accumulator tile *out* — `TSTORE` of a tall tile is wrong, `TPUSH`
of the same tile is fine. Variant C complicates even that: it never stores a tall tile (it
transposes to `[N,M]` first) and is still wrong, so either the transpose of a
matmul-derived tall tile is also affected, or storing a transposed tile is. Note the GLA
*does* transpose a tall vector-native tile successfully (`kbt = transpose(kb)`, `kb` is
`[C,dk]` tall), so the two cases differ in provenance, not shape.

Next measurement to settle it: a variant that keeps the tall Acc tile entirely on-chip, uses
it as a matmul operand, and stores only a square result — mirroring what the GLA does — and
compare against the same shape stored directly.

## Why this matters beyond us

Any pypto program with a tall matmul is affected, silently. It is also why the GLA operator
cannot use a head dim smaller than its chunk size — a completely ordinary configuration.
