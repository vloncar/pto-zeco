# What actually stops `C=128` — and why narrower operands cannot fix it

**Measured 2026-08-24, starting A3.** The ROADMAP's premise for A3 was wrong, and the
correction changes which stage should be built next. Everything below is measured, not argued.

## The roadmap said

> The 64 KB `Left`/`Right` operand buffers are the wall for `C=128`: a `[128,128]` fp32 tile is
> exactly 65536 B, the whole buffer, so there is no room to double-buffer. This is **not** a
> working-set problem and no amount of tiling addresses it.

That came from reading the failure message at **one** blocking setting (`head_blocks=1`). The
plan search reports only the FIRST buffer to overflow, so it says "Left" and stops.

## What the whole space actually says

`devtools/a3_c128_budget.py` walks every blocking at `C=128, dk=dv=128` and reports the
compiler's own overflow number for each. Three near-misses, and the closest is **not** an
operand buffer:

| setting | overflows | over by |
|---|---|---|
| head=4, value=2, ring_depth=1 | **Vec** | **256 B** |
| head=8, value=1, ring_depth=1 | Left | 4096 B |
| head=1, value=8, ring_depth=1 | Right | 4096 B |

`C=128` misses by **256 bytes of vector buffer**. It is a working-set problem after all, and a
very near one.

## Why the vector buffer is short: the ring reserve, not the tiles

The failure message is specific:

> Vec buffer usage (188672 bytes) exceeds platform limit (188416 bytes). **The first 65536
> bytes of that space are reserved by system.reserve_buffer** ... this is the cross-core pipe
> ring.

At ring depth 1 the reserve is exactly one copy of the **biggest tile crossing the cube/vector
boundary**, so working tiles get 122880 B and need 123136 B. Confirmed on two independent
shapes:

* `C=128, dv=128, value=2`: crossing candidates are `[C,C]`=65536 and `[C,BV]`=32768 →
  reserve 65536. **The `[C,C]` score matrix sets it.**
* `C=64, dk=dv=256, value=1`: `[C,C]`=16384, `[C,BV]`=65536 → reserve 65536. `[C,BV]` sets it.

So: **reserve = (largest crossing tile) x depth**, and at `C=128` the largest is the `[C,C]`
score matmul result.

## Why narrowing cannot shrink it

```
Error: pl.matmul: out_dtype=fp16 is not supported for Tile operands
       — the Cube accumulator fixes the result dtype, deduced as fp32 here.
```

A matmul **result** is always fp32. The tile that sets the reserve at `C=128` is a matmul
result, so no dtype choice can halve it. Narrower operands address `Left`/`Right` — the two
4096 B near-misses — and operand traffic, but **not** the constraint that actually binds.

Casting `tril` (exactly 0.0/1.0, so lossless) does not help either: it is one crossing tile of
`[C,C]`, but the score result is another of the same size, and the reserve takes the max.

## And GLA cannot narrow most operands anyway

`devtools/a3_precision_study.py`, fp64 golden vs each matmul's operands rounded:

| rounded | max abs err | relative | note |
|---|---|---|---|
| nothing (fp32) | 9.8e-05 | 9.0e-07 | reference |
| `tril` | 9.8e-05 | 9.0e-07 | **exactly free** — 0.0/1.0 only |
| decay matmul (`tril`+`la`) | 3.5e-02 | 3.2e-04 | |
| output matmul (`scores`+`v`) | 4.5e-02 | 4.1e-04 | |
| score matmul (`qt`+`kbt`) | **NaN** | | `kb = k/b` reaches **8.8e+07**, fp16 max is 65504 |
| stage1 update (`kbt`+`v`) | **NaN** | | same |

GLA divides by the within-chunk cumulative decay, and that decay is an exponential of a running
sum. `k/b` overflows fp16 and `q*b` underflows it. **These are not flash-attention's operands
and they do not get flash-attention's answer.** Device-measured fp16 matmul error is 3.03e-04
relative, bf16 2.41e-03 (`devtools/a3_capability_probe.py`) — consistent with the torch study.

Note also that mixed precision is refused outright: *"tile.matmul requires identical lhs and
rhs data types"*, so `tril` cannot go narrow while `la` stays fp32.

## What DOES fix it, exactly

Both `[C,C]` matmuls split over their **contraction** axis, with no accuracy cost:

    b      = tril @ la        =  sum over key-row blocks r of  tril[:, r] @ la[r, :]
    o_intra = scores @ v      =  sum over key-row blocks r of  scores[:, r] @ v[r, :]

Verified exact in fp64 (residual 2.8e-14 at every block size). The widest tile the cube ever
sees becomes `[C, BC]`, so the ring reserve shrinks with it.

**This also corrects the roadmap's description of A4.** A4 was written as flash-attention row
blocking, whose hard part was that "the within-chunk decay is a cumulative product down rows,
so it becomes a sequential scan carrying a `[1,BK]` running total". That applies to blocking the
**output rows**. Blocking the **contraction** is exact and needs no scan at all — `tril[:, r]`
already carries the right zeros. It is a mechanical change, not a rewrite.

## Also learned

* `pl.tile.matmul_acc` **works in fp32 on a2a3**, including as a `pl.range` loop carry
  (2.5e-07 relative). It cannot be seeded from a loaded zero tile — the accumulator must start
  as a real matmul result — so using it needs a peeled first iteration.
  It would move the score and output accumulators out of the vector buffer entirely.
* A same-dtype `pl.tile.cast` is rejected ("target_type fp32 equals input dtype"), so it cannot
  be used as a no-op control.
* A `@pl.program` class defined on stdin cannot be compiled at all ("Cannot retrieve source
  code") — the parser reads the source file. Probes must be real files.

## Recommendation

Do the contraction blocking (exact, delivers `C=128`) rather than narrowing (lossy, and does not
deliver it). Keep narrowing available as a later, opt-in performance lever for `Left`/`Right`
pressure and operand traffic, where its 3e-04 relative cost is a deliberate trade.
