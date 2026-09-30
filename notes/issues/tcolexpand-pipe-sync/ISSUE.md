# pto-isa: `TCOLEXPAND` (and the expand family) is tagged `PIPE_V` but lowers to a copy-engine op — under-synchronized

## Summary
`Op::TCOLEXPAND` is declared on the vector pipe:
```
include/pto/common/event.hpp:269:  PTO_DEFINE_OP_PIPE(Op::TCOLEXPAND, PIPE_V);
```
but its a2a3 device implementation is a **copy-engine** op, not a pure PIPE_V vector op:
```
include/pto/npu/a2a3/TColExpand.hpp:28:  pto_copy_ubuf_to_ubuf(dstPtr + i * dstStride, srcPtr, 1, lenBurst, 0, 0);
```
(a `pto_copy_ubuf_to_ubuf` per output row). Because the op is tagged `PIPE_V`, a
`pipe_barrier(PIPE_V)` after `TCOLEXPAND` is treated as sufficient to order a
following vector read of the destination — but it does **not** wait for the ubuf
copy to complete. A consumer then reads the destination before the broadcast lands.
The whole expand family (`TCOLEXPAND*`, `TROWEXPAND*`) is tagged the same way
(`event.hpp:269–276`).

The window is data-size dependent: at large tiles (C=128) the copy is slow enough
that the consumer happens to wait; at small tiles (C=32) it wins the race and reads
stale data — a wrong result, not a crash.

## Environment
- pto-isa `/opt/pto-isa`, a2a3 (910B2), fp32. (Confirmed present on the current tree,
  2026-07-27.)

## Repro (runtime, via a GLA chunk kernel)
`chunk_h_prep` broadcasts `g_total[1,K]` down C rows with `TCOLEXPAND`, then a `TSUB`
reads the result. Set the post-`TCOLEXPAND` barrier to `pipe_barrier(PIPE_V)` (in
`gla/implementations/simpler/kernels/aiv/chunk_h_prep.cpp`) and run `test_chunk_h.py
--platform a2a3`. Validated 2026-07-27 on PR #2135 / ptoas 0.52:

| case | tile | result (PIPE_V) |
|------|------|-----------------|
| C128_N2, C128_N4 | C=128 | **PASS** |
| C32_N2 | C=32 | **FAIL** — s_snap max_diff 0.423 |
| C32_N4 | C=32 | **FAIL** — 0.357 |
| C32_D64_N2 / _N4 | C=32 | **FAIL** — 0.036 / 0.403 |
| dk32_dv64_N4 | C=32 | **FAIL** — 0.707 |
| dk64_dv32_N2 | C=64 | PASS |

Restoring `pipe_barrier(PIPE_ALL)` makes every case pass. `decay = exp(g_total)` in
the same kernel (no `TCOLEXPAND`) is correct at both sizes, isolating the broadcast op.
The failure is on `s_snap` (state), which depends on `k_rest = k * exp(g_total - g_cs)`
where `g_total` was broadcast by the under-synchronized `TCOLEXPAND`.

## Simulator (a2a3sim)
**Does NOT reproduce on `a2a3sim`** — with the `PIPE_V` toggle, all `a2a3sim` cases
(incl. C=32: `C32_N2`, `C32_D64_N2`, `dk32_dv64_N4`) **PASS**, whereas the same cases
**FAIL on a2a3 hardware**. The simulator completes the ubuf copy synchronously, so a
`PIPE_V` barrier appears sufficient; the under-synchronization is a hardware-only
timing race. Sim gives false confidence here.

## Expected
A `pipe_barrier(PIPE_V)` around a `PIPE_V`-declared op should be sufficient to order
a following read. Either the op's declared pipe must match its actual lowering, or
the copy-pipe dependency must be emitted so the barrier waits for the copy.

## Suggested fix
Re-tag `TCOLEXPAND` in `event.hpp` to a pipe that actually drains its lowering. **Measured
on hardware, the only correct value is `PIPE_ALL`** (see the pipe matrix below) — which is
also what it defaulted to before commit `021789c0` put it in the pipe table.

## Upstream: pto-isa PR #212 — measured on a2a3, `PIPE_MTE1` is NOT correct
PR #212 retags `TCOLEXPAND` from `PIPE_V` to **`PIPE_MTE1`**. Its test-plan boxes are
unchecked and it notes the functional simulator cannot reproduce the race. We ran it on
real hardware (a2a3 910B2, device 6, `test_chunk_h.py`, one barrier line changed per run,
same session):

| barrier after `TCOLEXPAND` | C=128 | C=32 | verdict |
|---|---|---|---|
| `pipe_barrier(PIPE_V)` — current `main` | PASS | **FAIL** wrong values (0.036–0.71) | the original bug |
| `pipe_barrier(PIPE_MTE1)` — PR #212 via `TSYNC_IMPL` | **FAIL** 507018 → hang | **FAIL** hang | **worse — faults the core** |
| `set_flag/wait_flag(MTE1→V)` — PR #212 via `Event<TCOLEXPAND,TSUB>` | **FAIL** 507018 | **FAIL** | **worse — faults the core** |
| `pipe_barrier(PIPE_MTE3)` | PASS | **FAIL** wrong values (0.08–0.43) | insufficient (same shape as the bug) |
| `pipe_barrier(PIPE_ALL)` | PASS | **PASS** | **only correct option** (our workaround) |

`PIPE_MTE1` fails on **both** consumers of the tag: `TSYNC_IMPL<Op>` emits
`pipe_barrier(declared)`, and `Event<SrcOp,DstOp>` emits cross-pipe
`set_flag/wait_flag(srcPipe→dstPipe)`. Both raise `507018 ACL_ERROR_RT_AICPU_EXCEPTION` on
the *first* case (`C128_N2`, which passes even with the buggy `PIPE_V`), then the device
stalls (`S1:running-stalled`). This is consistent with MTE1 being the L1→L0A/L0B pipe on the
**cube (AIC)** side — every other `PIPE_MTE1`-tagged op in `event.hpp` (`TMOV_M2B/M2L/M2R`,
`TEXTRACT_M2LR`, `TIMG2COL`) is an L1→L0 transfer — so barriering/flagging on it inside an
**AIV vector kernel** is invalid. `TCOLEXPAND`'s lowering is a UB→UB `copy_ubuf_to_ubuf`,
which is not the MTE1 path.

**Control (rules out a degraded device):** immediately after the `PIPE_MTE1` run failed 8/8,
restoring `pipe_barrier(PIPE_ALL)` on the same device passed **8/8**.

**Recommendation:** tag `TCOLEXPAND` as `PIPE_ALL`, restoring the pre-`021789c0` behavior
that PR #212's own description calls "safe". `TSYNC_IMPL` explicitly permits `PIPE_ALL`;
`Event<TCOLEXPAND,…>` would then static-assert, which is the desired outcome — a copy-engine
op has no single source pipe to hang a cross-pipe flag on, so callers must use a full fence.

## Workaround in our code
`chunk_h_prep.cpp` uses `pipe_barrier(PIPE_ALL)` after `TCOLEXPAND` (comment in place).
