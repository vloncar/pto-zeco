# TCOLEXPAND (a2a3) is declared PIPE_V but lowers to a ubuf copy → under-synchronised at small tiles

## Summary
`TCOLEXPAND(dst, src)` (broadcast a `[1,K]` row down to `[C,K]`, `pto/npu/a2a3/TColExpand.hpp`)
is tagged `PTO_DEFINE_OP_PIPE(Op::TCOLEXPAND, PIPE_V)` in `pto/common/event.hpp`, but its body
issues `pto_copy_ubuf_to_ubuf` per output row — a copy-engine op, **not** a pure PIPE_V vector
op. A `pipe_barrier(PIPE_V)` after it therefore does **not** guarantee the broadcast has landed
before a following vector op reads `dst`. The window is data-dependent on tile size: at `C=128`
the copy is large/slow enough that the consumer happens to wait, but at `C=32` the consumer
(`TSUB`) reads `dst` before the copy completes → wrong result (not a crash).

## Repro (GLA chunk_h_prep, `pto-zeco`)
`k_rest = k * exp(g_total_broadcast - g_cs)` where `g_total` is broadcast down the rows with
`TCOLEXPAND`. On a2a3 HW:

| C   | k_rest max_diff vs torch |
|-----|--------------------------|
| 128 | 2.98e-08  (PASS)         |
| 32  | 1.04       (FAIL)        |

`decay = exp(g_total)` (same kernel, no TCOLEXPAND) is correct at both sizes → isolates the
broadcast, not the load or the exp. Every other op (matmul all modes, the non-broadcast
`chunk_o_prep`) passes at `C=32`, so it is specific to the expand family.

## Workaround (in-tree, `pto-zeco`)
Strengthen the post-`TCOLEXPAND` barrier from `pipe_barrier(PIPE_V)` to `pipe_barrier(PIPE_ALL)`
in `gla/implementations/simpler/kernels/aiv/chunk_h_prep.cpp` → `C=32` k_rest back to 2.98e-08.
`chunk_h_update`'s `TROWEXPANDMUL` was already wrapped in `PIPE_ALL` barriers, hence it was
unaffected.

## Proper upstream fix
Either make the expand-family ops (`TCOLEXPAND*`, `TROWEXPAND*`) carry the correct pipe of their
actual lowering (copy engine), or have PyPTO/pto-isa emit the copy-pipe dependency so a
`pipe_barrier(PIPE_V)` around a declared-PIPE_V op is actually sufficient. Until then, callers
must use `PIPE_ALL` (or an explicit copy-pipe flag/wait) around these ops.

Env: pto-isa `/opt/pto-isa`, a2a3 (910B2), fp32.

## Re-confirmed still-real 2026-07-27 (current pto-isa `/opt/pto-isa`)
Static check on the current tree: `include/pto/common/event.hpp:269`
`PTO_DEFINE_OP_PIPE(Op::TCOLEXPAND, PIPE_V)` (still PIPE_V), while
`include/pto/npu/a2a3/TColExpand.hpp:28` still lowers the body to `pto_copy_ubuf_to_ubuf` per row
(copy engine, not a pure PIPE_V op). So the tag↔lowering mismatch — and our `PIPE_ALL` workaround
in `chunk_h_prep.cpp` — are still needed. **Pursue upstream:** re-tag the expand family to the
copy pipe in `event.hpp`, or emit the copy-pipe dependency. Small, contained pto-isa change.
