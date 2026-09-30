# fp32 cube K-accumulation (TMATMUL_ACC) is unsupported on a2a3

**Status:** open — blocks GLA simpler backend F3 Phase 3 (head dim D > 128).
**Platform:** a2a3 (910B2). fp32 inputs, fp32 accumulator.

## Symptom

A tiled matmul that reduces a contraction dim `Kc > 128` must split `Kc` into
`<= 128` blocks and accumulate the partial products in the L0C accumulator across
successive cube calls — the standard K-tiling idiom:

```cpp
// ki over Kc-blocks, cTile persists (accumulate):
if (ki == 0) TMATMUL(cTile, l0a, l0b);        // init
else         TMATMUL_ACC(cTile, l0a, l0b);    // accumulate
```

For **fp16** inputs this works (it's the a2a3 `gemm_performance` / `gemm_ar` /
`allgather_gemm` and `flash_atten` idiom). For **fp32** inputs on a2a3 HW it is
**wrong**:

- Plain `TMATMUL` + `TMATMUL_ACC` (3-arg in-place *or* 4-arg cOut/cIn): the result
  is ~half-magnitude wrong (`max_diff ~0.6-0.74` on `*0.1`-scale inputs) — only one
  Kc-block's contribution survives. No crash.
- Explicit `AccPhase` (`TMATMUL<Partial>` / `TMATMUL_ACC<Partial|Final>`, any
  consistent or mixed phase): the AICore **hangs** — `run failed with code 507018`
  (drain timeout), wedging the run.
- The **CPU simulator passes** all of these (it accumulates correctly), so the bug
  is HW-only and sim cannot catch it.

Every `nK == 1` case (contraction fits one 128-block) is correct, including full
M/N **output** tiling (M,N up to 256) — only the cross-block K-accumulation fails.

## Root cause (hypothesis)

fp32 matmul on the a2a3 cube is emulated by decomposing each fp32 operand into
fp16 hi/lo parts and issuing several fp16 MMADs that themselves accumulate in L0C
(the tile doc: "on A2/A3 accumulation zero-initialises before the first phase, then
accumulates in-place"). A single fp32 `TMATMUL` is therefore already an internal
multi-phase accumulation sequence. Chaining a second `TMATMUL_ACC` re-enters that
sequence incorrectly — the second call's internal init drops the first block's
partial (→ half wrong), and the explicit-`AccPhase` path leaves the accumulator in
a state that never drains (→ 507018). There is **no fp32-input K-accumulation
reference** anywhere in `/opt/pto-isa` (all a2a3 gemms are fp16-in / fp32-acc).

## Impact on the GLA simpler backend

Only two GLA matmuls contract over the head dim `D` and thus need `Kc > 128` when
`D = 256`:

- `inter = q_eff @ S`      (NN, Kc = D)
- `Aqk   = q_eff @ k_eff^T` (NT, Kc = D)

`KV = k_rest^T @ v` (Kc = C), `intra = Aqk_m @ v` (Kc = C) and `gate_cumsum`
(Kc = C) all keep `Kc = C <= 128`, so they only need output (M/N) tiling, which
works. So Phase 2 (C != D, all dims <= 128) is unaffected; only D > 128 is blocked.

## Workaround (the F3 Phase 3 design, deferred)

Do the D-contraction K-reduction in the **vector** unit instead of the cube:

1. Give the matmul kernel `kStart` + `fullK` scalars so it computes one `<= 128`
   K-slice of a larger operand (its strided sub-tile GM load already supports this;
   the M/N tiling is HW-validated).
2. For `inter` / `Aqk` with `D > 128`, the chunk_o orchestration issues
   `ceil(D/128)` single-slice matmuls into partial buffers and sums them with the
   existing `chunk_o_elt` add (which needs its own D-column blocking — required for
   D=256 anyway).

This uses only proven single-`<= 128`-Kc `TMATMUL`s + a vector add; no fp32 cube
K-accumulation. Deferred by decision (2026-07-10) — Phase 2 (C != D, dims <= 128)
ships; D = 256 is future work.

## If revisiting the cube path

Worth trying before committing to the vector-reduce workaround: match
`flash_atten`'s deferred ping-pong sync exactly (no inline full `M` drain between
accumulating matmuls) with fp32 `BK = 64` blocks (two 64-wide L0 tiles fit the
64 KB L0A/L0B so ping-pong is possible; 128-wide fp32 fills a bank and forbids it).
The 507018 hang may be the inline `set(M,MTE1);wait` drain conflicting with the
still-open fp32 multi-phase accumulator rather than a fundamental limitation.
