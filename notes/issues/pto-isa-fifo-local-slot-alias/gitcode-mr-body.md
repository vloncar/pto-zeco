Fixes #521.

A cross-core `TPipe`'s consumer-side **local** ring was strided by the popped tile's own
size, while the **GM** ring in the same function is strided by the ring's fixed `SLOT_SIZE`:

```cpp
size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;   // GM: fixed
...
uint64_t localTileBase = fifo.V2C_CONSUMER_BUF
                       + (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
```

`SLOT_SIZE` is fixed at the pipe, but `TPOP` is templated per call site, so one ring may
carry tiles of different shapes. Slots must sit at fixed positions either way, so slot `i+1`
of a *smaller* tile landed inside slot `i` of a larger one and two tiles the FIFO considered
to be in different slots shared local memory. This changes the local stride to `SLOT_SIZE`
in both pop paths.

Through PyPTO the visible effect was `pl.matmul` silently returning wrong results whenever
the result tile is taller than wide (`N < M`), with no error or warning. The predicate is
exact: the two operands occupy `M*K` and `K*N` bytes at local offsets `0` and `K*N`, which
overlap iff `K*N < M*K` — `K` cancels. Issue #521 has the full 19-shape table and the
analysis.

## Validation (a2a3 hardware)

With only this change, nothing touched in PyPTO or PTOAS:

| | before | after |
|---|---|---|
| 18-config `pl.matmul` sweep | 8/18 | **18/18** |
| sequence-parallel GLA operator, 7 shapes | 3/7 | **7/7** |
| its worst case | max diff **72** | **2.67e-05** |

No regressions — the full a2a3 ST pipe suite against the patched header: `tpushpop_vc` (13),
`tpushpop_cv` (10), `tpushpop_vc_nosplit` (6), `tpushpop_cv_nosplit` (3), `tpushpop_fixpipe`
(4), `tpushpop_dir_both` (2), `tpushpop_dir_both_concurrent` (2), `tpushpop_subtile` (1),
`tmatmul` (16), plus the new case — 10/10 testcases green.

## Testcase

**`tpushpop_mixed_tile_size`** — the vector core pushes two differently-sized matmul operands
through one ring; the cube pops both before consuming either, then multiplies. Fails before
this change on the 4 unequal-tile cases, passes after. `case2`/`case5` push equal-sized tiles,
where the two strides coincide, and pass either way — the control that isolates tile-size
heterogeneity as the trigger rather than cross-core transport, `TMATMUL` or `TSTORE`.
`case4`–`case6` repeat the set over a `DIR_BOTH` pipe.

## Point for review

**This increases the local buffer footprint, and nothing bounds it.** The consumer now needs
`LOCAL_SLOT_NUM * SLOT_SIZE` bytes. `SLOT_SIZE >= ConsM * ConsN * sizeof(T)` always, so the
stride can only grow — including for *homogeneous* tiles, whenever a pipe declares a
`SLOT_SIZE` larger than the tile it actually carries.

There is no assertion available to catch that: `TPipe`'s constructor takes
`C2V_CONSUMER_BUF` / `V2C_CONSUMER_BUF` as bare `uint32_t` base offsets with no size
alongside them, so the pipe cannot know how much room it has. On a `DIR_BOTH` pipe whose two
bases were spaced for the old tighter packing, the C2V region could now run into the V2C one.
Every ST case passes, so nothing in-tree is affected, but out-of-tree callers might be.

If you would rather pack slots by prefix sum, or widen the constructor to take buffer sizes
and assert, say so and I will rework it — this takes the simple route deliberately.

Two smaller notes: `popVecTileFromGMFiFo` (C2V) carried the identical expression and is fixed
here too, covered indirectly by `tpushpop_vc` / `tpushpop_cv` rather than by a dedicated case.
And `npu/a5/TPush.hpp` has the same pattern — not touched, no a5 hardware here to test on.

<!-- ==================== REVISION NOTE (v2, 2026-08-12) ==================== -->
<!-- Not part of the MR body; kept here so the change is reviewable at a glance.

Addresses the bot review + the code-quality export. Squashed into the single commit.

Review findings:
 - P1 "std::vector sized by byte count": FIXED (now M*N). NB the reported failure mode was
   wrong — std::vector<T> v(n) value-initializes, so both tails were deterministically 0.0f
   and compared equal; no UB, no false pass/fail. Real cost was 4x the allocation and, more
   to the point, ResultCmp derives its error threshold from expected.size().
 - P2 unchecked ACL / rtGetC2cCtrlAddr / ReadFile returns: FIXED, using the ASSERT_ACL_OK /
   ASSERT_RT_OK macro shape already used by syncall/main.cpp.
 - P3 unused ConsM/ConsN in TPush.hpp: FIXED (removed).
 - P3 wrong suite name in a gen_data.py comment: FIXED (and the suite renamed anyway).

Code-quality export:
 - duplicate 13 lines across the two main.cpp: FIXED by dropping the cube-only testcase.
 - RunTallStore 52 nbnc lines > 50: gone with that testcase.
 - runMixedTallMatmul 100 nbnc lines > 50: FIXED, split into PushBothOperands (vector side)
   and PopBothAndMatmul (cube side).

Found while revising, missed by both passes:
 - the testcase was never added to AUTO_MODE_WHITELIST despite using TPUSH/TPOP, so auto-mode
   builds would have tried to compile it. Added.

Self-correction:
 - the cube-only testcase tmatmul_tall_output was justified by "every existing tmatmul shape
   has N >= M". False: TMATMULBIASTest 101x288x67 / 55x127x29 / 150x89x50 / 135x64x88 are all
   N < M and pass. Testcase dropped; the claim is corrected in the issue too.
 - renamed tmatmul_tall_output_mix -> tpushpop_mixed_tile_size: the defect is a TPipe FIFO
   bug and matmul is only the vehicle, so it belongs with the other tpushpop_* cases, named
   after the condition that triggers it.
-->
