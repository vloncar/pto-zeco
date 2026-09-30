## Summary

On a2a3, a cross-core `TPipe`'s **consumer-side local ring** is strided by the popped tile's
own size, while the **GM ring** in the same function is strided by the ring's fixed
`SLOT_SIZE`. Slots of a ring must be at fixed positions, so slot `i+1` of a *smaller* tile
lands **inside** slot `i` of a larger one. Two tiles that the FIFO considers to be in
different slots then share local memory and corrupt each other.

The visible effect through PyPTO is that **`pl.matmul` silently returns wrong results
whenever the result tile is taller than wide** (`N < M`). No error, no warning, no
diagnostic — the numbers are simply unrelated to the product. The simulator computes the
same shapes correctly, so it does not show up in simulation.

## Where

`include/pto/npu/a2a3/TPush.hpp`, `Consumer::popMatTileFromGMFiFo` (and identically in
`popVecTileFromGMFiFo` for the C2V direction):

```cpp
// GM ring: fixed slot stride
size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
...
// local ring: strided by THIS tile's size
uint64_t localTileBase = fifo.V2C_CONSUMER_BUF
                       + (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
```

The GM line still carries the previous formula as a trailing comment
(`// ConsM * ConsN * sizeof(T);`), which suggests the GM ring was migrated to `SLOT_SIZE`
and the local ring was not.

## Why the failure predicate is exactly `N < M`

For `[M,K] @ [K,N]` the two operands cross the pipe as Mat tiles:

* A occupies `M*K*sizeof(T)` bytes at local offset `0`
* B occupies `K*N*sizeof(T)` bytes at local offset `1 * K*N*sizeof(T)`

They overlap iff `K*N < M*K`, i.e. **`N < M`** — `K` cancels. Measured on a2a3 hardware,
fp32, this predicate matches all 19 shapes tried, including controls picked to break it:

| M | K | N | | result |
|---|---|---|---|---|
| 64 | 64 | 64 | square | ok 3.4e-07 |
| 64 | 64 | 32 | tall | **wrong, rel 1.23** |
| 64 | 64 | 16 | tall | **wrong, rel 0.91** |
| 64 | 32 | 32 | tall | **wrong, rel 1.14** |
| 128 | 64 | 64 | tall (N=64 fine at M=64, wrong at M=128) | **wrong, rel 1.18** |
| 32 | 64 | 32 | `N == M` but `N < K` | ok 4.8e-07 |
| 64 | 128 | 64 | `N == M`, `N < K` | ok 3.5e-07 |
| 32 | 32 | 64 | wide | ok 1.7e-07 |
| 64 | 64 | 128 | wide | ok 3.7e-07 |

Note it is *relative*, not an alignment threshold: `N = 64` is correct at `M = 64` and wrong
at `M = 128`.

## Reproducer

`tests/npu/a2a3/src/st/testcase/tpushpop_mixed_tile_size` (in the linked MR). The vector core
loads two operands and pushes them through a `DIR_V2C` pipe; the cube pops both and
multiplies. Popping two tiles before consuming either uses 2 of the ring's 8 slots, which is
what a FIFO of depth > 1 is for.

* `case1_unequal_64x64x32`, `case3_unequal_32x32x16` — **fail** on master
* `case2_equal_64x64x64` — passes (equal-sized tiles, every slot is `SLOT_SIZE` anyway)
* `case4`/`case5`/`case6` — same three over a `DIR_BOTH` pipe, same outcome

A cube-only variant of the same shapes, with the operands `TLOAD`ed directly instead of
pushed, passes at every one of them — which isolates the defect to the pipe rather than to
`TMATMUL` or `TSTORE`. It is not part of the MR: the existing `TMATMULBIASTest` cases
(101x288x67, 55x127x29, 150x89x50, 135x64x88) already cover cube-only `N < M`.

## Fix

Stride the local ring by `SLOT_SIZE`, matching the GM ring:

```cpp
uint64_t localTileBase = fifo.V2C_CONSUMER_BUF
                       + (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * RingFiFo::SLOT_SIZE;
```

Measured with only this change (nothing touched in PyPTO or PTOAS):

| | before | after |
|---|---|---|
| the 18-config matmul sweep above | 8/18 | **18/18** |
| a real GLA/ZeCO operator, 7 shapes | 3/7 | **7/7** |
| its worst case (`C=64, dv=32`) | max diff **72** | **2.67e-05** |

No regressions: the full a2a3 ST pipe suite is green against the patched header —
`tpushpop_vc` (13), `tpushpop_cv` (10), `tpushpop_vc_nosplit` (6), `tpushpop_cv_nosplit` (3),
`tpushpop_fixpipe` (4), `tpushpop_dir_both` (2), `tpushpop_dir_both_concurrent` (2),
`tpushpop_subtile` (1), `tmatmul` (16), plus the new case — 10/10 testcases.

## Three things worth your judgement

1. **The C2V direction has the same expression.** `popVecTileFromGMFiFo` strides
   `C2V_CONSUMER_BUF` the same payload-dependent way. The MR fixes both, but only the V2C/Mat
   path is covered by a testcase here — the C2V path is fixed by inspection.

2. **This increases the local buffer footprint.** `SLOT_SIZE` striding needs
   `LOCAL_SLOT_NUM * SLOT_SIZE` bytes where the old packing was tighter for heterogeneous
   tiles, and there is no assertion anywhere on the consumer buffer size. A program that
   sized its consumer buffer tightly could trade silent corruption for silent overflow. If
   you would rather pack the slots by prefix sum, or add a size assert, that is your call —
   the MR takes the simple route.

3. **a5 has the same pattern** in `npu/a5/TPush.hpp`. We have no a5 hardware, so we have not
   touched or tested it.

## Why this survived

Not because `N < M` was untested — four existing `tmatmul` cases already run it and pass,
since the cube loads its own operands and never touches a ring. The untested thing is
narrower: no existing case pushes tiles of *different sizes* through one ring and then reads
back the earlier one.

Found while debugging a sequence-parallel GLA operator built on PyPTO, where it presented as
"the operator is wrong whenever the head dim is smaller than the chunk size".
