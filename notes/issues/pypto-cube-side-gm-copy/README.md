# A pre-loop GM→GM copy is placed on the CUBE core when the kernel has a nested loop

**Status:** live, reproduced on a2a3 hardware, not yet filed upstream.
**Found:** 2026-08-21, doing task 5 (head-dim blocking for the pypto GLA chunk kernels).

## Symptom

The device kernel fails to build:

```
/opt/pto-isa/include/pto/npu/a2a3/TLoad.hpp:24:9: error: function type
'void (__ubuf__ void *, __gm__ void *, ...)' of 'copy_gm_to_ubuf_align_b32'
does not support the given target feature
```

and the matching `copy_ubuf_to_gm_align_b32` from `TStore.hpp`. The cube core has no vector
buffer, so a `TLOAD`/`TSTORE` between global memory and it cannot exist.

## What triggers it

An InCore kernel that

1. copies a tensor to another tensor **before** its main loop —

       sws = pl.store(pl.load(Srecv, [0, 0], [DK, DV]), [0, 0], Sws)

2. and contains a **nested** `pl.range` whose body consumes `Sws` in a matmul.

With a single-trip inner loop the copy is emitted into the vector half and everything builds.
Make the inner loop real and the same three statements move to the cube half:

```c++
#if defined(__DAV_CUBE__)
  Tile<TileType::Vec, float, 128, 128, ...> v23 = ...;
  TLOAD(v23, v27);        // Srecv  -> vector buffer      <-- on the CUBE core
  TMULS(v28, v23, v16);
  TSTORE(v32, v28);       // vector buffer -> Sws
```

Measured across shapes and plans (`../../devtools/t5_split_check.py`), counting
`TLOAD`/`TSTORE`/`TMULS` inside the `__DAV_CUBE__` region:

| blocks in the inner loop | cube-side GM↔vector ops |
|---|---|
| 1 | 0 |
| 2 | 3 |
| 4 | 3 |
| 8 | 3 |

Independent of chunk size and of both head dims — `C=16, dk=32, dv=16` shows it as readily as
`C=64, dk=128, dv=128`. The same kernel with the state kept in a **carried tile** instead of a
tensor is clean at every block count, so the trigger is specifically the pre-loop tensor copy.

## Why it is worse than an ordinary build error

**`ir.compile` returns success.** The device kernels are compiled later, when the runtime
prepares the program, so anything that only compiles the program sees a healthy result. In our
case a shape-fitting search happily selected a plan that could never run.

**a2a3sim accepts it.** The simulator has no such target restriction, so a simulator run of the
identical program passes and reports correct numbers. A simulator-only check certifies a kernel
that cannot be built for hardware — the same trap as
[fp32 cube K-accumulation](../fp32-cube-k-accumulation/), where the simulator also accumulated
happily.

## Not fixed by nudging

Putting a vector op in the middle of the copy (`pl.mul(..., 1.0)`, the idiom this codebase
already uses to detach a carry) does **not** move it back: the `TMULS` is simply emitted on the
cube side too. The placement is not following the op kind.

## Detecting it without hardware

`../../devtools/t5_split_check.py` brackets the `#if defined(__DAV_CUBE__)` region of the
generated `*_aic.cpp` — which `ir.compile` *does* write — and looks for those ops. That turns a
failure only visible at run-time on real hardware into a static check over generated source.

## What it costs us

Task 5 needs the `[dk, dv]` running state out of the vector buffer to reach value-dim 128: the
state alone is 65536 B of a 188416 B budget, and blocking the head dim cannot shrink it. Keeping
it in a tensor is the way to do that, and this defect blocks exactly that. The carried-tile form
we shipped instead reaches key-dim 128 but not value-dim 128.

## Before filing

Re-test against pypto `origin/main`, not our pin — the last two dependency bugs written up here
were already fixed upstream (see the cpu_stub cache-line macro note). The isolated main
environment is `../../devtools/pypto_main_env.sh`.
