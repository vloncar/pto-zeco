# `pl.transpose` feeding a matmul emits MLIR that ptoas refuses (pypto main)

**Status:** FOUND 2026-08-27 while moving our pin from `71020585` to `main` (`42881d7b`).
Not filed yet. **Blocks that pin move**: every transpose our GLA kernels feed into a matmul
stops compiling, which is most of them.
**Severity:** compile-time failure with a confusing message — the error names a ptoas MLIR
parse error, so it reads as a toolchain-version problem, and it is not: pypto's own
`toolchain/versions.env` pins `PTOAS_VERSION=v0.57`, which is exactly what we are running.

## What happens

```
ptoas compilation failed: error: use of value '%transpose_tmp' expects different type than
prior uses: '!pto.tile_buf<mat, 32x32xf32, valid=?x?, blayout=col_major, slayout=row_major>'
                       vs '!pto.tile_buf<vec, 32x32xf32, valid=?x?>'
Error: Failed to parse MLIR.
```

In the generated `.pto`, the scratch tile is **allocated** in the vector space and **used**
in the matrix space:

```mlir
%transpose_tmp = pto.alloc_tile ... : !pto.tile_buf<loc=vec,  ...>
pto.ttrans ins(%t__tmp_v1, %transpose_tmp : !pto.tile_buf<loc=mat, ...>, !pto.tile_buf<loc=mat, ...>)
```

## Root cause

`src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp` materialises the `ttrans` scratch while
flattening, and picks its memory space **then**:

```cpp
MemorySpace scratch_mem =
    in_type->memory_space_.has_value() ? *in_type->memory_space_ : MemorySpace::Vec;
```

`#2475` ("make an unset tile memory space mean 'the compiler places it'") changed what the
`else` branch means. An unset space is no longer "vector"; it is "not decided yet". So when
the transpose's input is later placed in `Mat` — which is what happens when the transposed
value is a matmul operand — the scratch keeps the `Vec` that was baked in before the decision,
and the two disagree. The scratch's space has to follow the input's *final* placement, not the
placement known at flatten time.

## Reproducer

`../../devtools/a7_transpose_probe.py` — one loaded tile, one `pl.transpose`, one `pl.matmul`,
compile only, no NPU:

| variant | our pin `71020585` | pypto main `42881d7b` |
|---|---|---|
| transposed value straight into the matmul | compiles | **FAIL** |
| separate load for each consumer | compiles | **FAIL** |
| a vector no-op (`pl.mul(..., 1.0)`) between transpose and matmul | compiles | compiles |

The third row is the tell: forcing the transposed value to be vector-resident makes the
scratch's baked-in `Vec` correct again.

## Why we are not shipping the workaround

Our forward and backward kernels have roughly a dozen transposes feeding matmuls. Inserting a
vector pass after each one costs both time and vector-buffer space — and vector-buffer space
is exactly what bounds the shapes we can run (A6 raised the backward's ceiling by fighting for
a few hundred bytes). Paying that to work around a compiler defect would give back the shapes
the last two tasks bought.

## Before filing

Re-test against `origin/main` at the time of filing — this write-up is against `42881d7b`,
and the last two dependency bugs written up here were already fixed upstream by the time we
looked. Isolated env: `../../devtools/tq_env_main.sh` (pypto main + simpler `799640e6` +
pto-isa `cd4a3d3f`, nothing under `/opt` modified).
