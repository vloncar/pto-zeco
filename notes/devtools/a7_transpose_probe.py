#!/usr/bin/env python3
"""Minimal case for the `%transpose_tmp` type clash that appears on pypto main.

`gla_grad_h` stops compiling when the pin moves to pypto main: ptoas rejects the generated
MLIR with "use of value '%transpose_tmp' expects different type than prior uses:
tile_buf<mat, ...> vs tile_buf<vec, ...>". The forward compiles clean, so it is not the
toolchain refusing everything -- something about this pattern.

The pattern under suspicion: ONE loaded tile feeding both a VECTOR op and a MATMUL operand
via `pl.transpose`. Two variants, so the answer names the trigger rather than the file:

  shared  -- one load, used by pl.mul AND by pl.transpose -> matmul
  split   -- two loads of the same block, one for each use

Compile-only; run it under both env shims to compare pins.
"""
from __future__ import annotations

import sys

import pypto.language as pl
from pypto import ir

C, DK, DV = 32, 32, 32


def build(shared: bool, staged: bool):
    @pl.program
    class TransposeProbe:
        @pl.function(type=pl.FunctionType.InCore)
        def k(
            self,
            S: pl.Tensor[[DK, DV], pl.FP32],
            P: pl.Tensor[[DK, DV], pl.FP32],
            V: pl.Tensor[[C, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[C, DK], pl.FP32]],
            R: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            s = pl.load(S, [0, 0], [DK, DV])
            v = pl.load(V, [0, 0], [C, DV])
            prod = pl.mul(s, pl.load(P, [0, 0], [DK, DV]))
            if staged:
                # A vector no-op between the transpose and the matmul, to see whether pinning
                # the transposed value to the VECTOR unit keeps the scratch tile's declared
                # space and its use in agreement.
                out = pl.matmul(v, pl.mul(pl.transpose(s, 0, 1), 1.0), out_dtype=pl.FP32)
            elif shared:
                # One value, two consumers: a vector multiply and a transposed matmul operand.
                out = pl.matmul(v, pl.transpose(s, 0, 1), out_dtype=pl.FP32)
            else:
                # Same arithmetic, but each consumer gets its own load.
                out = pl.matmul(v, pl.transpose(pl.load(S, [0, 0], [DK, DV]), 0, 1),
                                out_dtype=pl.FP32)
            return pl.store(out, [0, 0], O), pl.store(prod, [0, 0], R)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            S: pl.Tensor[[DK, DV], pl.FP32],
            P: pl.Tensor[[DK, DV], pl.FP32],
            V: pl.Tensor[[C, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[C, DK], pl.FP32]],
            R: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> pl.Tuple[pl.Tensor[[C, DK], pl.FP32], pl.Tensor[[DK, DV], pl.FP32]]:
            return self.k(S, P, V, O, R)

    return TransposeProbe


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    for label, shared, staged in [("shared load (one value, two uses)", True, False),
                                  ("split loads (one per use)", False, False),
                                  ("vector no-op before the matmul", False, True)]:
        try:
            ir.compile(build(shared, staged), platform=platform)
            print(f"  {label:<36} compiles")
        except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement
            msg = " ".join(str(exc).split())
            print(f"  {label:<36} FAIL: {msg[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
