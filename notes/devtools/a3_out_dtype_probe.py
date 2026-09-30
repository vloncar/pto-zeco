#!/usr/bin/env python3
"""Can the matmul RESULT be fp16, and can it be accumulated in a pl.range carry?

This matters because the tile that sizes the cross-core ring reserve is the matmul RESULT
crossing cube->vector, not its operands. At C=128 that reserve is 65536 B of a 188416 B
budget -- more than a third -- so halving the result halves the reserve too.

Must live in a real file: the parser reads the source, and a class defined on stdin fails
with "Cannot retrieve source code".
"""
from __future__ import annotations

import contextlib
import io
import sys

import pypto.language as pl
from pypto import ir

C = K = N = 64


@pl.program
class OutFp16:
    """fp32 operands, fp16 result."""

    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
           O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
        a = pl.load(A, [0, 0], [C, K])
        b = pl.load(B, [0, 0], [K, N])
        m = pl.matmul(a, b, out_dtype=pl.FP16)
        return pl.store(pl.tile.cast(m, pl.FP32), [0, 0], O)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
             O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
        return self.mm(A, B, O)


@pl.program
class AccFp16:
    """fp16 result accumulated across a pl.range carry, exactly as the score matrix would be."""

    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
           O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
        a = pl.load(A, [0, 0], [C, K])
        b = pl.load(B, [0, 0], [K, N])
        z = pl.tile.cast(pl.mul(pl.load(A, [0, 0], [C, N]), 0.0), pl.FP16)
        for i, (acc,) in pl.range(0, 2, init_values=(z,)):
            acc = pl.yield_(pl.add(acc, pl.matmul(a, b, out_dtype=pl.FP16)))
        return pl.store(pl.tile.cast(acc, pl.FP32), [0, 0], O)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
             O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
        return self.mm(A, B, O)


@pl.program
class OperandFp16OutFp32:
    """fp16 OPERANDS with an fp32 result -- narrows L0A/L0B but not the crossing tile."""

    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
           O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
        a = pl.tile.cast(pl.load(A, [0, 0], [C, K]), pl.FP16)
        b = pl.tile.cast(pl.load(B, [0, 0], [K, N]), pl.FP16)
        return pl.store(pl.matmul(a, b, out_dtype=pl.FP32), [0, 0], O)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
             O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
        return self.mm(A, B, O)


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    for name, prog in [("FP16 result accumulated in pl.range", AccFp16),
                       ("FP16 operands -> fp32 result", OperandFp16OutFp32)]:
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                ir.compile(prog, platform=platform)
            print(f"  {name:<38} compiles")
        except Exception as e:  # noqa: BLE001 -- capability report
            print(f"  {name:<38} {type(e).__name__}: {str(e).splitlines()[0][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
