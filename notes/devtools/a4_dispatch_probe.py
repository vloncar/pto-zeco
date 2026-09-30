#!/usr/bin/env python3
"""Can a Python-constant `if` in a HOST orchestrator select between two chip kernels?

The two stage2 formulations cannot be unified: at one key-row block you want the scores
accumulated across head blocks (ONE within-chunk matmul per chunk), and at more than one you
want the output accumulated (no [C,C] tile, but a matmul per head block per key-row block).
Measured, shipping only the second costs +24% at dk=dv=128.

If a constant `if` selects at parse time, both kernels can live in ONE class and the plan
picks. If not, the two formulations need separate factories -- which duplicates the ring and
the host orchestrator as well, the way P=1 vs P>1 already does.
"""
from __future__ import annotations

import contextlib
import io
import sys

import pypto.language as pl
from pypto import ir

C = 32
USE_B = False   # the compile-time choice


@pl.program
class TwoKernels:
    @pl.function(type=pl.FunctionType.InCore)
    def kern_a(self, X: pl.Tensor[[C, C], pl.FP32],
               O: pl.Out[pl.Tensor[[C, C], pl.FP32]]) -> pl.Tensor[[C, C], pl.FP32]:
        return pl.store(pl.mul(pl.load(X, [0, 0], [C, C]), 2.0), [0, 0], O)

    @pl.function(type=pl.FunctionType.InCore)
    def kern_b(self, X: pl.Tensor[[C, C], pl.FP32], Big: pl.Tensor[[512, 512], pl.FP32],
               O: pl.Out[pl.Tensor[[C, C], pl.FP32]]) -> pl.Tensor[[C, C], pl.FP32]:
        # Deliberately far too big for the vector buffer: 8 live [512, 512] fp32 tiles.
        big = pl.load(Big, [0, 0], [512, 512])
        acc = pl.mul(big, 1.5)
        for i, (a,) in pl.range(0, 8, init_values=(acc,)):
            a = pl.yield_(pl.add(a, pl.mul(big, 2.0)))
        return pl.store(pl.mul(pl.load(X, [0, 0], [C, C]), 3.0), [0, 0], O)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_a(self, X: pl.Tensor[[C, C], pl.FP32],
               O: pl.Out[pl.Tensor[[C, C], pl.FP32]]) -> pl.Tensor[[C, C], pl.FP32]:
        return self.kern_a(X, O)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_b(self, X: pl.Tensor[[C, C], pl.FP32], Big: pl.Tensor[[512, 512], pl.FP32],
               O: pl.Out[pl.Tensor[[C, C], pl.FP32]]) -> pl.Tensor[[C, C], pl.FP32]:
        return self.kern_b(X, Big, O)

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(self, X: pl.Tensor[[1, C, C], pl.FP32], Big: pl.Tensor[[512, 512], pl.FP32],
                  O: pl.Out[pl.Tensor[[1, C, C], pl.FP32]]) -> pl.Tensor[[1, C, C], pl.FP32]:
        for r in pl.range(1):
            X_r = X[r]
            O_r = O[r]
            if USE_B:
                self.chip_b(X_r, Big, O_r, device=r)
            else:
                self.chip_a(X_r, O_r, device=r)
        return O


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ir.compile(TwoKernels, platform=platform)
        print("  constant-if dispatch, UNSELECTED kernel cannot fit: COMPILES "
              "-> only dispatched kernels are built")
    except Exception as e:  # noqa: BLE001 -- capability report
        print(f"  constant-if dispatch in host_orch: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
