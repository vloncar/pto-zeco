#!/usr/bin/env python3
"""Does `pl.full` work where `pl.create_tensor(init_value=...)` used to?

Upstream removed `init_value` from `tensor.create` (pypto #2530, 2026-08-27): passing anything
but None now raises, because the runtime dropped its create-info fill. The replacement it
names is `pl.full`, which materialises a constant tensor. Our four `init_value=0` call sites
break on the next pin bump, so the question is whether the replacement is usable from a HOST
orchestrator -- and, separately, whether it finally gives us the ONES column that the original
bug (#2505) denied us.

Compile-only: no NPU needed.
"""
from __future__ import annotations

import sys

import pypto.language as pl
from pypto import ir

C, DK = 32, 32


def build(value: float, cols: int, literal: bool = False):
    @pl.program
    class FullProbe:
        @pl.function(type=pl.FunctionType.InCore)
        def copy(self, Z: pl.Tensor[[C, cols], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, cols], pl.FP32]]) -> pl.Tensor[[C, cols], pl.FP32]:
            return pl.store(pl.load(Z, [0, 0], [C, cols]), [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, Z: pl.Tensor[[C, cols], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, cols], pl.FP32]]) -> pl.Tensor[[C, cols], pl.FP32]:
            return self.copy(Z, O)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, O: pl.Out[pl.Tensor[[1, C, cols], pl.FP32]]):
            # The VALUE is written as a literal in one variant and read from the enclosing
            # scope in the other: pypto parses this source, and a closure name can reach the
            # op as an ir::Expr where it wants a Python float -- the same trap as the
            # ring-depth attribute (C2 in ROADMAP.md).
            if literal:
                zc = pl.full([C, cols], dtype=pl.FP32, value=0.0)
            else:
                zc = pl.full([C, cols], pl.FP32, value)
            for r in pl.range(1):
                self.chip(zc, O[r], device=r)
            return O

    return FullProbe


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    for label, value, cols, lit in [
            ("full(value from scope)", 0.0, DK, False),
            ("full(dtype=, value=) kwargs", 0.0, DK, True),
            ("full(dtype=, value=) kwargs, [C,1]", 0.0, 1, True)]:
        try:
            ir.compile(build(value, cols, lit), platform=platform)
            print(f"  {label:<38} compiles")
        except Exception as exc:  # noqa: BLE001 -- a refusal IS the answer
            print(f"  {label:<38} {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
