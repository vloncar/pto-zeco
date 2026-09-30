#!/usr/bin/env python3
"""A3 capability probe: what can actually shrink the C=128 working set?

Kernel bodies are written out literally -- the parser reads this source, so a dtype or a
flag passed through a closure is rejected ("Unsupported closure variable type"). Each case
is behind a factory so that one rejection does not abort the others.

Cases:
  fp32 baseline            reference
  fp16 x fp16              narrow operands (mixed fp16 x fp32 is rejected by the op itself)
  bf16 x bf16              ditto, fewer mantissa bits
  matmul_acc, straight     does the L0C accumulator work in fp32 at all?
  matmul_acc, loop-carried does it survive being a pl.range carry -- which is how the chunk
                           kernels would actually use it, to keep the [C,C] score accumulator
                           and the [C,BV] output accumulator OUT of the vector buffer
Usage: a3_capability_probe.py <platform> [device]
"""
from __future__ import annotations

import contextlib
import io
import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

C = K = N = 64
TRIPS = 3


def make_base():
    @pl.program
    class Base:
        @pl.function(type=pl.FunctionType.InCore)
        def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
               O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            a = pl.load(A, [0, 0], [C, K])
            b = pl.load(B, [0, 0], [K, N])
            return pl.store(pl.matmul(a, b, out_dtype=pl.FP32), [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            return self.mm(A, B, O)
    return Base


def make_fp16():
    @pl.program
    class Fp16:
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
    return Fp16


def make_bf16():
    @pl.program
    class Bf16:
        @pl.function(type=pl.FunctionType.InCore)
        def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
               O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            a = pl.tile.cast(pl.load(A, [0, 0], [C, K]), pl.BF16)
            b = pl.tile.cast(pl.load(B, [0, 0], [K, N]), pl.BF16)
            return pl.store(pl.matmul(a, b, out_dtype=pl.FP32), [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            return self.mm(A, B, O)
    return Bf16


def make_acc_straight():
    @pl.program
    class AccStraight:
        @pl.function(type=pl.FunctionType.InCore)
        def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
               O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            a = pl.load(A, [0, 0], [C, K])
            b = pl.load(B, [0, 0], [K, N])
            acc = pl.matmul(a, b, out_dtype=pl.FP32)
            acc = pl.tile.matmul_acc(acc, a, b)
            acc = pl.tile.matmul_acc(acc, a, b)
            return pl.store(acc, [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            return self.mm(A, B, O)
    return AccStraight


def make_acc_loop():
    @pl.program
    class AccLoop:
        @pl.function(type=pl.FunctionType.InCore)
        def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
               O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            a = pl.load(A, [0, 0], [C, K])
            b = pl.load(B, [0, 0], [K, N])
            acc0 = pl.matmul(a, b, out_dtype=pl.FP32)
            for i, (acc,) in pl.range(0, TRIPS - 1, init_values=(acc0,)):
                acc = pl.yield_(pl.tile.matmul_acc(acc, a, b))
            return pl.store(acc, [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            return self.mm(A, B, O)
    return AccLoop


CASES = [
    ("fp32 x fp32 (baseline)", make_base, 1),
    ("fp16 x fp16", make_fp16, 1),
    ("bf16 x bf16", make_bf16, 1),
    ("fp32 matmul_acc, straight", make_acc_straight, TRIPS),
    ("fp32 matmul_acc, LOOP CARRY", make_acc_loop, TRIPS),
]


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.manual_seed(3)
    A, B = torch.randn(C, K), torch.randn(K, N)
    for name, factory, mult in CASES:
        gold = (A.double() @ B.double()) * mult
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                compiled = ir.compile(factory(), platform=platform)
            O = torch.zeros(C, N)
            compiled(A, B, O, config=RunConfig(platform=platform, device_id=dev))
            e = (O.double() - gold).abs().max().item()
            print(f"  {name:<30} RUNS   max|err|={e:.3e}  rel={e / gold.abs().max().item():.2e}")
        except Exception as e:  # noqa: BLE001 -- reporting capability, not handling
            print(f"  {name:<30} {type(e).__name__}: {str(e).splitlines()[0][:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------
# Follow-up: matmul_acc has to be SEEDED. In the chunk kernels the accumulator starts at
# zero, and the cheapest zero available is a loaded one (that is how `zc` already seeds the
# output accumulator). If matmul_acc accepts a loaded tile, the kernel change is a one-line
# swap; if it insists on a tile that is already a matmul result, the seed costs an extra
# matmul per chunk.
def make_acc_from_loaded_zero():
    @pl.program
    class AccFromZero:
        @pl.function(type=pl.FunctionType.InCore)
        def mm(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
               Z: pl.Tensor[[C, N], pl.FP32],
               O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            a = pl.load(A, [0, 0], [C, K])
            b = pl.load(B, [0, 0], [K, N])
            z = pl.load(Z, [0, 0], [C, N])
            for i, (acc,) in pl.range(0, TRIPS, init_values=(z,)):
                acc = pl.yield_(pl.tile.matmul_acc(acc, a, b))
            return pl.store(acc, [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, A: pl.Tensor[[C, K], pl.FP32], B: pl.Tensor[[K, N], pl.FP32],
                 Z: pl.Tensor[[C, N], pl.FP32],
                 O: pl.Out[pl.Tensor[[C, N], pl.FP32]]) -> pl.Tensor[[C, N], pl.FP32]:
            return self.mm(A, B, Z, O)
    return AccFromZero


if __name__ == "__main__" or True:
    import torch as _t
    _plat = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    _dev = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    _t.manual_seed(3)
    _A, _B = _t.randn(C, K), _t.randn(K, N)
    _gold = (_A.double() @ _B.double()) * TRIPS
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _c = ir.compile(make_acc_from_loaded_zero(), platform=_plat)
        _O = _t.zeros(C, N)
        _c(_A, _B, _t.zeros(C, N), _O, config=RunConfig(platform=_plat, device_id=_dev))
        _e = (_O.double() - _gold).abs().max().item()
        print(f"  {'matmul_acc from LOADED ZERO':<30} RUNS   max|err|={_e:.3e}  "
              f"rel={_e / _gold.abs().max().item():.2e}")
    except Exception as _e:  # noqa: BLE001
        print(f"  {'matmul_acc from LOADED ZERO':<30} {type(_e).__name__}: "
              f"{str(_e).splitlines()[0][:88]}")
