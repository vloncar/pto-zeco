#!/usr/bin/env python3
"""F3.1c: characterise the tall-matmul defect — where exactly is the boundary, and is it fp32-only?

`f31c_matmul_probe.py` established that a plain `[M,K] @ [K,N]` on a2a3 hardware is wrong
whenever **N < M**, correct when N == M or N > M, with K irrelevant:

    64x64x64  ok      64x64x32  WRONG (rel 1.2)     32x32x64  ok
    32x32x32  ok      32x32x16  WRONG (rel 0.9)     64x64x128 ok
    128x128x128 ...   128x64x64 WRONG (rel 1.2)

128x64x64 is the case that makes it *relative*: N=64 fails at M=128 but passes at M=64, so
this is not "N below some fixed fractal width". This sweep answers what an upstream report
has to state precisely:

  1. **Is it exactly N < M?** Includes N == M with N < K (`32x64x32`), which the N<M rule
     says must PASS, and several N<M shapes at different M.
  2. **Is it fp32-only?** a2a3's fp32 cube path already has one known hole (fp32 K-blocked
     accumulation, `allscan/issues/fp32-cube-k-accumulation/`), so whether fp16 shares this
     decides if it is one defect or two.
  3. **Does the destination width matter?** Storing `[M,N]` into a `[M, 2N]` tensor keeps
     the matmul identical and changes only the store stride, separating "the matmul result
     is wrong" from "the store writes it wrong".

Usage: python3 devtools/f31c_matmul_boundary.py <device> <platform>
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl

# (M, K, N, expectation-under-the-N<M-rule)
SHAPES = [
    (16, 16, 16, "ok"),
    (32, 32, 32, "ok"),
    (128, 128, 128, "ok"),
    (32, 64, 32, "ok    <- N==M but N<K: rule says PASS"),
    (64, 128, 64, "ok    <- N==M, N<K"),
    (16, 32, 32, "ok    <- wide"),
    (32, 32, 16, "WRONG"),
    (128, 128, 64, "WRONG"),
    (128, 128, 32, "WRONG"),
    (32, 16, 16, "WRONG <- small, N<M"),
]

# Same question in fp16 (inputs fp16, fp32 accumulate) on the shapes that bracket it.
FP16_SHAPES = [(64, 64, 64, "ok"), (64, 64, 32, "WRONG?"), (32, 32, 16, "WRONG?")]


def build(M: int, K: int, N: int, dt: str = "FP32", dest_mult: int = 1):
    """Z = X @ Y, stored at [0,0] into a [M, N*dest_mult] destination."""
    dn = N * dest_mult
    src = f'''
@pl.program
class MatmulBoundary:
    @pl.function(type=pl.FunctionType.InCore)
    def mm(
        self,
        X: pl.Tensor[[{M}, {K}], pl.{dt}],
        Y: pl.Tensor[[{K}, {N}], pl.{dt}],
        Z: pl.Out[pl.Tensor[[{M}, {dn}], pl.FP32]],
    ) -> pl.Tensor[[{M}, {dn}], pl.FP32]:
        x = pl.load(X, [0, 0], [{M}, {K}])
        y = pl.load(Y, [0, 0], [{K}, {N}])
        z = pl.matmul(x, y, out_dtype=pl.FP32)
        return pl.store(z, [0, 0], Z)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(
        self,
        X: pl.Tensor[[{M}, {K}], pl.{dt}],
        Y: pl.Tensor[[{K}, {N}], pl.{dt}],
        Z: pl.Out[pl.Tensor[[{M}, {dn}], pl.FP32]],
    ) -> pl.Tensor[[{M}, {dn}], pl.FP32]:
        return self.mm(X, Y, Z)
'''
    return pl.parse(src)


def run_one(ir, RunConfig, platform, device, M, K, N, dt="FP32", dest_mult=1):
    torch.manual_seed(7)
    tdt = torch.float16 if dt == "FP16" else torch.float32
    X = torch.randn(M, K).to(tdt)
    Y = torch.randn(K, N).to(tdt)
    ref = (X.float() @ Y.float())
    try:
        compiled = ir.compile(build(M, K, N, dt, dest_mult), platform=platform)
        Z = torch.zeros(M, N * dest_mult, dtype=torch.float32)
        compiled(X, Y, Z, config=RunConfig(platform=platform, device_id=device))
        rel = (Z[:, :N] - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
        tol = 1e-2 if dt == "FP16" else 1e-3
        return f"{rel:.2e}" + ("" if rel < tol else " WRONG")
    except Exception as exc:  # noqa: BLE001 - a build/run failure is a result here
        return f"ERR {type(exc).__name__}: {str(exc).splitlines()[0][:60]}"


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2]

    from canary import BAD_CARD_EXIT, check
    if platform != "a2a3sim" and not check(device, platform):
        return BAD_CARD_EXIT

    from pypto import ir
    from pypto.runtime.runner import RunConfig

    print(f"\n=== fp32: is the rule exactly N < M?   ({platform} dev {device})")
    print(f"{'M':>4} {'K':>4} {'N':>4}  {'result':<14} expected")
    print("-" * 60)
    for (M, K, N, exp) in SHAPES:
        print(f"{M:>4} {K:>4} {N:>4}  "
              f"{run_one(ir, RunConfig, platform, device, M, K, N):<14} {exp}")

    print("\n=== fp16 inputs (fp32 accumulate): same defect, or fp32-only?")
    print(f"{'M':>4} {'K':>4} {'N':>4}  {'result':<14} expected")
    print("-" * 60)
    for (M, K, N, exp) in FP16_SHAPES:
        print(f"{M:>4} {K:>4} {N:>4}  "
              f"{run_one(ir, RunConfig, platform, device, M, K, N, 'FP16'):<14} {exp}")

    print("\n=== does the DESTINATION width matter? (same matmul, wider store target)")
    print(f"{'M':>4} {'K':>4} {'N':>4}  {'dest [M,N]':<14} {'dest [M,2N]':<14}")
    print("-" * 60)
    for (M, K, N) in [(64, 64, 32), (32, 32, 16), (64, 64, 64)]:
        a = run_one(ir, RunConfig, platform, device, M, K, N, "FP32", 1)
        b = run_one(ir, RunConfig, platform, device, M, K, N, "FP32", 2)
        print(f"{M:>4} {K:>4} {N:>4}  {a:<14} {b:<14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
