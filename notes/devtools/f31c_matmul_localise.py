#!/usr/bin/env python3
"""F3.1c: is the tall matmul's DATA wrong, or only its store?

`f31c_matmul_probe.py` showed `[M,K] @ [K,N]` is wrong on a2a3 whenever N < M. Both the
"store straight to GM" and the "pass through a vector op first" variants failed identically,
so the cross-core pipe is not involved — but both variants still end in a `TSTORE` of a tall
`[M, N]` tile, so they cannot tell a wrong *product* from a wrong *store*.

A detail from the GLA kernel says the store is the more likely culprit: there,
`b = tril[C,C] @ la[C,dk]` is tall whenever dk < C, yet `C=64, dk=32, dv=64` is CORRECT on
hardware. The difference is that `b` is consumed on-chip and never stored to GM, while the
output `o_n` (tall exactly when dv < C) is stored. That predicts dk-tall matmuls are fine
and only tall *stores* corrupt — which is precisely the observed "dv < C, dk irrelevant" rule.

Three variants of the same product, per shape:

  A  store [M,N] directly            -- the known-bad baseline
  B  pad N up to M, store [M,M]      -- is padding a usable workaround?
  C  transpose to [N,M], store wide  -- SAME tall matmul, but a wide store

C is the decisive one. If C is correct, the tall `TMATMUL` produced correct data and the
defect is in storing a tall tile. If C is also wrong, the product itself is wrong.

Usage: python3 devtools/f31c_matmul_localise.py <device> <platform>
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl

# 128x128x64 is omitted: the transpose variant needs an extra [M,N] vector tile and
# overflows the 184 KB vector buffer, so the three variants would not be comparable.
SHAPES = [(64, 64, 32), (32, 32, 16), (64, 64, 16), (64, 64, 64)]


def build_direct(M, K, N):
    return pl.parse(f'''
@pl.program
class MMDirect:
    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, X: pl.Tensor[[{M}, {K}], pl.FP32], Y: pl.Tensor[[{K}, {N}], pl.FP32],
           Z: pl.Out[pl.Tensor[[{M}, {N}], pl.FP32]]) -> pl.Tensor[[{M}, {N}], pl.FP32]:
        x = pl.load(X, [0, 0], [{M}, {K}])
        y = pl.load(Y, [0, 0], [{K}, {N}])
        z = pl.matmul(x, y, out_dtype=pl.FP32)
        return pl.store(z, [0, 0], Z)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, X: pl.Tensor[[{M}, {K}], pl.FP32], Y: pl.Tensor[[{K}, {N}], pl.FP32],
             Z: pl.Out[pl.Tensor[[{M}, {N}], pl.FP32]]) -> pl.Tensor[[{M}, {N}], pl.FP32]:
        return self.mm(X, Y, Z)
''')


def build_padded(M, K, N):
    """Y is passed already zero-padded to [K, M]; the product is square."""
    return pl.parse(f'''
@pl.program
class MMPadded:
    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, X: pl.Tensor[[{M}, {K}], pl.FP32], Y: pl.Tensor[[{K}, {M}], pl.FP32],
           Z: pl.Out[pl.Tensor[[{M}, {M}], pl.FP32]]) -> pl.Tensor[[{M}, {M}], pl.FP32]:
        x = pl.load(X, [0, 0], [{M}, {K}])
        y = pl.load(Y, [0, 0], [{K}, {M}])
        z = pl.matmul(x, y, out_dtype=pl.FP32)
        return pl.store(z, [0, 0], Z)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, X: pl.Tensor[[{M}, {K}], pl.FP32], Y: pl.Tensor[[{K}, {M}], pl.FP32],
             Z: pl.Out[pl.Tensor[[{M}, {M}], pl.FP32]]) -> pl.Tensor[[{M}, {M}], pl.FP32]:
        return self.mm(X, Y, Z)
''')


def build_transposed(M, K, N):
    """Same tall matmul, then transpose so the STORE is wide instead of tall."""
    return pl.parse(f'''
@pl.program
class MMTransposed:
    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, X: pl.Tensor[[{M}, {K}], pl.FP32], Y: pl.Tensor[[{K}, {N}], pl.FP32],
           Z: pl.Out[pl.Tensor[[{N}, {M}], pl.FP32]]) -> pl.Tensor[[{N}, {M}], pl.FP32]:
        x = pl.load(X, [0, 0], [{M}, {K}])
        y = pl.load(Y, [0, 0], [{K}, {N}])
        z = pl.matmul(x, y, out_dtype=pl.FP32)
        zv = pl.mul(z, 1.0)
        zt = pl.transpose(zv, 0, 1)
        return pl.store(zt, [0, 0], Z)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, X: pl.Tensor[[{M}, {K}], pl.FP32], Y: pl.Tensor[[{K}, {N}], pl.FP32],
             Z: pl.Out[pl.Tensor[[{N}, {M}], pl.FP32]]) -> pl.Tensor[[{N}, {M}], pl.FP32]:
        return self.mm(X, Y, Z)
''')


def rel(got, want):
    return (got - want).abs().max().item() / max(want.abs().max().item(), 1e-30)


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2]

    from canary import BAD_CARD_EXIT, check
    if platform != "a2a3sim" and not check(device, platform):
        return BAD_CARD_EXIT

    from pypto import ir
    from pypto.runtime.runner import RunConfig

    cfg = lambda: RunConfig(platform=platform, device_id=device)  # noqa: E731

    print(f"\n=== same product, three ways to get it out   ({platform} dev {device})")
    print(f"{'M':>4} {'K':>4} {'N':>4}  {'A store [M,N]':<16} {'B pad->[M,M]':<16} {'C transpose->[N,M]':<18}")
    print("-" * 76)
    for (M, K, N) in SHAPES:
        torch.manual_seed(7)
        X = torch.randn(M, K)
        Y = torch.randn(K, N)
        ref = X @ Y
        out = []

        try:
            Z = torch.zeros(M, N)
            ir.compile(build_direct(M, K, N), platform=platform)(X, Y, Z, config=cfg())
            out.append(f"{rel(Z, ref):.2e}")
        except Exception as e:  # noqa: BLE001
            out.append(f"ERR {type(e).__name__}")

        try:
            Ypad = torch.zeros(K, M)
            Ypad[:, :N] = Y
            Z = torch.zeros(M, M)
            ir.compile(build_padded(M, K, N), platform=platform)(X, Ypad, Z, config=cfg())
            out.append(f"{rel(Z[:, :N], ref):.2e}")
        except Exception as e:  # noqa: BLE001
            out.append(f"ERR {type(e).__name__}")

        try:
            Z = torch.zeros(N, M)
            ir.compile(build_transposed(M, K, N), platform=platform)(X, Y, Z, config=cfg())
            out.append(f"{rel(Z, ref.t()):.2e}")
        except Exception as e:  # noqa: BLE001
            out.append(f"ERR {type(e).__name__}")

        marks = [("" if x.startswith("0") or "e-" in x and float(x.split()[0]) < 1e-3 else " W")
                 if not x.startswith("ERR") else "" for x in out]
        print(f"{M:>4} {K:>4} {N:>4}  {out[0] + marks[0]:<16} {out[1] + marks[1]:<16} "
              f"{out[2] + marks[2]:<18}")

    print("\nC correct + A wrong  => the tall TMATMUL is fine; storing a tall tile is the bug.")
    print("C wrong too          => the product itself is wrong.")
    print("B correct            => padding N up to M is a usable workaround.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
