#!/usr/bin/env python3
"""F3.1c: is the tall-tile defect in the STORE, or in the matmul's Acc result?

Established so far: `[M,K] @ [K,N]` with N < M returns garbage; with a position-encoded
input (`X = I`, `Y[i,j] = i*1000+j`) the [64,32] result is **not a permutation** of Y — every
element is undecodable and the max error is 9.6e10 against a reference bounded by 63031 —
while the [64,64] control is **bit-exact** (0.0). Garbage of that magnitude means data that
was never the product, i.e. an out-of-range read, not a misplaced write.

`TSTORE` and `TPUSH` cannot be told apart by reasoning: `pushAcc2GMFiFo` in
`npu/a2a3/TPush.hpp` calls the same `TSTORE_IMPL`, so both reach
`TStoreAccNz2nd` (`common/arch/memory/tstore_common.hpp:163`). This separates the store from
the matmul by removing the matmul:

  T1  load  [M,N] -> store [M,N]                 pure vector path, NO cube, NO Acc tile
  T2  load  [M,N] -> mul   -> store              same, with a vector op
  T3  matmul I @ Y -> store [M,N]                the known-bad case
  T4  matmul I @ Ypad -> store [M,M]             the known-good padded case

T1/T2 correct + T3 wrong  => the store of a tall tile is FINE; the Acc/matmul result is bad.
T1/T2 wrong               => storing a tall tile is broken irrespective of the cube.

Values are position-encoded throughout so wrong output can be read as coordinates rather
than as a norm.

Usage: python3 devtools/f31c_tstore_isolate.py <device> <platform> [M] [N]
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl


def _src_copy(M, N, with_op):
    """No cube at all: GM -> Vec -> GM, tile is [M, N] (tall when N < M)."""
    op = "        v2 = pl.mul(v, 2.0)\n        return pl.store(v2, [0, 0], Z)" if with_op \
        else "        return pl.store(v, [0, 0], Z)"
    return f'''
@pl.program
class TileCopy:
    @pl.function(type=pl.FunctionType.InCore)
    def cp(self, Y: pl.Tensor[[{M}, {N}], pl.FP32],
           Z: pl.Out[pl.Tensor[[{M}, {N}], pl.FP32]]) -> pl.Tensor[[{M}, {N}], pl.FP32]:
        v = pl.load(Y, [0, 0], [{M}, {N}])
{op}

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(self, Y: pl.Tensor[[{M}, {N}], pl.FP32],
             Z: pl.Out[pl.Tensor[[{M}, {N}], pl.FP32]]) -> pl.Tensor[[{M}, {N}], pl.FP32]:
        return self.cp(Y, Z)
'''


def _src_matmul(M, K, N):
    return f'''
@pl.program
class MM:
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
'''


def encoded(rows, cols):
    t = torch.zeros(rows, cols, dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            t[i, j] = i * 1000 + j
    return t


def report(name, Z, ref):
    err = (Z - ref).abs().max().item()
    ok = err < 1e-3
    print(f"  {name:<34} max abs err {err:<12.4e} {'ok' if ok else 'WRONG'}")
    if not ok:
        flat = Z.flatten()
        print(f"      first 8 returned: " + " ".join(f"{v:.4g}" for v in flat[:8].tolist()))
        print(f"      first 8 expected: " + " ".join(f"{v:.4g}" for v in ref.flatten()[:8].tolist()))
        print(f"      |Z| max {Z.abs().max().item():.4g}   |ref| max {ref.abs().max().item():.4g}")
        exact = (Z == ref).sum().item()
        print(f"      exactly-right elements: {exact} / {Z.numel()}")
    return ok


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2]
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    K = M

    from canary import BAD_CARD_EXIT, check
    if platform != "a2a3sim" and not check(device, platform):
        return BAD_CARD_EXIT

    from pypto import ir
    from pypto.runtime.runner import RunConfig
    cfg = lambda: RunConfig(platform=platform, device_id=device)  # noqa: E731

    print(f"\n=== M={M} K={K} N={N}  (tall: N<M is {N < M})   {platform} dev {device}")

    Y = encoded(M, N)
    Z = torch.zeros(M, N)
    ir.compile(pl.parse(_src_copy(M, N, False)), platform=platform)(Y, Z, config=cfg())
    report("T1 load->store, no cube", Z, Y)

    Z = torch.zeros(M, N)
    ir.compile(pl.parse(_src_copy(M, N, True)), platform=platform)(Y, Z, config=cfg())
    report("T2 load->mul->store, no cube", Z, Y * 2.0)

    X = torch.eye(M, K, dtype=torch.float32)
    Yk = encoded(K, N)
    Z = torch.zeros(M, N)
    ir.compile(pl.parse(_src_matmul(M, K, N)), platform=platform)(X, Yk, Z, config=cfg())
    report("T3 matmul I@Y -> store", Z, X @ Yk)

    Ypad = torch.zeros(K, M)
    Ypad[:, :N] = Yk
    Z = torch.zeros(M, M)
    ir.compile(pl.parse(_src_matmul(M, K, M)), platform=platform)(X, Ypad, Z, config=cfg())
    report("T4 matmul I@Ypad -> store [M,M]", Z[:, :N], (X @ Ypad)[:, :N])

    print("\nT1/T2 ok + T3 wrong => the tall STORE is fine; the cube/Acc result is the bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
