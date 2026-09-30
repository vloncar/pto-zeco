#!/usr/bin/env python3
"""F3.1c: decode EXACTLY how a tall matmul's result is laid out wrong.

Pass/fail tables cannot say whether the data is garbage or merely misplaced. This makes the
corruption legible:

    X = I(M)                    (identity, so the product is Y itself)
    Y[i, j] = i * 1000 + j      (every element names its own coordinates)
    Z = X @ Y                   (must equal Y)

Then each returned value decodes to the (row, col) it was *supposed* to come from, so a
wrong result reads directly as a permutation — and the permutation is the bug's signature.
A pure stride error shows up as `Z[i][j] == Y[f(i,j)]` for some simple f; uninitialised or
truncated data shows up as values that decode to nothing.

Why this pins the layer: pypto's PTO IR and PTOAS's generated ISA are both correct for these
shapes (checked), so the suspect is pto-isa's Acc->GM path,
`TStoreAccNz2nd` in `include/pto/common/arch/memory/tstore_common.hpp`, which programs
`copy_matrix_cc_to_gm` with

    mSize     = validRow            (64)
    nSize     = validCol            (32)
    srcStride = TileData::Rows      (64)   <- same for [64,64] and [64,32]
    dstD      = gStride3

`srcStride` is what the hardware uses to walk the L0C fractal layout. If it should depend on
the tile's *column* count (the NZ fractal-column stride) rather than its row count, then a
square tile is accidentally correct and a tall one is not — which is exactly the observed
pattern. The permutation printed here says whether that is what is happening.

Usage: python3 devtools/f31c_tstore_pattern.py <device> <platform>
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl

CASES = [(64, 64, 32, "TALL — wrong"), (64, 64, 64, "square — control")]


def build(M, K, N):
    return pl.parse(f'''
@pl.program
class TStorePattern:
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


def decode(v):
    """value -> (row, col) it encodes, or None if it is not one of our encoded values."""
    if v != v or abs(v) > 1e8:
        return None
    iv = int(round(float(v)))
    if abs(float(v) - iv) > 1e-3:
        return None
    r, c = divmod(iv, 1000)
    return (r, c) if 0 <= r < 4096 and 0 <= c < 1000 else None


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2]

    from canary import BAD_CARD_EXIT, check
    if platform != "a2a3sim" and not check(device, platform):
        return BAD_CARD_EXIT

    from pypto import ir
    from pypto.runtime.runner import RunConfig

    for (M, K, N, note) in CASES:
        print(f"\n{'=' * 72}\n=== M={M} K={K} N={N}   {note}\n{'=' * 72}")
        X = torch.eye(M, K, dtype=torch.float32)
        Y = torch.zeros(K, N, dtype=torch.float32)
        for i in range(K):
            for j in range(N):
                Y[i, j] = i * 1000 + j
        ref = X @ Y

        Z = torch.zeros(M, N, dtype=torch.float32)
        ir.compile(build(M, K, N), platform=platform)(
            X, Y, Z, config=RunConfig(platform=platform, device_id=device))

        bad = (Z - ref).abs().max().item()
        print(f"max abs err {bad:.4e}")
        if bad == 0:
            print("exact — nothing to decode")
            continue

        # Where did each returned element actually come from?
        print("\nZ[i][j] decoded as the (row,col) of Y that the value belongs to:")
        print("  i\\j " + " ".join(f"{j:>9}" for j in range(min(N, 6))))
        for i in list(range(min(M, 8))) + ([M - 1] if M > 8 else []):
            cells = []
            for j in range(min(N, 6)):
                d = decode(Z[i, j].item())
                cells.append(f"{d[0]:>4},{d[1]:<4}" if d else f"{'?':>9}")
            print(f"  {i:>3} " + " ".join(cells))

        # Is it a consistent affine map on the flattened index?
        print("\n  (i,j) -> came from (r,c):")
        hits, misses = [], 0
        for i in range(M):
            for j in range(N):
                d = decode(Z[i, j].item())
                if d is None:
                    misses += 1
                elif d != (i, j):
                    hits.append(((i, j), d))
        print(f"    {len(hits)} misplaced, {misses} undecodable, "
              f"{M * N - len(hits) - misses} correct")
        for (src, dst) in hits[:8]:
            si, sj = src
            di, dj = dst
            print(f"    Z[{si},{sj}] holds Y[{di},{dj}]   "
                  f"(flat dst {si * N + sj:>5} <- flat src {di * N + dj:>5}, "
                  f"delta {di * N + dj - (si * N + sj):>6})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
