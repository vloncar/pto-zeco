#!/usr/bin/env python3
"""B4 pre-flight #2: does the DSL actually support the ops the backward kernels need?

The forward uses a narrow, well-trodden slice of pypto. The backward needs five things it
never touched, each of which would be expensive to debug inside a 500-line program:

  1. a `[DK,1]` -> `[1,DK]` shape flip (the forward only ever goes `[1,DK]` -> `[DK,1]`)
  2. `col_expand_add(tile[C,DK], row[1,DK])`   -- the whole-tile g_total correction
  3. `row_sum(tile, tmp)` inside an InCore chunk loop
  4. TWO loop carries, one of them a `[DK,1]` vector
  5. reverse chunk indexing, `off = (N-1-m)*C`, as a pl.range affine expression
  6. (dropped: `pl.tile.full([shape], dtype, value)` is not callable from the DSL —
     the parser binds it to the scalar overload and rejects the shape list)

Each is checked against a torch golden, so "it compiled" is not mistaken for "it works".

Usage: python3 devtools/b4_op_probe.py <platform> [device_id]
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

C, DK, DV, N = 16, 8, 16, 4


def build():
    @pl.program
    class ProbeProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def probe(
            self,
            X: pl.Tensor[[N * C, DK], pl.FP32],
            S: pl.Tensor[[N * DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            OutA: pl.Out[pl.Tensor[[N * C, DK], pl.FP32]],
            OutV: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
        ):
            # (6) `pl.tile.full([shape], dtype, value)` is NOT callable from the DSL — the
            # parser binds it to the scalar-valued overload and rejects the shape list. Loop
            # inits therefore come from host constant tensors, as the forward already does.
            acc0 = pl.load(zero, [0, 0], [DK, DV])
            vec0 = pl.load(onev, [0, 0], [DK, 1])
            # NB: `oa, ov = OutA, OutV` does NOT compile — a tuple assignment inside an
            # InCore kernel makes a tuple temp that PTO codegen cannot materialize
            # ("cannot materialize symbol '_tuple_tmp__ssa_v0'"). One name per line.
            oa = OutA
            ov = OutV
            # (4) two carries, one of them the [DK,1] vector.
            for m, (acc, vec) in pl.range(0, N, init_values=(acc0, vec0)):
                # (5) reverse indexing.
                off = (N - 1 - m) * C
                soff = (N - 1 - m) * DK
                x = pl.load(X, [off, 0], [C, DK])
                s = pl.load(S, [soff, 0], [DK, DV])
                # (3) row_sum with an explicit scratch tile -> [DK,1].
                tmp = pl.tile.create([DK, DV], pl.FP32)
                rs = pl.row_sum(pl.add(s, acc), tmp)
                # (1) the [DK,1] -> [1,DK] flip.
                rs_row = pl.tile.reshape(rs, [1, DK])
                # (2) whole-tile broadcast add of that row.
                oa = pl.store(pl.tile.col_expand_add(x, rs_row), [off, 0], oa)
                ov = pl.store(pl.mul(vec, rs), [soff, 0], ov)
                acc, vec = pl.yield_(pl.add(acc, s), pl.mul(vec, rs))
            return oa, ov

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            X: pl.Tensor[[N * C, DK], pl.FP32],
            S: pl.Tensor[[N * DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            OutA: pl.Out[pl.Tensor[[N * C, DK], pl.FP32]],
            OutV: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
        ):
            return self.probe(X, S, zero, onev, OutA, OutV)

    return ProbeProgram


def golden(X, S):
    OutA = torch.zeros(N * C, DK)
    OutV = torch.zeros(N * DK, 1)
    acc = torch.zeros(DK, DV)
    vec = torch.ones(DK, 1)
    for m in range(N):
        off, soff = (N - 1 - m) * C, (N - 1 - m) * DK
        s = S[soff:soff + DK]
        rs = (s + acc).sum(dim=1, keepdim=True)
        OutA[off:off + C] = X[off:off + C] + rs.reshape(1, DK)
        OutV[soff:soff + DK] = vec * rs
        acc = acc + s
        vec = vec * rs
    return OutA, OutV


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    torch.manual_seed(7)
    X = torch.randn(N * C, DK)
    S = torch.randn(N * DK, DV) * 0.1
    gA, gV = golden(X, S)

    compiled = ir.compile(build(), platform=platform)
    print("compiled OK", flush=True)
    OutA = torch.zeros(N * C, DK)
    OutV = torch.zeros(N * DK, 1)
    compiled(X, S, torch.zeros(DK, DV), torch.ones(DK, 1), OutA, OutV, config=RunConfig(platform=platform, device_id=device_id))

    eA = (OutA - gA).abs().max().item()
    eV = (OutV - gV).abs().max().item()
    print(f"col_expand_add + reshape[DK,1]->[1,DK] + reverse-index : max err {eA:.3e}")
    print(f"[DK,1] loop carry * row_sum                            : max err {eV:.3e}")
    ok = eA < 1e-3 and eV < 1e-3
    print("PROBE PASS" if ok else "PROBE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
