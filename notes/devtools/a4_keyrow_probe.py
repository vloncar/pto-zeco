#!/usr/bin/env python3
"""A4: block the two [C,C] matmuls over their KEY-ROW (contraction) axis.

`C=128` misses by 256 B of vector buffer, and a third of that buffer is the cross-core ring
reserve, sized by the largest tile crossing cube<->vector -- the [C,C] score matmul result.
Results are fp32 by construction, so narrowing cannot shrink it (see
allscan/issues/pypto-c128-wall/). Blocking the contraction can, exactly.

Two decompositions, both exact:

  b       = tril @ la   = sum_r  tril[:, r] @ la[r, :]        (work-neutral: same MACs)
  o_intra = (sum_j X_j) (*) tril @ v
          = sum_j sum_r ( X_j[:, r] (*) tril[:, r] ) @ v[r, :]

The second uses the fact that MASKING IS LINEAR, so it distributes over the head-block sum.
That is what removes the [C,C] score matrix entirely: the output is accumulated directly and
no full score matrix is ever built. Its price is doing the within-chunk output matmul once per
(head block, key-row block) instead of once per chunk.

Usage: a4_keyrow_probe.py <platform> [device] [NB] [NC] [C dk dv N]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

C, DK, DV, N = 128, 128, 128, 3
L = N * C
UNIT = 16


def build(NB: int, NC: int, NV: int = 1, slot: int = 1):
    BK, BC, BV = DK // NB, C // NC, DV // NV

    @pl.program
    class KeyRowProbe:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": slot})
        def scan(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Bnd: pl.Tensor[[DK, DV], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Gsnap: pl.Tensor[[N * DK, 1], pl.FP32],
            zc: pl.Tensor[[C, BV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            out = O
            for n in pl.range(0, N):
                off = n * C
                soff = n * DK
              # value blocking, as in the shipped program
                for w in pl.range(0, NV):
                 vof = w * BV
                 oi0 = pl.load(zc, [0, 0], [C, BV])
                # Head blocks OUTSIDE the key-row blocks, so the within-chunk decay -- which
                # every query row needs in full -- is computed exactly once per head block.
                 for j, (o_acc,) in pl.range(0, NB, init_values=(oi0,)):
                     dof = j * BK
                     q_b = pl.load(Q, [off, dof], [C, BK])
                     k_b = pl.load(Kmat, [off, dof], [C, BK])
                     # Decay, contraction-blocked. tril[:, r] already carries the causal zeros,
                     # so each block is an independent product and no scan is needed. Same
                     # total MACs as the single [C,C] @ [C,BK] product it replaces.
                     z_b = pl.mul(q_b, 0.0)
                     for r, (b_acc,) in pl.range(0, NC, init_values=(z_b,)):
                         rof = r * BC
                         tril_r = pl.load(tril, [0, rof], [C, BC])
                         la_r = pl.log(pl.load(A, [off + rof, dof], [BC, BK]))
                         b_fin = pl.yield_(
                             pl.add(b_acc, pl.matmul(tril_r, la_r, out_dtype=pl.FP32)))
                     b_b = pl.exp(b_fin)
                     qt_b = pl.mul(q_b, b_b)
                     kbt_b = pl.transpose(pl.div(k_b, b_b), 0, 1)
                     # State term: independent of key rows, so it sits outside the r loop.
                     s_snap = pl.load(Ssnap, [soff + dof, vof], [BK, BV])
                     g_blk = pl.load(Gsnap, [soff + dof, 0], [BK, 1])
                     b_blk = pl.load(Bnd, [dof, vof], [BK, BV])
                     s_blk = pl.add(pl.tile.row_expand_mul(b_blk, g_blk), s_snap)
                     o_st = pl.add(o_acc, pl.matmul(qt_b, s_blk, out_dtype=pl.FP32))
                     # Within-chunk term, accumulated straight into the output. The widest tile
                     # the cube sees here is [C, BC], never [C, C].
                     for r2, (o_in,) in pl.range(0, NC, init_values=(o_st,)):
                         rof2 = r2 * BC
                         tril_2 = pl.load(tril, [0, rof2], [C, BC])
                         v_r = pl.load(Vmat, [off + rof2, vof], [BC, BV],
                                       target_memory=pl.MemorySpace.Mat)
                         kbt_r = pl.tile.slice(kbt_b, [BK, BC], [0, rof2])
                         x = pl.mul(pl.matmul(qt_b, kbt_r, out_dtype=pl.FP32), tril_2)
                         o_fin = pl.yield_(
                             pl.add(o_in, pl.matmul(x, v_r, out_dtype=pl.FP32)))
                     o_acc = pl.yield_(o_fin)
                 out = pl.store(o_acc, [off, vof], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Bnd: pl.Tensor[[DK, DV], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Gsnap: pl.Tensor[[N * DK, 1], pl.FP32],
            zc: pl.Tensor[[C, BV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            return self.scan(Q, Kmat, Vmat, A, tril, Bnd, Ssnap, Gsnap, zc, O)

    return KeyRowProbe


def host_side(k, v, a, tril):
    S = torch.zeros(DK, DV)
    G = torch.ones(DK, 1)
    snaps = torch.zeros(N * DK, DV)
    decays = torch.zeros(N * DK, 1)
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        snaps[n * DK:(n + 1) * DK] = S
        decays[n * DK:(n + 1) * DK] = G
        S = gamma * (S + (k[s] / b).T @ v[s])
        G = G * gamma
    return snaps, decays


def golden(q, k, v, a, tril, B):
    O = torch.zeros(L, DV)
    S = B.clone()
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        qt, kb = q[s] * b, k[s] / b
        O[s] = qt @ S + ((qt @ kb.T) * tril) @ v[s]
        S = gamma * (S + kb.T @ v[s])
    return O


def main():
    global C, DK, DV, N, L
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    NB = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    NC = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    NV = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    if len(sys.argv) > 6:
        C, DK, DV, N = (int(x) for x in sys.argv[6:10])
        L = N * C
    if (DK % NB or (DK // NB) % UNIT or C % NC or (C // NC) % UNIT
            or DV % NV or (DV // NV) % UNIT):
        print(f"SKIP C={C} dk={DK} dv={DV} NB={NB} NC={NC} NV={NV}: block not a multiple of {UNIT}")
        return 0
    torch.manual_seed(5)
    q = torch.randn(L, DK)
    k = torch.randn(L, DK)
    v = torch.randn(L, DV)
    a = torch.rand(L, DK) * 0.4 + 0.6
    tril = torch.tril(torch.ones(C, C))
    B = torch.randn(DK, DV) * 0.1
    snaps, decays = host_side(k, v, a, tril)
    gO = golden(q, k, v, a, tril, B)

    print(f"building C={C} dk={DK} dv={DV} N={N} NB={NB} NC={NC} "
          f"NV={NV} (BK={DK // NB} BC={C // NC} BV={DV // NV})", flush=True)
    compiled = ir.compile(build(NB, NC, NV), platform=platform)
    print("compiled OK", flush=True)
    O = torch.zeros(L, DV)
    compiled(q, k, v, a, tril, B, snaps, decays, torch.zeros(C, DV // NV), O,
             config=RunConfig(platform=platform, device_id=device_id))
    e = (O - gO).abs().max().item()
    print(f"max|O - golden| = {e:.3e}")
    print("PASS" if e < 1e-2 else "FAIL")
    return 0 if e < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
