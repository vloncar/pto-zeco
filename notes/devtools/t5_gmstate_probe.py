#!/usr/bin/env python3
"""Task 5: can the [DK,DV] chunk state live in a scratch TENSOR instead of a carried tile?

Blocking the head dim works (``t5_block_probe.py``) but stops short of the value dim, because
the state is a ``[DK, DV]`` **loop-carried tile** and stays whole in vector memory however
finely the head dim is cut. At C=64, DK=DV=128 that is 65536 B of a 188416 B budget for the
carry alone, before the ring reservation and every working tile.

If the state lives in a scratch tensor instead, only a ``[BK, BV]`` slice is ever in vector
memory. That needs two things nobody here has relied on:

  1. a ``pl.store`` into a scratch tensor in chunk ``n`` being **visible** to a ``pl.load``
     from it in chunk ``n+1`` -- i.e. the loads and stores stay ordered when the tensor is
     threaded through nested ``pl.range`` loops rather than carried as a tile;
  2. read-then-write of the *same* rows within one block iteration keeping that order.

Both are checked against a torch golden, at several shapes, so "it compiled" is never mistaken
for "it works".

Usage: python3 devtools/t5_gmstate_probe.py <platform> [device] [NB] [C DK DV N]
"""
from __future__ import annotations

import os
import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

C, DK, DV, N = 16, 32, 16, 3
L = N * C
INNER_COLS = 16


def build(NB: int, slot: int = 4):
    BK = DK // NB
    SLOT = slot

    @pl.program
    class GmStateProbe:
        @pl.function(type=pl.FunctionType.InCore)
        def scan(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            Sws: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            pl.func_attr({"slot_num": 1})
            tril_t = pl.load(tril, [0, 0], [C, C])
            out = O
            # Seed the scratch state from the boundary. Threaded like `out`: a tensor
            # reassigned by pl.store carries its ordering without being an iter_arg.
            sws = pl.store(pl.load(Srecv, [0, 0], [DK, DV]), [0, 0], Sws)
            for n in pl.range(0, N):
                off = n * C
                v = pl.load(Vmat, [off, 0], [C, DV])
                sc0 = pl.mul(tril_t, 0.0)
                oi0 = pl.mul(v, 0.0)
                for j, (scores_acc, o_inter) in pl.range(0, NB, init_values=(sc0, oi0)):
                    dof = j * BK
                    q_b = pl.load(Q, [off, dof], [C, BK])
                    k_b = pl.load(Kmat, [off, dof], [C, BK])
                    a_b = pl.load(A, [off, dof], [C, BK])
                    la_b = pl.log(a_b)
                    b_b = pl.exp(pl.matmul(tril_t, la_b, out_dtype=pl.FP32))
                    gamma_b = pl.exp(pl.tile.reshape(pl.tile.col_sum(la_b), [BK, 1]))
                    qt_b = pl.mul(q_b, b_b)
                    kb_b = pl.div(k_b, b_b)
                    kbt_b = pl.transpose(kb_b, 0, 1)
                    # (1)/(2) read this block's state rows straight out of the scratch tensor
                    s_blk = pl.load(sws, [dof, 0], [BK, DV])
                    sc_n = pl.add(scores_acc, pl.matmul(qt_b, kbt_b, out_dtype=pl.FP32))
                    oi_n = pl.add(o_inter, pl.matmul(qt_b, s_blk, out_dtype=pl.FP32))
                    kv_b = pl.matmul(kbt_b, v, out_dtype=pl.FP32)
                    s_blk_new = pl.tile.row_expand_mul(pl.add(s_blk, kv_b), gamma_b)
                    # ... and write them back, after the read above.
                    sws = pl.store(s_blk_new, [dof, 0], sws)
                    scores_acc, o_inter = pl.yield_(sc_n, oi_n)
                scores = pl.mul(scores_acc, tril_t)
                o_n = pl.add(o_inter, pl.matmul(scores, v, out_dtype=pl.FP32))
                out = pl.store(o_n, [off, 0], out)
            return out, sws

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            Sws: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            return self.scan(Q, Kmat, Vmat, A, tril, Srecv, O, Sws)

    return GmStateProbe


def golden(q, k, v, a, tril, S):
    O = torch.zeros(L, DV)
    S = S.clone()
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        qt, kb = q[s] * b, k[s] / b
        O[s] = qt @ S + ((qt @ kb.T) * tril) @ v[s]
        S = gamma * (S + kb.T @ v[s])
    return O, S


def main():
    global C, DK, DV, N, L
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    NB = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    slot = int(os.environ.get("T5_SLOT", "4"))
    if len(sys.argv) > 4:
        C, DK, DV, N = (int(x) for x in sys.argv[4:8])
        L = N * C
    if DK % NB or (DK // NB) % INNER_COLS:
        print(f"SKIP C={C} DK={DK} DV={DV} NB={NB}: block width not a multiple of {INNER_COLS}")
        return 0
    torch.manual_seed(5)
    q = torch.randn(L, DK)
    k = torch.randn(L, DK)
    v = torch.randn(L, DV)
    a = torch.rand(L, DK) * 0.4 + 0.6
    tril = torch.tril(torch.ones(C, C))
    S0 = torch.randn(DK, DV) * 0.1
    gO, gS = golden(q, k, v, a, tril, S0)

    print(f"building C={C} DK={DK} DV={DV} N={N} NB={NB} (BK={DK // NB}) slot={slot}", flush=True)
    compiled = ir.compile(build(NB, slot), platform=platform)
    print("compiled OK", flush=True)
    O = torch.zeros(L, DV)
    Sws = torch.zeros(DK, DV)
    compiled(q, k, v, a, tril, S0, O, Sws,
             config=RunConfig(platform=platform, device_id=device_id))
    eO = (O - gO).abs().max().item()
    eS = (Sws - gS).abs().max().item()
    print(f"max|O - golden| = {eO:.3e}")
    print(f"max|S - golden| = {eS:.3e}")
    ok = eO < 1e-3 and eS < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
