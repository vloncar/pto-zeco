#!/usr/bin/env python3
"""Does a CARRY-FREE stage2 fit -- and build -- at dk=dv=128?

Three copies of the [dk,dv] state are 60% of the vector budget and head-dim blocking touches
none of them, because the state is a loop carry. `t5_snapshot_math.py` proves the carry is
removable: the recurrence is linear, so

    S_n(boundary) = G_n * boundary + S_n(0)

and stage1 already walks the chunks from zero. If it stores its per-chunk snapshot and running
decay, stage2 rebuilds the state it needs per chunk instead of carrying it -- which is exactly
what pto-kernels' KDA chunk_o does ("each reads its own s_snapshots entry").

Then the state is READ-ONLY per chunk, so it blocks over the head dim like everything else:
[BK, DV] live instead of 3 x [dk, dv]. This probe checks that claim end to end -- footprint,
device build, and numbers against a torch golden.

Usage: python3 devtools/t5_nocarry_probe.py <platform> [device] [NB] [C dk dv N]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

C, DK, DV, N = 64, 128, 128, 3
L = N * C
INNER_COLS = 16


def build(NB: int, slot: int = 1):
    BK = DK // NB

    @pl.program
    class NoCarryProbe:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": slot})
        def scan(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Bnd: pl.Tensor[[DK, DV], pl.FP32],       # the boundary state
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],  # stage1's per-chunk snapshot, from zero
            Gsnap: pl.Tensor[[N * DK, 1], pl.FP32],   # stage1's running decay per chunk
            zc: pl.Tensor[[C, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            tril_t = pl.load(tril, [0, 0], [C, C])
            out = O
            # No pl.range init_values anywhere for the state: there is no carry to keep.
            for n in pl.range(0, N):
                off = n * C
                soff = n * DK
                v = pl.load(Vmat, [off, 0], [C, DV], target_memory=pl.MemorySpace.Mat)
                sc0 = pl.mul(tril_t, 0.0)
                oi0 = pl.load(zc, [0, 0], [C, DV])
                for j, (scores_acc, o_inter) in pl.range(0, NB, init_values=(sc0, oi0)):
                    dof = j * BK
                    q_b = pl.load(Q, [off, dof], [C, BK])
                    k_b = pl.load(Kmat, [off, dof], [C, BK])
                    a_b = pl.load(A, [off, dof], [C, BK])
                    la_b = pl.log(a_b)
                    b_b = pl.exp(pl.matmul(tril_t, la_b, out_dtype=pl.FP32))
                    qt_b = pl.mul(q_b, b_b)
                    kb_b = pl.div(k_b, b_b)
                    kbt_b = pl.transpose(kb_b, 0, 1)
                    # Rebuild THIS block of the state: G_n * boundary + snapshot. Only
                    # [BK, DV] is ever live -- the whole point.
                    s_snap = pl.load(Ssnap, [soff + dof, 0], [BK, DV])
                    g_blk = pl.load(Gsnap, [soff + dof, 0], [BK, 1])
                    b_blk = pl.load(Bnd, [dof, 0], [BK, DV])
                    s_blk = pl.add(pl.tile.row_expand_mul(b_blk, g_blk), s_snap)
                    sc_n = pl.add(scores_acc, pl.matmul(qt_b, kbt_b, out_dtype=pl.FP32))
                    oi_n = pl.add(o_inter, pl.matmul(qt_b, s_blk, out_dtype=pl.FP32))
                    scores_acc, o_inter = pl.yield_(sc_n, oi_n)
                scores = pl.mul(scores_acc, tril_t)
                o_n = pl.add(o_inter, pl.matmul(scores, v, out_dtype=pl.FP32))
                out = pl.store(o_n, [off, 0], out)
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
            zc: pl.Tensor[[C, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            return self.scan(Q, Kmat, Vmat, A, tril, Bnd, Ssnap, Gsnap, zc, O)

    return NoCarryProbe


def host_side(k, v, a, tril):
    """What stage1 would produce: per-chunk snapshot from zero, and running decay."""
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
    NB = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    if len(sys.argv) > 4:
        C, DK, DV, N = (int(x) for x in sys.argv[4:8])
        L = N * C
    if DK % NB or (DK // NB) % INNER_COLS:
        print(f"SKIP C={C} dk={DK} dv={DV} NB={NB}: block width not a multiple of {INNER_COLS}")
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

    print(f"building C={C} dk={DK} dv={DV} N={N} NB={NB} (BK={DK // NB})", flush=True)
    compiled = ir.compile(build(NB), platform=platform)
    print("compiled OK", flush=True)
    O = torch.zeros(L, DV)
    compiled(q, k, v, a, tril, B, snaps, decays, torch.zeros(C, DV), O,
             config=RunConfig(platform=platform, device_id=device_id))
    e = (O - gO).abs().max().item()
    print(f"max|O - golden| = {e:.3e}")
    print("PASS" if e < 1e-2 else "FAIL")
    return 0 if e < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
