#!/usr/bin/env python3
"""A1 stage1: emit the per-chunk snapshot + running decay, and stop carrying [dk, dv].

stage2 can drop its state carry only if stage1 hands it `S_n(0)` and `G_n` per chunk
(`t5_snapshot_math.py`, `t5_nocarry_probe.py`). But stage1 itself still carries the full
[dk, dv] state, so at dk=dv=128 it becomes the new ceiling -- three copies are 196608 B of a
188416 B buffer.

Nothing in stage1 contracts over the head dim (`kb^T @ v` has it as the OUTPUT row dim), so
the head-dim blocks are independent for the WHOLE scan, not just within a chunk. Swap the
loops -- block outer, chunk inner -- and the live carry is [BK, DV] regardless of dk. The
price is re-reading V once per block; V is staged in L1, and it is what makes dk arbitrary.

Usage: python3 devtools/a1_stage1_probe.py <platform> [device] [NB] [C dk dv N]
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
    class Stage1Probe:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": slot})
        def stage1(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            ones_k: pl.Tensor[[DK, 1], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Gsnap: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            tril_t = pl.load(tril, [0, 0], [C, C])
            snaps = Ssnap
            decays = Gsnap
            tot = Stot
            # Head-dim block OUTSIDE the chunk scan: each block's state is independent of
            # every other block's, for the whole slice.
            for j in pl.range(0, NB):
                dof = j * BK
                s0 = pl.load(zero, [dof, 0], [BK, DV])
                g0 = pl.load(ones_k, [dof, 0], [BK, 1])
                for n, (s_run, g_run) in pl.range(0, N, init_values=(s0, g0)):
                    off = n * C
                    soff = n * DK
                    v = pl.load(Vmat, [off, 0], [C, DV], target_memory=pl.MemorySpace.Mat)
                    # Snapshot the state BEFORE this chunk's update -- that is exactly the
                    # `S_n(0)` stage2 needs.
                    snaps = pl.store(s_run, [soff + dof, 0], snaps)
                    decays = pl.store(g_run, [soff + dof, 0], decays)
                    k_b = pl.load(Kmat, [off, dof], [C, BK])
                    a_b = pl.load(A, [off, dof], [C, BK])
                    la_b = pl.log(a_b)
                    b_b = pl.exp(pl.matmul(tril_t, la_b, out_dtype=pl.FP32))
                    gamma_b = pl.exp(pl.tile.reshape(pl.tile.col_sum(la_b), [BK, 1]))
                    kb_b = pl.div(k_b, b_b)
                    kv_b = pl.matmul(pl.transpose(kb_b, 0, 1), v, out_dtype=pl.FP32)
                    s_new = pl.tile.row_expand_mul(pl.add(s_run, kv_b), gamma_b)
                    g_new = pl.mul(g_run, gamma_b)
                    s_fin, g_fin = pl.yield_(s_new, g_new)
                tot = pl.store(s_fin, [dof, 0], tot)
            return (snaps, decays, tot)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            ones_k: pl.Tensor[[DK, 1], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Gsnap: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.stage1(A, Kmat, Vmat, tril, zero, ones_k, Ssnap, Gsnap, Stot)

    return Stage1Probe


def golden(k, v, a, tril):
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
    return snaps, decays, S


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
    torch.manual_seed(7)
    k = torch.randn(L, DK)
    v = torch.randn(L, DV)
    a = torch.rand(L, DK) * 0.4 + 0.6
    tril = torch.tril(torch.ones(C, C))
    g_snaps, g_decays, g_tot = golden(k, v, a, tril)

    print(f"building C={C} dk={DK} dv={DV} N={N} NB={NB} (BK={DK // NB})", flush=True)
    compiled = ir.compile(build(NB), platform=platform)
    print("compiled OK", flush=True)
    snaps = torch.zeros(N * DK, DV)
    decays = torch.zeros(N * DK, 1)
    tot = torch.zeros(DK, DV)
    compiled(a, k, v, tril, torch.zeros(DK, DV), torch.ones(DK, 1), snaps, decays, tot,
             config=RunConfig(platform=platform, device_id=device_id))
    es = (snaps - g_snaps).abs().max().item()
    ed = (decays - g_decays).abs().max().item()
    et = (tot - g_tot).abs().max().item()
    print(f"max|snap diff| = {es:.3e}   max|decay diff| = {ed:.3e}   max|Stot diff| = {et:.3e}")
    ok = max(es, ed, et) < 1e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
