"""F2 Phase 2 — isolate stage1: does the miscompile survive with NO distribution at all?

The stage-split diagnostic (f2_diag.py) showed, at P=2/N=8/C=16:
    err_S_local = 3.16 (rank 0) / 2.37 (rank 1)   <- stage1's OWN output is wrong
    err_S_recv  = 3.16 == rank 0's stage1 error, ring copy bitwise-identical
    err_O_mixed = 1e-5 at every chunk             <- stage2 is CORRECT given its input
So stage1 alone produces the wrong value and everything downstream faithfully propagates it.

Phase 0's "P=1 passes at every N" does NOT contradict that: the P=1 fused program never runs
stage1 (with no ring, S_total is dead and pypto DCEs it) — so P=1 was never evidence that
stage1 is fine, and F2b may need no distributed context whatsoever.

This script runs stage1 as a plain SINGLE-DEVICE program (no ring, no P, no host orchestrator)
and compares against gla_chunk_scan:
  * variant "final" — the shipping pattern: post-loop `s_fin = pl.yield_(s_new); store(s_fin)`.
  * variant "trace" — stores s_new EVERY iteration into Sall[n], revealing the first chunk at
    which the running state diverges.

Usage: python3 scratchpad/f2_stage1_only.py <device> [C] [N_csv]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl
from gla.common import gla_chunk_scan, make_gla_inputs


def build_stage1_final(L: int, C: int, dk: int, dv: int):
    """Shipping stage1: one post-loop read of the carry."""
    N = L // C
    DK, DV = dk, dv

    @pl.program
    class Stage1FinalProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage1(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> pl.Tensor[[DK, DV], pl.FP32]:
            tril_t = pl.load(tril, [0, 0], [C, C])
            ones_cc_t = pl.load(ones_cc, [0, 0], [C, C])
            ones_cdv_t = pl.load(ones_cdv, [0, 0], [C, DV])
            s_init = pl.load(zero, [0, 0], [DK, DV])
            for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
                off = n * C
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                g_row_full = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
                g_full = pl.exp(pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
                kb = pl.div(k, b)
                kbar = pl.mul(kb, g_row_full)
                kv = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
                s_scaled = pl.mul(s_run, g_full)
                s_new = pl.add(s_scaled, kv)
                s_fin = pl.yield_(s_new)
            return pl.store(s_fin, [0, 0], Stot)

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> pl.Tensor[[DK, DV], pl.FP32]:
            return self.gla_stage1(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Stot)

    return Stage1FinalProgram


def build_stage1_trace(L: int, C: int, dk: int, dv: int):
    """Same recurrence, but the state after every chunk is stored to Sall[n]."""
    N = L // C
    DK, DV = dk, dv
    NDK = N * dk

    @pl.program
    class Stage1TraceProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage1_trace(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            Sall: pl.Out[pl.Tensor[[NDK, DV], pl.FP32]],
        ) -> pl.Tensor[[NDK, DV], pl.FP32]:
            tril_t = pl.load(tril, [0, 0], [C, C])
            ones_cc_t = pl.load(ones_cc, [0, 0], [C, C])
            ones_cdv_t = pl.load(ones_cdv, [0, 0], [C, DV])
            s_init = pl.load(zero, [0, 0], [DK, DV])
            out = Sall
            for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
                off = n * C
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                g_row_full = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
                g_full = pl.exp(pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
                kb = pl.div(k, b)
                kbar = pl.mul(kb, g_row_full)
                kv = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
                s_scaled = pl.mul(s_run, g_full)
                s_new = pl.add(s_scaled, kv)
                out = pl.store(s_new, [n * DK, 0], out)
                s_run = pl.yield_(s_new)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            Sall: pl.Out[pl.Tensor[[NDK, DV], pl.FP32]],
        ) -> pl.Tensor[[NDK, DV], pl.FP32]:
            return self.gla_stage1_trace(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return Stage1TraceProgram


def reference_states(C: int, N: int, dk: int, dv: int):
    """Per-chunk end states from the torch reference: ref[n] = state after chunk n."""
    L = N * C
    Q, K, V, A = make_gla_inputs(1, L, dk, dv)
    S_prev, _, _, S_total, _ = gla_chunk_scan(Q[0], K[0], V[0], A[0], C)
    ref = [S_prev[n + 1] for n in range(N - 1)] + [S_total]
    return Q, K, V, A, torch.stack(ref)


def run_one(N: int, C: int, device: int, platform: str = "a2a3"):
    from pypto import ir
    from pypto.runtime.runner import RunConfig

    dk = dv = C
    L = N * C
    Q, K, V, A, ref = reference_states(C, N, dk, dv)
    tril = torch.tril(torch.ones(C, C))
    ones_cc = torch.ones(C, C)
    ones_cdv = torch.ones(C, dv)
    zero = torch.zeros(dk, dv)
    cfg = RunConfig(platform=platform, device_id=device)

    compiled = ir.compile(build_stage1_final(L, C, dk, dv), platform=platform)
    Stot = torch.zeros(dk, dv)
    compiled(A[0], K[0], V[0], tril, ones_cc, ones_cdv, zero, Stot, config=cfg)
    err_final = (Stot - ref[-1]).abs().max().item()

    compiled_t = ir.compile(build_stage1_trace(L, C, dk, dv), platform=platform)
    Sall = torch.zeros(N * dk, dv)
    compiled_t(A[0], K[0], V[0], tril, ones_cc, ones_cdv, zero, Sall, config=cfg)
    per_chunk = [(Sall[n * dk:(n + 1) * dk] - ref[n]).abs().max().item() for n in range(N)]

    return err_final, per_chunk


if __name__ == "__main__":
    device = int(sys.argv[1].split(",")[0])
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2, 4, 8, 16]

    print(f"=== stage1 STANDALONE (single device, no distribution) C=dk=dv={C} ===")
    bad = 0
    for N in Ns:
        try:
            ef, pc = run_one(N, C, device)
            ok = ef < 1e-2
            bad += 0 if ok else 1
            print(f"N={N:>3}  final_err={ef:.3e}  {'PASS' if ok else '*** FAIL ***'}")
            print(f"      per-chunk err (trace variant): " + " ".join(f"{e:.2e}" for e in pc))
        except Exception as e:  # noqa: BLE001 - keep sweeping
            bad += 1
            print(f"N={N:>3}  ERROR: {str(e)[:200]}")
    sys.exit(1 if bad else 0)
