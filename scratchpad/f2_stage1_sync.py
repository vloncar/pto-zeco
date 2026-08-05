"""F2 Phase 2 — is the corruption fixed by an explicit barrier in the loop body?

Established so far (stage1 standalone, one device, no distribution, C=dk=dv=16):
  * N=2/4/6 clean; N=5/7/8/16 corrupted, always first at a chunk index == 2 (mod 4);
  * 4 is the cube<->vector TPipe slot depth (`TPipe<0, DIR_BOTH, 1024, 4, 4, true>`);
  * generated device code for N=4 (clean) and N=8 (corrupted) is byte-identical except the
    loop trip-count constant -> not a structural codegen difference;
  * the SAME binary with the SAME inputs differs across dispatches (N=5: run0 clean,
    run1/run2 corrupted) -> a RACE.

If a barrier at the end of the loop body removes the corruption, the defect is a missing
synchronisation in the emitted cube<->vector pipeline (7 TPUSH + 4 TPOP per iteration through
a 4-slot TPipe), not a loop-carry value bug.

NOTE: the three variants are spelled out as separate classes on purpose. A Python-level
`if variant == ...` inside a `@pl.function` body is parsed as DSL control flow and fails with
"For loop has 1 iteration arguments but 0 return variables".

Usage: python3 scratchpad/f2_stage1_sync.py <device> [C] [N_csv] [repeats]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl
from f2_stage1_only import reference_states

TOL = 1e-2


def build_plain(L: int, C: int, dk: int, dv: int):
    N = L // C
    DK, DV = dk, dv
    NDK = N * dk

    @pl.program
    class Stage1PlainProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage1_sync(
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
            return self.gla_stage1_sync(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return Stage1PlainProgram


def build_fence(L: int, C: int, dk: int, dv: int):
    N = L // C
    DK, DV = dk, dv
    NDK = N * dk

    @pl.program
    class Stage1FenceProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage1_sync(
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
                pl.system.fence()
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
            return self.gla_stage1_sync(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return Stage1FenceProgram


def build_barall(L: int, C: int, dk: int, dv: int):
    N = L // C
    DK, DV = dk, dv
    NDK = N * dk

    @pl.program
    class Stage1BarAllProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage1_sync(
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
                pl.system.bar_all()
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
            return self.gla_stage1_sync(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return Stage1BarAllProgram


BUILDERS = {"plain": build_plain, "fence": build_fence, "barall": build_barall}


def run(variant: str, N: int, C: int, device: int, repeats: int, platform: str = "a2a3"):
    from pypto import ir
    from pypto.runtime.runner import RunConfig
    from gla.common import make_gla_inputs

    dk = dv = C
    L = N * C
    _, _, _, _, ref = reference_states(C, N, dk, dv)
    _, K, V, A = make_gla_inputs(1, L, dk, dv)
    cfg = RunConfig(platform=platform, device_id=device)
    compiled = ir.compile(BUILDERS[variant](L, C, dk, dv), platform=platform)

    out = []
    for _ in range(repeats):
        Sall = torch.zeros(N * dk, dv)
        compiled(A[0], K[0], V[0], torch.tril(torch.ones(C, C)), torch.ones(C, C),
                 torch.ones(C, dv), torch.zeros(dk, dv), Sall, config=cfg)
        errs = [(Sall[n * dk:(n + 1) * dk] - ref[n]).abs().max().item() for n in range(N)]
        out.append(errs)
    return out


if __name__ == "__main__":
    device = int(sys.argv[1].split(",")[0])
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 16]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    for variant in BUILDERS:
        for N in Ns:
            try:
                runs = run(variant, N, C, device, R)
                for i, errs in enumerate(runs):
                    bad = [n for n, e in enumerate(errs) if e > TOL]
                    print(f"[{variant:6s}] N={N:>3} run{i}: "
                          f"{'CLEAN' if not bad else 'bad=' + str(bad)}  max={max(errs):.3e}",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{variant:6s}] N={N:>3} ERROR: {str(e)[:180]}", flush=True)
        print("", flush=True)
