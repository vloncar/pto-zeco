"""F2 Phase 2 — does the corruption period track the cross-core ring depth?

The corruption injects every 4 iterations (first bad index always == 2 mod 4) and 4 is the
default bidirectional slot count that pypto picks:

    src/ir/transforms/utils/cross_core_pipe.cpp
      int GetSlotNumForDirMask(int dir_mask) {
        return dir_mask == (kDirMaskC2V | kDirMaskV2C) ? 4 : 8;
      }

pypto exposes an override — `pl.cross_core_slot(slot_num=N)` on a scope, or an InCore
function attr `slot_num` (read in expand_mixed_kernel_pass.cpp:1195) — which changes both the
reserved buffer size and the `slot_num` on `*_initialize_pipe`, i.e. the `TPipe<...>` ring depth.

If the corruption period follows slot_num, slot reuse in that ring is the mechanism. If it
stays at 4 or simply vanishes, it is not (or not only) ring depth.

Usage: python3 scratchpad/f2_slotnum.py <device> [C] [N_csv] [repeats]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl

C = 16          # overridden by argv[2]; the kernels read it as a closure variable
DK = DV = 16
TOL = 1e-2


def set_tile(c: int):
    """Set the chunk/tile size for subsequently built programs (slot_size = c*c*4 bytes)."""
    global C, DK, DV
    C = DK = DV = c


def _build_default(N: int):
    L = N * C
    NDK = N * DK

    @pl.program
    class SlotNumDefaultProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def chunk_scan(
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
            return self.chunk_scan(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return SlotNumDefaultProgram


def _build_s2(N: int):
    L = N * C
    NDK = N * DK

    @pl.program
    class SlotNumS2Program:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": 2})
        def chunk_scan(
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
            return self.chunk_scan(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return SlotNumS2Program


def _build_s8(N: int):
    L = N * C
    NDK = N * DK

    @pl.program
    class SlotNumS8Program:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": 8})
        def chunk_scan(
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
            return self.chunk_scan(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return SlotNumS8Program


def _build_s16(N: int):
    L = N * C
    NDK = N * DK

    @pl.program
    class SlotNumS16Program:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": 16})
        def chunk_scan(
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
            return self.chunk_scan(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return SlotNumS16Program


def _build_s32(N: int):
    L = N * C
    NDK = N * DK

    @pl.program
    class SlotNumS32Program:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": 32})
        def chunk_scan(
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
            return self.chunk_scan(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return SlotNumS32Program


BUILDERS = {"default": _build_default, "8": _build_s8, "16": _build_s16, "32": _build_s32}


def reference(A, K, V, N, tril, ones_cc, ones_cdv):
    s = torch.zeros(DK, DV)
    ref = []
    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        la = torch.log(A[lo:hi])
        b = torch.exp(tril @ la)
        g_row_full = torch.exp(ones_cc @ la)
        g_full = torch.exp(la.t() @ ones_cdv)
        kbar = (K[lo:hi] / b) * g_row_full
        s = s * g_full + kbar.t() @ V[lo:hi]
        ref.append(s.clone())
    return torch.stack(ref)


def run(N: int, variant: str, device: int, repeats: int, platform: str = "a2a3"):
    from pypto import ir
    from pypto.runtime.runner import RunConfig

    L = N * C
    torch.manual_seed(42)
    K = torch.randn(L, DK)
    V = torch.randn(L, DV)
    A = 0.9 + 0.1 * torch.sigmoid(torch.randn(L, DK))
    tril = torch.tril(torch.ones(C, C))
    ones_cc = torch.ones(C, C)
    ones_cdv = torch.ones(C, DV)
    ref = reference(A, K, V, N, tril, ones_cc, ones_cdv)

    compiled = ir.compile(BUILDERS[variant](N), platform=platform)
    cfg = RunConfig(platform=platform, device_id=device)
    for i in range(repeats):
        Sall = torch.zeros(N * DK, DV)
        compiled(A, K, V, tril, ones_cc, ones_cdv, torch.zeros(DK, DV), Sall, config=cfg)
        errs = [(Sall[n * DK:(n + 1) * DK] - ref[n]).abs().max().item() for n in range(N)]
        bad = [n for n, e in enumerate(errs) if e > TOL]
        print(f"[slot_num={variant:>10}] N={N:>3} run{i}: "
              f"{'CLEAN' if not bad else 'bad=' + str(bad)}  max={max(errs):.3e}", flush=True)


if __name__ == "__main__":
    device = int(sys.argv[1].split(",")[0])
    set_tile(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 16]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    print(f"=== C=dk=dv={C}  slot_size={C*C*4} B ===", flush=True)
    for variant in BUILDERS:
        for N in Ns:
            try:
                run(N, variant, device, R)
            except Exception as e:  # noqa: BLE001
                print(f"[slot_num={variant:>10}] N={N:>3} ERROR: {str(e)[:200]}", flush=True)
        print("", flush=True)
