#!/usr/bin/env python3
"""B4 debug: check each backward kernel SEPARATELY against its torch emulation.

Why this exists: the P=1 suite passing is much weaker evidence than it looks. At P=1 the
boundary values are all zero, and that zeroes most of the program:

  * `grad_h`'s ENTIRE state path is multiplied by `dSloc`/`dcvec` == 0 — `dV_h`, `dK_h`,
    `dg_cs_h` and all three `dgamma` correction terms vanish, so `b`, `gamma`, `S_prev` and
    `c_prev` are never really exercised there;
  * `grad_o`'s `dS_recv` accumulator and `dc_prev` are computed but DEAD (nothing reads them);
  * `recompute`'s `S_total` is DEAD (it only feeds the ring).

So a P=1 pass leaves `S_total`, `dS_recv`, `dc_prev` and the whole of `grad_h`'s recurrence
unverified. This runs the three kernels as a standalone P=1 program with every input and
output exposed, fed **artificially non-zero** `S_recv` / `dS_total` / `dgamma`, and diffs each
output against `b4_math_check`'s emulation. Everything except the two rings is then covered —
on the simulator, with no card.

Usage: python3 devtools/b4_kernel_probe.py [platform]
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

sys.path.insert(0, "/root/workspace/allscan")
from devtools.b4_math_check import k_grad_h, k_grad_o, k_recompute  # noqa: E402

L, C, DK, DV = 64, 16, 16, 16
N = L // C


def build():
    """The three kernels, each with every buffer as a real parameter."""

    @pl.program
    class KernelProbe:
        @pl.function(type=pl.FunctionType.InCore)
        def recompute(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Cprev: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            tril_t = pl.load(tril, [0, 0], [C, C])
            s_init = pl.load(zero, [0, 0], [DK, DV])
            c_init = pl.load(onev, [0, 0], [DK, 1])
            snap = Ssnap
            cp = Cprev
            for n, (s_run, c_run) in pl.range(0, N, init_values=(s_init, c_init)):
                off = n * C
                soff = n * DK
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                gamma = pl.exp(pl.tile.reshape(pl.tile.col_sum(la), [DK, 1]))
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                snap = pl.store(s_run, [soff, 0], snap)
                cp = pl.store(c_run, [soff, 0], cp)
                kb = pl.div(k, b)
                kv = pl.matmul(pl.transpose(kb, 0, 1), v, out_dtype=pl.FP32)
                s_new = pl.tile.row_expand_mul(pl.add(s_run, kv), gamma)
                c_new = pl.mul(c_run, gamma)
                s_fin, c_fin = pl.yield_(s_new, c_new)
            return snap, cp, pl.store(s_fin, [0, 0], Stot)

        @pl.function(type=pl.FunctionType.InCore)
        def grad_o(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            dOmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            dQ: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dKo: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dVo: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dgcso: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dH: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCp: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            dSrecv: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            tril_t = pl.load(tril, [0, 0], [C, C])
            srecv_t = pl.load(Srecv, [0, 0], [DK, DV])
            acc0 = pl.load(zero, [0, 0], [DK, DV])
            oq = dQ
            ok = dKo
            ov = dVo
            og = dgcso
            oh = dH
            oc = dCp
            for n, (acc,) in pl.range(0, N, init_values=(acc0,)):
                off = n * C
                soff = n * DK
                q = pl.load(Q, [off, 0], [C, DK])
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                do = pl.load(dOmat, [off, 0], [C, DV])
                la = pl.log(a)
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                qt = pl.mul(q, b)
                kb = pl.div(k, b)
                scores = pl.mul(pl.matmul(qt, pl.transpose(kb, 0, 1), out_dtype=pl.FP32), tril_t)
                sprev = pl.load(Ssnap, [soff, 0], [DK, DV])
                cprev = pl.load(Cprev, [soff, 0], [DK, 1])
                hmat = pl.add(sprev, pl.tile.row_expand_mul(srecv_t, cprev))
                dqt = pl.matmul(do, pl.transpose(hmat, 0, 1), out_dtype=pl.FP32)
                dh_n = pl.matmul(pl.transpose(qt, 0, 1), do, out_dtype=pl.FP32)
                oh = pl.store(dh_n, [soff, 0], oh)
                tmp = pl.tile.create([DK, DV], pl.FP32)
                oc = pl.store(pl.row_sum(pl.mul(dh_n, srecv_t), tmp), [soff, 0], oc)
                acc_n = pl.add(acc, pl.tile.row_expand_mul(dh_n, cprev))
                dsc = pl.mul(pl.matmul(do, pl.transpose(v, 0, 1), out_dtype=pl.FP32), tril_t)
                ov = pl.store(
                    pl.matmul(pl.transpose(scores, 0, 1), do, out_dtype=pl.FP32), [off, 0], ov)
                dqt2 = pl.add(dqt, pl.matmul(dsc, kb, out_dtype=pl.FP32))
                dkin = pl.matmul(pl.transpose(dsc, 0, 1), qt, out_dtype=pl.FP32)
                dq_n = pl.mul(dqt2, b)
                dko_n = pl.div(dkin, b)
                oq = pl.store(dq_n, [off, 0], oq)
                ok = pl.store(dko_n, [off, 0], ok)
                og = pl.store(pl.sub(pl.mul(dq_n, q), pl.mul(dko_n, k)), [off, 0], og)
                acc_fin = pl.yield_(acc_n)
            return oq, ok, ov, og, oh, oc, pl.store(acc_fin, [0, 0], dSrecv)

        @pl.function(type=pl.FunctionType.InCore)
        def grad_h(
            self,
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            dH: pl.Tensor[[N * DK, DV], pl.FP32],
            dCp: pl.Tensor[[N * DK, 1], pl.FP32],
            dKo: pl.Tensor[[L, DK], pl.FP32],
            dVo: pl.Tensor[[L, DV], pl.FP32],
            dgcso: pl.Tensor[[L, DK], pl.FP32],
            dStot: pl.Tensor[[DK, DV], pl.FP32],
            dgam: pl.Tensor[[DK, 1], pl.FP32],
            dK: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dV: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dA: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
        ):
            tril_t = pl.load(tril, [0, 0], [C, C])
            triu_t = pl.load(triu, [0, 0], [C, C])
            ds_init = pl.load(dStot, [0, 0], [DK, DV])
            dc_init = pl.load(dgam, [0, 0], [DK, 1])
            okk = dK
            ovv = dV
            oaa = dA
            for m, (dsloc, dcvec) in pl.range(0, N, init_values=(ds_init, dc_init)):
                off = (N - 1 - m) * C
                soff = (N - 1 - m) * DK
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                gamma = pl.exp(pl.tile.reshape(pl.tile.col_sum(la), [DK, 1]))
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                sprev = pl.load(Ssnap, [soff, 0], [DK, DV])
                cprev = pl.load(Cprev, [soff, 0], [DK, 1])
                dsl_p = pl.tile.row_expand_mul(dsloc, gamma)
                dcv_p = pl.mul(dcvec, gamma)
                kb = pl.div(k, b)
                dv_h = pl.matmul(kb, dsl_p, out_dtype=pl.FP32)
                dk_h = pl.div(pl.matmul(v, pl.transpose(dsl_p, 0, 1), out_dtype=pl.FP32), b)
                dkk = pl.mul(dk_h, k)
                tmp = pl.tile.create([DK, DV], pl.FP32)
                c1 = pl.tile.reshape(pl.row_sum(pl.mul(dsl_p, sprev), tmp), [1, DK])
                c2 = pl.tile.reshape(pl.mul(dcv_p, cprev), [1, DK])
                corr = pl.add(pl.add(c1, c2), pl.tile.col_sum(dkk))
                dgcs = pl.sub(pl.load(dgcso, [off, 0], [C, DK]), dkk)
                rcs = pl.matmul(triu_t, dgcs, out_dtype=pl.FP32)
                oaa = pl.store(pl.div(pl.tile.col_expand_add(rcs, corr), a), [off, 0], oaa)
                okk = pl.store(pl.add(pl.load(dKo, [off, 0], [C, DK]), dk_h), [off, 0], okk)
                ovv = pl.store(pl.add(pl.load(dVo, [off, 0], [C, DV]), dv_h), [off, 0], ovv)
                dsloc_n = pl.add(dsl_p, pl.load(dH, [soff, 0], [DK, DV]))
                dcvec_n = pl.add(dcv_p, pl.load(dCp, [soff, 0], [DK, 1]))
                dsl_f, dcv_f = pl.yield_(dsloc_n, dcvec_n)
            return okk, ovv, oaa

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_recompute(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Cprev: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            return self.recompute(A, Kmat, Vmat, tril, zero, onev, Ssnap, Cprev, Stot)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_grad_o(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            dOmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            dQ: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dKo: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dVo: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dgcso: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dH: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCp: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            dSrecv: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            return self.grad_o(Q, Kmat, Vmat, A, dOmat, tril, Ssnap, Cprev, Srecv, zero,
                               dQ, dKo, dVo, dgcso, dH, dCp, dSrecv)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_grad_h(
            self,
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            dH: pl.Tensor[[N * DK, DV], pl.FP32],
            dCp: pl.Tensor[[N * DK, 1], pl.FP32],
            dKo: pl.Tensor[[L, DK], pl.FP32],
            dVo: pl.Tensor[[L, DV], pl.FP32],
            dgcso: pl.Tensor[[L, DK], pl.FP32],
            dStot: pl.Tensor[[DK, DV], pl.FP32],
            dgam: pl.Tensor[[DK, 1], pl.FP32],
            dK: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dV: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dA: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
        ):
            return self.grad_h(Kmat, Vmat, A, tril, triu, Ssnap, Cprev, dH, dCp,
                               dKo, dVo, dgcso, dStot, dgam, dK, dV, dA)

    return KernelProbe


def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3sim"
    torch.manual_seed(3)
    A = 0.9 + 0.1 * torch.sigmoid(torch.randn(L, DK))
    Qt = torch.randn(L, DK)
    Kt = torch.randn(L, DK)
    Vt = torch.randn(L, DV)
    dOt = torch.randn(L, DV)
    tril = torch.tril(torch.ones(C, C))
    triu = torch.triu(torch.ones(C, C))
    zero = torch.zeros(DK, DV)
    onev = torch.ones(DK, 1)

    # The whole point: boundary values a P=1 run would leave at zero.
    Srecv = torch.randn(DK, DV) * 0.5
    dStot = torch.randn(DK, DV) * 0.5
    dgam = torch.randn(DK, 1) * 0.5

    gS, gC, gT = k_recompute(Kt, Vt, A, tril, C)
    gdQ, gdKo, gdVo, gdgcso, gdH, gdCp, gdSrecv = k_grad_o(
        Qt, Kt, Vt, A, dOt, tril, C, gS, gC, Srecv)
    gdK, gdV, gdA = k_grad_h(Kt, Vt, A, tril, triu, C, gS, gC, gdH, gdCp,
                             gdKo, gdVo, gdgcso, dStot, dgam)

    compiled = ir.compile(build(), platform=platform)
    cfg = RunConfig(platform=platform, device_id=0)
    # One program, three Orchestration entries: select by name (a bare `compiled(...)`
    # would pick the default entry, whose signature is whichever kernel sorted last).
    run_recompute = compiled["chip_recompute"]
    run_grad_o = compiled["chip_grad_o"]
    run_grad_h = compiled["chip_grad_h"]

    Ssnap = torch.zeros(N * DK, DV)
    Cprev = torch.zeros(N * DK, 1)
    Stot = torch.zeros(DK, DV)
    run_recompute(A, Kt, Vt, tril, zero, onev, Ssnap, Cprev, Stot, config=cfg)

    dQ = torch.zeros(L, DK)
    dKo = torch.zeros(L, DK)
    dVo = torch.zeros(L, DV)
    dgcso = torch.zeros(L, DK)
    dH = torch.zeros(N * DK, DV)
    dCp = torch.zeros(N * DK, 1)
    dSrecv = torch.zeros(DK, DV)
    run_grad_o(Qt, Kt, Vt, A, dOt, tril, Ssnap, Cprev, Srecv, zero,
                    dQ, dKo, dVo, dgcso, dH, dCp, dSrecv, config=cfg)

    dK = torch.zeros(L, DK)
    dV = torch.zeros(L, DV)
    dA = torch.zeros(L, DK)
    run_grad_h(Kt, Vt, A, tril, triu, Ssnap, Cprev, dH, dCp, dKo, dVo, dgcso,
                    dStot, dgam, dK, dV, dA, config=cfg)

    checks = [
        ("recompute", [("Ssnap", Ssnap, gS), ("Cprev", Cprev, gC), ("Stot", Stot, gT)]),
        ("grad_o", [("dQ", dQ, gdQ), ("dKo", dKo, gdKo), ("dVo", dVo, gdVo),
                    ("dgcso", dgcso, gdgcso), ("dH", dH, gdH), ("dCp", dCp, gdCp),
                    ("dSrecv", dSrecv, gdSrecv)]),
        ("grad_h", [("dK", dK, gdK), ("dV", dV, gdV), ("dA", dA, gdA)]),
    ]
    bad = []
    for kernel, items in checks:
        print(f"--- {kernel}")
        for nm, got, ref in items:
            rel = ((got - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
            ok = rel < 1e-4
            bad += [] if ok else [f"{kernel}.{nm}"]
            print(f"  {'ok ' if ok else 'BAD'} {nm:7s} rel {rel:.3e}   |ref|max {ref.abs().max().item():.4f}")
    print("\nKERNEL PROBE PASS" if not bad else "\nKERNEL PROBE FAIL: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
