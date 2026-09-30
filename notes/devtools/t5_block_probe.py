#!/usr/bin/env python3
"""Task 5 pre-flight: can the DSL express a DK-blocked chunk scan at all?

The arithmetic is already proven identical in torch (``t5_math_check.py``). What is NOT
established is whether pypto can express it. Three things it has never been asked to do:

  1. **slice a loop-carried tile** -- ``pl.tile.slice(s_run, [BK, DV], [j*BK, 0])``. The
     ``[DK, DV]`` state is carried across chunks; blocking DK only pays if a ``[BK, DV]``
     row-slice can be read out of that carry.
  2. **assemble back into a whole tile** -- ``pl.tile.assemble(target, blk, [j*BK, 0])``,
     rebuilding the next state one block-row at a time.
  3. **sum matmul partials in the vector unit** -- ``qt @ S`` and ``qt @ kb^T`` both contract
     over DK, so blocking turns each into ``NB`` partials added with ``pl.add``. fp32
     accumulation inside the cube is broken on a2a3, so this is the only route.

Q/K/A need no tile slicing at all: a DK block is just a narrower ``pl.load``. Only the
carried state does.

The block loop is unrolled in PYTHON, not a nested ``pl.range``. NB is small and static, and
it sidesteps nested loop-carry entirely -- one unknown at a time.

Checked against a torch golden, so "it compiled" is never mistaken for "it works".

Usage: python3 devtools/t5_block_probe.py <platform> [device_id] [NB]
"""
from __future__ import annotations

import os
import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

# Overridable from argv so the probe can be walked up to the real target shapes.
C, DK, DV, N = 16, 32, 16, 3
L = N * C

# A tile's column count must be a multiple of 16 (ptoas `innerCols`), so the DK block width
# is not free: BK = DK // NB must be >= 16 and a multiple of 16. NB=4 at DK=32 gives BK=8 and
# fails at codegen, not at parse -- "'pto.alloc_tile' op expects result boxed tile cols to be
# a multiple of innerCols (16), but got 8".
INNER_COLS = 16


def build(NB: int, slot: int = 4):
    BK = DK // NB
    SLOT = slot

    @pl.program
    class BlockProbe:
        # The cube<->vector pipe ring is reserved off the TOP of the vector buffer, sized as
        # (biggest crossing tile) x (ring depth), before a single tile is allocated. At
        # C=64, DV=128 that is 131072 B of a 188416 B budget, and it does NOT shrink with DK
        # blocking. The overflow message points at `pl.cross_core_slot(slot_num=N)` "on the
        # enclosing pl.at(...)", but a DECLARED InCore function has no enclosing pl.at --
        # wrapping the body in one fails with "InCore ScopeStmt found in non-InCore function".
        # ExpandMixedKernel reads the override off a function ATTR, so that is the route here.
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
            Sout: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            pl.func_attr({"slot_num": SLOT})
            tril_t = pl.load(tril, [0, 0], [C, C])
            s_init = pl.load(Srecv, [0, 0], [DK, DV])
            out = O
            sout = Sout
            for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
                off = n * C
                v = pl.load(Vmat, [off, 0], [C, DV])
                s_run_v = pl.mul(s_run, 1.0)  # detach carry for matmul/slice
                # Zero seeds for the two accumulators, without new host params.
                sc0 = pl.mul(tril_t, 0.0)
                oi0 = pl.mul(v, 0.0)
                # The block loop is a pl.range, NOT a Python `for`: pypto parses the
                # function's own source, so a Python loop inside a kernel is read as a device
                # loop -- one that reassigns names without pl.yield_, which is exactly the
                # "N iteration arguments but 0 return variables" the parser rejects.
                for j, (scores_acc, o_inter, s_acc) in pl.range(
                    0, NB, init_values=(sc0, oi0, s_run_v)
                ):
                    dof = j * BK
                    # (Q/K/A block = a narrower load; no tile slicing needed.)
                    q_b = pl.load(Q, [off, dof], [C, BK])
                    k_b = pl.load(Kmat, [off, dof], [C, BK])
                    a_b = pl.load(A, [off, dof], [C, BK])
                    la_b = pl.log(a_b)
                    b_b = pl.exp(pl.matmul(tril_t, la_b, out_dtype=pl.FP32))
                    gamma_b = pl.exp(pl.tile.reshape(pl.tile.col_sum(la_b), [BK, 1]))
                    qt_b = pl.mul(q_b, b_b)
                    kb_b = pl.div(k_b, b_b)
                    kbt_b = pl.transpose(kb_b, 0, 1)
                    # (1) slice the carried state.
                    s_blk = pl.tile.slice(s_run_v, [BK, DV], [dof, 0])
                    # (3) vector-unit accumulation of the two DK-contracting matmuls.
                    sc_p = pl.matmul(qt_b, kbt_b, out_dtype=pl.FP32)
                    oi_p = pl.matmul(qt_b, s_blk, out_dtype=pl.FP32)
                    sc_n = pl.add(scores_acc, sc_p)
                    oi_n = pl.add(o_inter, oi_p)
                    kv_b = pl.matmul(kbt_b, v, out_dtype=pl.FP32)
                    s_blk_new = pl.tile.row_expand_mul(pl.add(s_blk, kv_b), gamma_b)
                    # (2) rebuild the next state one block-row at a time.
                    s_acc_n = pl.tile.assemble(s_acc, s_blk_new, [dof, 0])
                    scores_acc, o_inter, s_acc = pl.yield_(sc_n, oi_n, s_acc_n)
                scores = pl.mul(scores_acc, tril_t)
                o_n = pl.add(o_inter, pl.matmul(scores, v, out_dtype=pl.FP32))
                out = pl.store(o_n, [off, 0], out)
                s_run = pl.yield_(s_acc)
            sout = pl.store(s_run, [0, 0], sout)
            return out, sout

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
            Sout: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            return self.scan(Q, Kmat, Vmat, A, tril, Srecv, O, Sout)

    return BlockProbe


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
        print(f"SKIP  C={C} DK={DK} DV={DV} N={N} NB={NB}: "
              f"BK={DK // NB if not DK % NB else '?'} is not a multiple of {INNER_COLS}")
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
    Sout = torch.zeros(DK, DV)
    compiled(q, k, v, a, tril, S0, O, Sout,
             config=RunConfig(platform=platform, device_id=device_id))
    eO = (O - gO).abs().max().item()
    eS = (Sout - gS).abs().max().item()
    print(f"max|O - golden| = {eO:.3e}")
    print(f"max|S - golden| = {eS:.3e}")
    ok = eO < 1e-3 and eS < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
