"""Fully-fused distributed PyPTO ZeCO **backward**: five phases in ONE program (B4).

The mirror of :mod:`.fused_program`. Per rank ``r`` (device ``r``) the single ``host_orch``
runs, with no host round-trip between any of them:

1. **recompute** (InCore) — re-runs the forward chunk scan, but *snapshotting* the per-chunk
   pre-state ``S_prev[n]`` and cumulative decay ``c_prev[n]`` that the backward reads. Also
   emits ``S_total``, the forward ring's input. (Activations are recomputed rather than
   carried: :meth:`gla.common.ZeCoImpl.backward` is stateless by contract, and ``[N,dk,dv]``
   of snapshots is cheaper to regenerate than to ship.)
2. **AllScan ring** (first/middle/last) — the same forward boundary scan as the forward
   program, giving each rank its ``S_recv``. Needed twice over: ``grad_o`` reconstructs
   ``H_n = S_prev[n] + c_prev[n]*S_recv``, and the reverse ring reduces ``dgamma`` against it.
3. **grad_o** (InCore) — the output-stage adjoints, one pass over chunks in forward order:
   ``dQ``, the intra-chunk halves of ``dK``/``dV``, the log-domain gate grad ``dg_cs``, the
   per-chunk ``dH_n``/``dc_prev[n]`` that feed the state stage, and the accumulated
   ``dS_recv`` that feeds the reverse ring.
4. **reverse ring** (source/middle/terminal) — the adjoint of the boundary scan, flowing
   ``r -> r-1``, producing ``dS_total[r]`` and ``dgamma[r]``.
5. **grad_h** (InCore) — the reverse chunk recurrence: carries the state adjoint ``dSloc``
   and decay adjoint ``dcvec`` backwards over chunks, adds the state-path halves of
   ``dK``/``dV``, and finishes the gate backward into ``dA``.

``P == 1`` is a native path (:func:`_build_p1_backward_program`): no boundary, so phases 2
and 4 vanish and 1/3/5 run from a zero ``S_recv`` / ``dS_total`` / ``dgamma``.

**P=1 and P>1 MUST be separate factory functions**, and the three compute kernels are
therefore written out twice. A conditionally-defined method in a ``@pl.program`` class body
is silently NOT registered (the forward learned this the hard way — it collapses the program
to its last kernel and ranks > 0 never receive their boundary), and the bodies cannot be
factored into a shared helper because ``@pl.program`` parses the class *source*: a call to a
module-level Python function is not a call the parser can expand. ``@pl.inline`` defers
parsing and would expand in place, but its behaviour on ``pl.Out`` store chains and tuple
returns is unproven, so this follows the forward's precedent (which duplicates ``gla_stage2``
for the same reason). The duplicate pairs are kept adjacent and both are exercised by
``gla/tests/test_pypto_gla_backward.py`` against the same golden, so drift shows up as a
test failure rather than as silently different numbers.

Why the reverse ring is simpler here than in :mod:`allscan.implementations.pypto.program_backward`
-------------------------------------------------------------------------------------------------
The standalone AllScan backward takes a host-assembled ``g_out`` and a host-supplied
``out_prev``. Fused, both disappear:

* ``d[p] = g_out[p] + gamma[p+1]*d[p+1]`` and ``g_out[p] = dS_recv[p+1]`` (rank ``p``'s
  boundary *is* rank ``p+1``'s ``S_recv``), so the message rank ``p+1`` sends to ``p`` —
  ``dS_recv[p+1] + gamma[p+1]*d[p+1]`` — **is** ``d[p]``. The receiver adds nothing; no
  cross-rank gather of ``g_out`` is needed.
* ``out_prev[p] == S_recv[p]``, which the rank already holds device-locally from phase 2.

Rank ``P-1`` is the source with ``d = 0`` (``out[P-1]`` feeds nothing), so it writes zeros to
``dS_total``/``dgamma``; rank 0 is the terminal and ``dgamma[0]`` is zero because ``gamma[0]``
is unused.

Shape restructuring the DSL forces (validated in ``../devtools/b4_math_check.py``)
----------------------------------------------------------------------------------
``gamma`` is a ``[dk,1]`` per-key-dim vector, so ``k * (gamma/b)`` — a *column* broadcast
over a ``[C,dk]`` tile — is not expressible. Every use is refactored to push ``gamma`` onto
the ``[dk,dv]`` state instead, exactly the trick F3.1 used in the forward::

    dV_h = (k/b) @ (gamma*dSloc)            not  (k*gamma/b) @ dSloc
    dK_h = (v @ (gamma*dSloc)^T) / b        not  (v @ dSloc^T) * (gamma/b)

and the three ``dgamma`` terms are formed already carrying their gamma factor. One
``row_expand_mul`` on ``dSloc`` therefore replaces three column broadcasts per chunk.

The gate gradient is carried in the **log domain** (``dg_cs = db * b``), so ``db`` never
exists and the ``b`` factors cancel: ``dg_cs`` from the output stage is ``dQ*q - dK_o*k`` and
from the state stage ``-dK_h*k``. The single-row update ``db[C-1] += dgamma`` becomes a
whole-tile ``col_expand_add`` applied *after* the reverse cumulative sum: row ``C-1`` is
``>= t`` for every ``t``, so it contributes the same constant to every row. The reverse
cumsum itself is a matmul by an upper-triangular ones matrix, not a scan.

Vector-buffer budget
--------------------
``grad_o`` is the widest kernel in either direction — three ``[C,C]`` tiles, ~12 ``[C,dk]``
row tiles and ~6 ``[dk,dv]`` state tiles live at once, roughly double the forward's stage2,
which itself sits at 96% of the 184 KB budget at ``C=D=64``. The backward therefore tops out
below the forward; the reachable set is measured rather than assumed (see the B4 entry in
ROADMAP.md). Blocking it further is task 5's job, and the same DK/DV design serves both.

Every distributed / HCCL run must set ``LD_PRELOAD=<cann>/lib64/libhccl.so``.
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld


def _build_p1_backward_program(L: int, C: int, dk: int, dv: int):
    """P == 1 native path: recompute -> grad_o -> grad_h, all from a zero boundary.

    A single rank has no neighbour, so ``S_recv``, ``dS_total`` and ``dgamma`` are all zero
    and both rings are dead. The three kernels are byte-identical to their counterparts in
    :func:`build_fused_backward_program` — see the module docstring for why they cannot be
    shared.
    """
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    N = L // C
    P, DK, DV = 1, dk, dv

    @pl.program
    class FusedBackwardP1Program:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_recompute(
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
            """Forward chunk scan, snapshotting S_prev[n] / c_prev[n] before each update."""
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
        def gla_grad_o(
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
            """Output-stage adjoints; chunks independent apart from the dS_recv sum."""
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
        def gla_grad_h(
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
            """Reverse chunk recurrence + gate backward. pl.range only counts up, so the
            reverse walk is n = N-1-m; the n>0 guard on the carry update is dropped because
            at n==0 the new carry is dead and an InCore branch costs more than one add."""
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
                # Push gamma onto the state ONCE; everything downstream is broadcast-free.
                # This also detaches the raw iter_arg, which cannot feed a matmul directly.
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
        ) -> pl.Tuple[
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.gla_recompute(A, Kmat, Vmat, tril, zero, onev, Ssnap, Cprev, Stot)

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
        ) -> pl.Tuple[
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DV], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.gla_grad_o(Q, Kmat, Vmat, A, dOmat, tril, Ssnap, Cprev, Srecv, zero,
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
        ) -> pl.Tuple[
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DV], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
        ]:
            return self.gla_grad_h(Kmat, Vmat, A, tril, triu, Ssnap, Cprev, dH, dCp,
                                   dKo, dVo, dgcso, dStot, dgam, dK, dV, dA)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            Qmat: pl.Tensor[[P, L, dk], pl.FP32],
            Kmat: pl.Tensor[[P, L, dk], pl.FP32],
            Vmat: pl.Tensor[[P, L, dv], pl.FP32],
            A: pl.Tensor[[P, L, dk], pl.FP32],
            dOmat: pl.Tensor[[P, L, dv], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            zerov: pl.Tensor[[dk, 1], pl.FP32],
            onev: pl.Tensor[[dk, 1], pl.FP32],
            dQ: pl.Out[pl.Tensor[[P, L, dk], pl.FP32]],
            dK: pl.Out[pl.Tensor[[P, L, dk], pl.FP32]],
            dV: pl.Out[pl.Tensor[[P, L, dv], pl.FP32]],
            dA: pl.Out[pl.Tensor[[P, L, dk], pl.FP32]],
        ):
            """P == 1: no boundary, so S_recv / dS_total / dgamma are the zero tensors and
            both rings are gone. ``gammas`` is unused."""
            Ssnap = pl.create_tensor([P, N * dk, dv], dtype=pl.FP32)
            Cprev = pl.create_tensor([P, N * dk, 1], dtype=pl.FP32)
            Stot = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            dH = pl.create_tensor([P, N * dk, dv], dtype=pl.FP32)
            dCp = pl.create_tensor([P, N * dk, 1], dtype=pl.FP32)
            dSrecv = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            dKo = pl.create_tensor([P, L, dk], dtype=pl.FP32)
            dVo = pl.create_tensor([P, L, dv], dtype=pl.FP32)
            dgcso = pl.create_tensor([P, L, dk], dtype=pl.FP32)

            # `sl_r` (S_total) and `dsr` (dS_recv) are unpacked but unused, and pypto's
            # UnusedVariableCheck says so: both exist only to feed the rings, which P=1 does
            # not have. They are kept — the tuple element order maps positionally onto the
            # callee's Out params, so an element cannot be dropped from the middle, and
            # keeping the kernels identical to the P>1 pair is worth three warnings.
            for r in pl.range(P):
                snap, cp, sl_r = self.chip_recompute(
                    A[r], Kmat[r], Vmat[r], tril, zero, onev,
                    Ssnap[r], Cprev[r], Stot[r], device=r)
                dq_r, dko, dvo, dgo, dh, dcp, dsr = self.chip_grad_o(
                    Qmat[r], Kmat[r], Vmat[r], A[r], dOmat[r], tril, snap, cp, zero, zero,
                    dQ[r], dKo[r], dVo[r], dgcso[r], dH[r], dCp[r], dSrecv[r], device=r)
                self.chip_grad_h(
                    Kmat[r], Vmat[r], A[r], tril, triu, snap, cp, dh, dcp, dko, dvo, dgo,
                    zero, zerov, dK[r], dV[r], dA[r], device=r)
            return dQ, dK, dV, dA

    return FusedBackwardP1Program


def build_fused_backward_program(L: int, C: int, dk: int, dv: int, K: int, P: int):
    """Build the fully-fused ``recompute + ring + grad_o + reverse-ring + grad_h`` program.

    Args:
        L: Tokens per device. C: chunk size (``L % C == 0``, ``N = L // C``).
        dk, dv: key/query and value dims. K: ring pipeline depth (``dk % K == 0``).
        P: ranks / devices. ``P == 1`` builds the native single-rank program (no rings).

    Returns:
        A ``@pl.program`` whose ``host_orch`` takes ``(Qmat, Kmat, Vmat, A, dOmat, gammas,
        tril, triu, zero, zerov, onev)`` and writes ``(dQ, dK, dV, dA)`` ``[P, L, ...]``.
    """
    assert dk % K == 0, f"dk ({dk}) must be divisible by K ({K})"
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    if P == 1:
        return _build_p1_backward_program(L, C, dk, dv)

    BLOCK = dk // K
    N = L // C
    DK, DV = dk, dv

    @pl.program
    class FusedBackwardProgram:
        # ---- phase 1: forward recompute with per-chunk snapshots ----
        @pl.function(type=pl.FunctionType.InCore)
        def gla_recompute(
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
            """Forward chunk scan, snapshotting S_prev[n] / c_prev[n] before each update.

            Identical arithmetic to the forward's ``gla_stage1`` (same gamma-factored state
            update); the only additions are the two snapshot stores and the ``c_prev``
            carry. Snapshots live at row offset ``n*DK`` because a tile is physically 2D.
            """
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
        ) -> pl.Tuple[
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.gla_recompute(A, Kmat, Vmat, tril, zero, onev, Ssnap, Cprev, Stot)

        # ---- phase 3: output-stage adjoints ----
        @pl.function(type=pl.FunctionType.InCore)
        def gla_grad_o(
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
            """Output-stage adjoints. Chunks are independent apart from the dS_recv sum::

                H_n     = S_prev[n] + c_prev[n]*S_recv     reconstruct's inter-chunk history
                dQt     = dO @ H_n^T  +  dsc @ (k/b)
                dH_n    = (q*b)^T @ dO                     -> the state stage's dS_prev[n]
                dsc     = (dO @ v^T) * tril                intra-chunk masked attention
                dV_o    = scores^T @ dO ,  dK_o = (dsc^T @ (q*b)) / b
                dg_cs_o = dQ*q - dK_o*k                    log-domain gate grad
            """
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
        ) -> pl.Tuple[
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DV], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.gla_grad_o(Q, Kmat, Vmat, A, dOmat, tril, Ssnap, Cprev, Srecv, zero,
                                   dQ, dKo, dVo, dgcso, dH, dCp, dSrecv)

        # ---- phase 5: state-stage adjoints + gate backward ----
        @pl.function(type=pl.FunctionType.InCore)
        def gla_grad_h(
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
            """Reverse chunk recurrence + gate backward. pl.range only counts up, so the
            reverse walk is n = N-1-m; the n>0 guard on the carry update is dropped because
            at n==0 the new carry is dead and an InCore branch costs more than one add."""
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
                # Push gamma onto the state ONCE; everything downstream is broadcast-free.
                # This also detaches the raw iter_arg, which cannot feed a matmul directly.
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
        ) -> pl.Tuple[
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DV], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
        ]:
            return self.gla_grad_h(Kmat, Vmat, A, tril, triu, Ssnap, Cprev, dH, dCp,
                                   dKo, dVo, dgcso, dStot, dgam, dK, dV, dA)

        # ---- phase 2: forward AllScan ring (identical to the forward program's) ----
        @pl.function(type=pl.FunctionType.InCore)
        def allscan_first_step(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                S_send_k = pl.load(S_local, [offset_k, 0], [BLOCK, dv])
                S_out = pl.store(S_send_k, [offset_k, 0], S_out)
                pld.tile.remote_store(S_send_k, target=dst, peer=peer_next, offsets=[offset_k, 0])
                pld.system.notify(target=signal, peer=peer_next, offsets=[kk, 0], value=1,
                                  op=pld.NotifyOp.AtomicAdd)
            return S_out

        @pl.function(type=pl.FunctionType.InCore)
        def allscan_middle_step(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_recv: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                pld.system.wait(signal=signal, offsets=[kk, 0], expected=1, cmp=pld.WaitCmp.Ge)
                S_recv_k = pl.load(dst, [offset_k, 0], [BLOCK, dv])
                S_recv = pl.store(S_recv_k, [offset_k, 0], S_recv)
                S_local_k = pl.load(S_local, [offset_k, 0], [BLOCK, dv])
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK, 1])
                scaled_recv_k = pl.tile.row_expand_mul(S_recv_k, gamma_k)
                S_send_k = pl.tile.add(S_local_k, scaled_recv_k)
                pld.tile.remote_store(S_send_k, target=dst, peer=peer_next, offsets=[offset_k, 0])
                pld.system.notify(target=signal, peer=peer_next, offsets=[kk, 0], value=1,
                                  op=pld.NotifyOp.AtomicAdd)
            return S_recv

        @pl.function(type=pl.FunctionType.InCore)
        def allscan_last_step(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_recv: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                pld.system.wait(signal=signal, offsets=[kk, 0], expected=1, cmp=pld.WaitCmp.Ge)
                S_recv_k = pl.load(dst, [offset_k, 0], [BLOCK, dv])
                S_recv = pl.store(S_recv_k, [offset_k, 0], S_recv)
            return S_recv

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_first(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            return self.allscan_first_step(S_local, S_out, dst, signal, peer_next)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_middle(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_recv: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            return self.allscan_middle_step(S_local, gamma, S_recv, dst, signal, peer_next)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_last(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_recv: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            return self.allscan_last_step(S_local, gamma, S_recv, dst, signal)

        # ---- phase 4: reverse ring. The message p -> p-1 IS d[p-1]. ----
        @pl.function(type=pl.FunctionType.InCore)
        def bwd_source_step(
            self,
            dSrecv: pl.Tensor[[dk, dv], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            zerov: pl.Tensor[[dk, 1], pl.FP32],
            dStot: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgam: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ):
            """Rank P-1: ``out[P-1]`` feeds nothing, so ``d[P-1] = 0`` and both local grads
            are zero. The outgoing message is this rank's own ``dS_recv``, which IS
            ``d[P-2]``."""
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                dStot = pl.store(pl.load(zero, [offset_k, 0], [BLOCK, dv]), [offset_k, 0], dStot)
                dgam = pl.store(pl.load(zerov, [offset_k, 0], [BLOCK, 1]), [offset_k, 0], dgam)
                msg_k = pl.load(dSrecv, [offset_k, 0], [BLOCK, dv])
                pld.tile.remote_store(msg_k, target=dst, peer=peer_prev, offsets=[offset_k, 0])
                pld.system.notify(target=signal, peer=peer_prev, offsets=[kk, 0], value=1,
                                  op=pld.NotifyOp.AtomicAdd)
            return dStot, dgam

        @pl.function(type=pl.FunctionType.InCore)
        def bwd_middle_step(
            self,
            dSrecv: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            Srecv: pl.Tensor[[dk, dv], pl.FP32],
            dStot: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgam: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ):
            """Middle ranks: the received block IS ``d[p]`` — nothing is added to it.
            ``dgamma`` reduces against this rank's own ``S_recv`` (== ``out[p-1]``), so no
            peer data is needed for it either."""
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                pld.system.wait(signal=signal, offsets=[kk, 0], expected=1, cmp=pld.WaitCmp.Ge)
                d_k = pl.load(dst, [offset_k, 0], [BLOCK, dv])
                dStot = pl.store(d_k, [offset_k, 0], dStot)

                # Send before reducing: keeps d_k from coexisting with the row_sum's product
                # and scratch tiles (the same Vec-budget argument as the AllScan backward).
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK, 1])
                dsr_k = pl.load(dSrecv, [offset_k, 0], [BLOCK, dv])
                msg_k = pl.tile.add(dsr_k, pl.tile.row_expand_mul(d_k, gamma_k))
                pld.tile.remote_store(msg_k, target=dst, peer=peer_prev, offsets=[offset_k, 0])
                pld.system.notify(target=signal, peer=peer_prev, offsets=[kk, 0], value=1,
                                  op=pld.NotifyOp.AtomicAdd)

                sr_k = pl.load(Srecv, [offset_k, 0], [BLOCK, dv])
                tmp_k = pl.tile.create([BLOCK, dv], pl.FP32)
                dgam = pl.store(pl.row_sum(pl.tile.mul(d_k, sr_k), tmp_k), [offset_k, 0], dgam)
            return dStot, dgam

        @pl.function(type=pl.FunctionType.InCore)
        def bwd_terminal_step(
            self,
            zerov: pl.Tensor[[dk, 1], pl.FP32],
            dStot: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgam: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ):
            """Rank 0: receive ``d[0]``, store it, stop. ``gamma[0]`` is unused, so
            ``dgamma[0]`` is written as zero here rather than left to the host."""
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                pld.system.wait(signal=signal, offsets=[kk, 0], expected=1, cmp=pld.WaitCmp.Ge)
                d_k = pl.load(dst, [offset_k, 0], [BLOCK, dv])
                dStot = pl.store(d_k, [offset_k, 0], dStot)
                dgam = pl.store(pl.load(zerov, [offset_k, 0], [BLOCK, 1]), [offset_k, 0], dgam)
            return dStot, dgam

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_bwd_source(
            self,
            dSrecv: pl.Tensor[[dk, dv], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            zerov: pl.Tensor[[dk, 1], pl.FP32],
            dStot: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgam: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ) -> pl.Tuple[
            pl.Tensor[[dk, dv], pl.FP32],
            pl.Tensor[[dk, 1], pl.FP32],
        ]:
            return self.bwd_source_step(dSrecv, zero, zerov, dStot, dgam, dst, signal, peer_prev)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_bwd_middle(
            self,
            dSrecv: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            Srecv: pl.Tensor[[dk, dv], pl.FP32],
            dStot: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgam: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ) -> pl.Tuple[
            pl.Tensor[[dk, dv], pl.FP32],
            pl.Tensor[[dk, 1], pl.FP32],
        ]:
            return self.bwd_middle_step(dSrecv, gamma, Srecv, dStot, dgam, dst, signal, peer_prev)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_bwd_terminal(
            self,
            zerov: pl.Tensor[[dk, 1], pl.FP32],
            dStot: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgam: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ) -> pl.Tuple[
            pl.Tensor[[dk, dv], pl.FP32],
            pl.Tensor[[dk, 1], pl.FP32],
        ]:
            return self.bwd_terminal_step(zerov, dStot, dgam, dst, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            Qmat: pl.Tensor[[P, L, dk], pl.FP32],
            Kmat: pl.Tensor[[P, L, dk], pl.FP32],
            Vmat: pl.Tensor[[P, L, dv], pl.FP32],
            A: pl.Tensor[[P, L, dk], pl.FP32],
            dOmat: pl.Tensor[[P, L, dv], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            zerov: pl.Tensor[[dk, 1], pl.FP32],
            onev: pl.Tensor[[dk, 1], pl.FP32],
            dQ: pl.Out[pl.Tensor[[P, L, dk], pl.FP32]],
            dK: pl.Out[pl.Tensor[[P, L, dk], pl.FP32]],
            dV: pl.Out[pl.Tensor[[P, L, dv], pl.FP32]],
            dA: pl.Out[pl.Tensor[[P, L, dk], pl.FP32]],
        ):
            """Per rank r on device r: recompute -> forward ring -> grad_o -> reverse ring
            -> grad_h, in ONE ascending loop.

            The reverse ring flows r -> r-1 while the loop submits r ascending, so rank 0
            blocks on a message from rank P-1 whose sender is submitted later. That looks
            like a deadlock and was investigated as one -- but it is exactly the structure
            of `allscan/implementations/pypto/program_backward.py`, which is HW-validated at
            P>=2, so submission order is evidently not what sequences these. A descending
            loop is not expressible anyway: MaterializeCommDomainScopes requires `device=` to
            BE an enclosing pl.range induction variable ("device= Var is not the induction
            variable of any enclosing pl.range loop") and rejects a non-unit step ("device=r
            over a non-unit-step loop is not supported (step=-1)").

            The two rings keep SEPARATE window buffers: they traverse the same ranks in
            opposite directions, and sharing one would let a reverse message land in a slot
            whose forward value is still live. NOTE: two comm windows in one program is the
            one structural thing this program does that no validated program does, and P>1
            currently fails on device here -- see the B4 entry in ROADMAP.md.
            """
            fdst_buf = pld.alloc_window_buffer(dk * dv * 4)
            fsig_buf = pld.alloc_window_buffer(K * 4)
            bdst_buf = pld.alloc_window_buffer(dk * dv * 4)
            bsig_buf = pld.alloc_window_buffer(K * 4)

            Ssnap = pl.create_tensor([P, N * dk, dv], dtype=pl.FP32)
            Cprev = pl.create_tensor([P, N * dk, 1], dtype=pl.FP32)
            Stot = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            S_out_all = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            S_recv_all = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            dH = pl.create_tensor([P, N * dk, dv], dtype=pl.FP32)
            dCp = pl.create_tensor([P, N * dk, 1], dtype=pl.FP32)
            dSrecv = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            dKo = pl.create_tensor([P, L, dk], dtype=pl.FP32)
            dVo = pl.create_tensor([P, L, dv], dtype=pl.FP32)
            dgcso = pl.create_tensor([P, L, dk], dtype=pl.FP32)
            dStot = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            dgam = pl.create_tensor([P, dk, 1], dtype=pl.FP32)

            for r in pl.range(P):
                fdst = pld.window(fdst_buf, [dk, dv], dtype=pl.FP32)
                fsig = pld.window(fsig_buf, [K, 1], dtype=pl.INT32)
                bdst = pld.window(bdst_buf, [dk, dv], dtype=pl.FP32)
                bsig = pld.window(bsig_buf, [K, 1], dtype=pl.INT32)

                snap, cp, sl_r = self.chip_recompute(
                    A[r], Kmat[r], Vmat[r], tril, zero, onev,
                    Ssnap[r], Cprev[r], Stot[r], device=r)

                # Same boundary phi as the forward: rank 0 has none (S_recv = 0).
                if r == 0:
                    self.chip_orch_first(sl_r, S_out_all[r], fdst, fsig, r + 1, device=r)
                    boundary = zero
                elif r == P - 1:
                    boundary = self.chip_orch_last(
                        sl_r, gammas[r], S_recv_all[r], fdst, fsig, device=r)
                else:
                    boundary = self.chip_orch_middle(
                        sl_r, gammas[r], S_recv_all[r], fdst, fsig, r + 1, device=r)

                dq_r, dko, dvo, dgo, dh, dcp, dsr = self.chip_grad_o(
                    Qmat[r], Kmat[r], Vmat[r], A[r], dOmat[r], tril, snap, cp, boundary, zero,
                    dQ[r], dKo[r], dVo[r], dgcso[r], dH[r], dCp[r], dSrecv[r], device=r)

                # Reverse ring. Rank P-1 sources it with d = 0; rank 0 terminates it.
                if r == P - 1:
                    dst_r, dgam_r = self.chip_bwd_source(
                        dsr, zero, zerov, dStot[r], dgam[r], bdst, bsig, r - 1, device=r)
                elif r == 0:
                    dst_r, dgam_r = self.chip_bwd_terminal(
                        zerov, dStot[r], dgam[r], bdst, bsig, device=r)
                else:
                    dst_r, dgam_r = self.chip_bwd_middle(
                        dsr, gammas[r], boundary, dStot[r], dgam[r], bdst, bsig, r - 1,
                        device=r)

                self.chip_grad_h(
                    Kmat[r], Vmat[r], A[r], tril, triu, snap, cp, dh, dcp, dko, dvo, dgo,
                    dst_r, dgam_r, dK[r], dV[r], dA[r], device=r)
            return dQ, dK, dV, dA

    return FusedBackwardProgram
