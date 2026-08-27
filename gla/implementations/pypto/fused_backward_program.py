"""Fully-fused distributed PyPTO ZeCO **backward**: five phases in ONE program (B4).

The mirror of :mod:`.fused_program`. Per rank ``r`` (device ``r``) the single ``host_orch``
runs, with no host round-trip between any of them:

1. **recompute** (InCore) — re-runs the forward chunk scan, but *recording* what the two
   adjoint kernels re-read: the per-chunk pre-state ``S_prev[n]``, the cumulative decay
   ``c_prev[n]``, and the within-chunk decay ``b = exp(tril @ log A)``. Also emits
   ``S_total``, the forward ring's input. (Activations are recomputed rather than carried:
   :meth:`gla.common.ZeCoImpl.backward` is stateless by contract, and ``[N,dk,dv]`` of
   snapshots is cheaper to regenerate than to ship.)
2. **AllScan ring** (first/middle/last) — the same forward boundary scan as the forward
   program, giving each rank its ``S_recv``. Needed twice over: ``grad_o`` reconstructs
   ``H_n = S_prev[n] + c_prev[n]*S_recv``, and the reverse ring reduces ``dgamma`` against it.
3. **grad_o** (InCore) — the output-stage adjoints, one pass over chunks in forward order:
   ``dQ``, the intra-chunk halves of ``dK``/``dV``, the log-domain gate grad ``dg_cs``, the
   per-chunk ``dH_n``/``dc_prev[n]`` that feed the state stage, and the accumulated
   ``dS_recv`` that feeds the reverse ring.
4. **reverse ring** (source/middle/terminal) — the adjoint of the boundary scan, flowing
   ``r -> r-1``, producing ``dS_total[r]`` and ``dgamma[r]``.
5. **grad_h** (InCore) — the reverse chunk recurrence. It *records* the state adjoint
   ``dSloc`` and decay adjoint ``dcvec`` per chunk rather than carrying them through the
   work, then adds the state-path halves of ``dK``/``dV`` and finishes the gate backward
   into ``dA`` from those records.

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

Vector-buffer budget (A6)
-------------------------
``grad_o`` used to be the widest kernel in either direction — three ``[C,C]`` tiles, ~12
``[C,dk]`` row tiles and ~6 ``[dk,dv]`` state tiles live at once — which capped the whole
backward at ``C<=32, D<=64`` while the forward reached ``C=128, D=1024``. All four of the
forward's levers apply here too, plus two things the forward never needed:

* **no carry.** ``recompute``'s state and ``grad_h``'s reverse walk are both diagonal in the
  head and value dims, so each blocks to ``[BK, BV]`` once its block loops sit OUTSIDE the
  chunk scan. ``grad_o``'s only carry was ``dS_recv = sum_n dH[n]*c_prev[n]``, a plain sum
  over chunks: it moved to a small pass of its own, so the main pass carries nothing at all
  and its chunks are mutually independent.
* **head / value / key-row blocking** exactly as the forward: nothing contracts over the
  value dim, both ``[C,C]`` products (the within-chunk decay and the score matrix) block over
  their contraction axis with ``tril[:, r]`` carrying the causal zeros, and the reverse
  cumulative sum blocks the same way with ``triu[:, r]``.
* **more passes, each ordered by its own reduction.** The backward's outputs reduce over
  DIFFERENT axes — ``dV`` sums over head blocks, ``dQ``/``dK``/``dA`` over value blocks — and
  this kernel blocks both. One loop nest would have to hold one of those sums across its
  outer loop, which is the very tile being split. So ``grad_o`` and ``grad_h`` each run three
  passes over the chunks, and every write is a plain store: nothing accumulates through GM.
* **the decay is recorded, not recomputed.** All three kernels need ``b``, it costs two
  matmuls per (chunk, head block) to build, and once it is a plain load the adjoint kernels
  can order their loops by what their reductions want instead of by what the decay costs.

**One rule the DSL does not enforce**: seed every accumulator from a zero TENSOR, never from
``x * 0.0``. A multiply is vector work, and when the accumulator it seeds is consumed by
matmuls the core splitter may place that multiply on the CUBE half, which has no vector unit;
the DEVICE build then fails on ``set_vector_mask`` long after ``ir.compile`` has returned
success, and a2a3sim does not catch it either. The same hazard rules out GM->GM copy passes.
It is not fully avoidable — at some blockings the splitter puts even a zero-tensor LOAD on the
cube — so :func:`.fused_program.compile_blocked` inspects the generated cube half and rejects
such a plan the way it rejects one that overflows the buffer.

The reachable set is measured rather than assumed — see the B4 entry in ROADMAP.md and
``devtools/a6_compile_probe.py``, which maps it with no NPU at all.

Every distributed / HCCL run must set ``LD_PRELOAD=<cann>/lib64/libhccl.so``.
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from gla.implementations.pypto.fused_program import compile_blocked


def compile_fused_backward(L, C, dk, dv, P, *, platform, distributed_config=None,
                           plans=None, log=None):
    """Compile the fused backward at the cheapest blocking that fits the vector buffer.

    Same five levers and the same search as the forward (:func:`.fused_program.compile_blocked`)
    -- they compete for the same buffer, and the backward is where losing that search shows up
    as a shape that cannot be run at all rather than merely one that runs slowly.

    Returns ``(compiled, (head_blocks, value_blocks, ring_depth, ring_blocks, key_row_blocks))``.
    """
    from pypto import ir

    def build_one(nb, nv, slot, rb, nc):
        program = build_fused_backward_program(L, C, dk, dv, rb, P, nb, nv, slot, nc)
        kwargs = {"platform": platform}
        if distributed_config is not None:
            kwargs["distributed_config"] = distributed_config
        return ir.compile(program, **kwargs)

    return compile_blocked(build_one, C, dk, dv, distributed=P > 1, plans=plans, log=log,
                           what="fused backward")


def _build_p1_backward_program(L: int, C: int, dk: int, dv: int,
                               nb: int = 1, nv: int = 1, slot: int = 4, nc: int = 1):
    """P == 1 native path: recompute -> grad_o -> grad_h, all from a zero boundary.

    A single rank has no neighbour, so ``S_recv``, ``dS_total`` and ``dgamma`` are all zero
    and both rings are dead. The three kernels are byte-identical to their counterparts in
    :func:`build_fused_backward_program` — see the module docstring for why they cannot be
    shared.
    """
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    N = L // C
    P, DK, DV = 1, dk, dv
    NB, BK, NV, BV, NC, BC, SLOT = nb, dk // nb, nv, dv // nv, nc, C // nc, slot
    # One zero tile source for every accumulator seed. The widest seed is [C, BK] or
    # [C, BC], so it has to be at least as wide as both the head block and the chunk.
    ZW = max(C, dk)

    @pl.program
    class FusedBackwardP1Program:
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": SLOT})
        def gla_recompute(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Cprev: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Bs: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            """Forward chunk scan, recording everything the two adjoint kernels re-read.

            Four outputs: the per-chunk snapshot ``Ssnap[n] = S_n(0)``, the running decay
            ``Cprev[n] = prod_{m<n} gamma_m``, the WITHIN-chunk decay ``Bs = exp(tril @ log A)``
            and the end-of-slice state ``Stot`` that feeds the ring.

            ``Bs`` is recorded rather than recomputed. All three kernels need it, it costs two
            matmuls per (chunk, head block) to build, and once it is a plain load the adjoint
            kernels can order their loops by what their REDUCTIONS want instead of by what the
            decay costs -- which is what lets dV be stored rather than accumulated.

            Head-dim block OUTSIDE the chunk scan, exactly as the forward's stage1: nothing
            here contracts over the head dim (``kb^T @ v`` has it as the OUTPUT row dim), so
            block j's state depends only on block j for the whole slice. That makes the live
            carry [BK, BV] instead of [dk, dv] -- three copies of a [128, 128] carry are
            196608 B of a 188416 B buffer, which no amount of inner blocking can move.
            """
            snaps = Ssnap
            cp = Cprev
            bs = Bs
            tot = Stot
            for j in pl.range(0, NB):
                dof = j * BK
                for w in pl.range(0, NV):
                    vof = w * BV
                    s0 = pl.load(zero, [dof, vof], [BK, BV])
                    c0 = pl.load(onev, [dof, 0], [BK, 1])
                    for n, (s_run, c_run) in pl.range(0, N, init_values=(s0, c0)):
                        off = n * C
                        soff = n * DK
                        # Snapshot BEFORE this chunk's update: that is exactly S_n(0).
                        snaps = pl.store(s_run, [soff + dof, vof], snaps)
                        cp = pl.store(c_run, [soff + dof, 0], cp)
                        k_b = pl.load(Kmat, [off, dof], [C, BK])
                        la_b = pl.log(pl.load(A, [off, dof], [C, BK]))
                        # Decay, blocked over the KEY-ROW (contraction) axis: ``tril[:, r]``
                        # already carries the causal zeros, so each block is an independent
                        # product, no scan is needed, and the MAC count is unchanged.
                        zb = pl.load(zc, [0, 0], [C, BK])
                        for r, (b_acc,) in pl.range(0, NC, init_values=(zb,)):
                            rof = r * BC
                            tril_r = pl.load(tril, [0, rof], [C, BC])
                            la_r = pl.log(pl.load(A, [off + rof, dof], [BC, BK]))
                            b_fin = pl.yield_(
                                pl.add(b_acc, pl.matmul(tril_r, la_r, out_dtype=pl.FP32)))
                        b_b = pl.exp(b_fin)
                        # Written once per value block with the same numbers, like Cprev: a
                        # device conditional to write it only at w == 0 costs more than the
                        # store does.
                        bs = pl.store(b_b, [off, dof], bs)
                        gamma_b = pl.exp(pl.tile.reshape(pl.tile.col_sum(la_b), [BK, 1]))
                        kbt_b = pl.transpose(pl.div(k_b, b_b), 0, 1)
                        # (K/b)^T @ V contracts over the chunk rows too, so it blocks the same
                        # way and for the same reason.
                        zkv = pl.load(zero, [dof, vof], [BK, BV])
                        for r2, (kv_acc,) in pl.range(0, NC, init_values=(zkv,)):
                            rof2 = r2 * BC
                            v_r = pl.load(Vmat, [off + rof2, vof], [BC, BV],
                                          target_memory=pl.MemorySpace.Mat)
                            kbt_r = pl.tile.slice(kbt_b, [BK, BC], [0, rof2])
                            kv_b = pl.yield_(
                                pl.add(kv_acc, pl.matmul(kbt_r, v_r, out_dtype=pl.FP32)))
                        # S = gamma * (S + (K/b)^T @ V), exactly as the reference factors it.
                        s_new = pl.tile.row_expand_mul(pl.add(s_run, kv_b), gamma_b)
                        c_new = pl.mul(c_run, gamma_b)
                        s_fin, c_fin = pl.yield_(s_new, c_new)
                    tot = pl.store(s_fin, [dof, vof], tot)
            return snaps, cp, bs, tot
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": SLOT})
        def gla_grad_o(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            dOmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
            dQ: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dKo: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dVo: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dgcso: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dH: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCp: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            dSrecv: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            """Output-stage adjoints, in three passes over the chunks::

                H_n     = S_prev[n] + c_prev[n]*S_recv     reconstruct's inter-chunk history
                dQt     = dO @ H_n^T  +  dsc @ (k/b)
                dH_n    = (q*b)^T @ dO                     -> the state stage's dS_prev[n]
                dsc     = (dO @ v^T) * tril                intra-chunk masked attention
                dV_o    = scores^T @ dO ,  dK_o = (dsc^T @ (q*b)) / b
                dg_cs_o = dQ*q - dK_o*k                    log-domain gate grad

            Three passes because the outputs reduce over DIFFERENT axes and this kernel blocks
            both: dQ and dK sum over value blocks, dV sums over head blocks. One loop nest
            would have to hold one of those sums across its outer loop -- which is the very
            tile being split -- so each pass orders its loops for its own reduction, and every
            write is a plain store.

            The within-chunk decay is READ from ``Bs`` rather than recomputed; that is what
            makes a second and third pass cheap enough to be worth having.
            """
            oq = dQ
            ok = dKo
            ov = dVo
            og = dgcso
            oh = dH
            oc = dCp
            # ---- pass 1: dQ, dK_o, dg_cs, dH, dc_prev. Head blocks outside, so the value
            # sums that dQ and dK need land in tiles.
            for n in pl.range(0, N):
                off = n * C
                soff = n * DK
                for j in pl.range(0, NB):
                    dof = j * BK
                    q_b = pl.load(Q, [off, dof], [C, BK])
                    k_b = pl.load(Kmat, [off, dof], [C, BK])
                    b_b = pl.load(Bs, [off, dof], [C, BK])
                    qt_b = pl.mul(q_b, b_b)
                    kb_b = pl.div(k_b, b_b)
                    cprev = pl.load(Cprev, [soff + dof, 0], [BK, 1])
                    # Accumulators are seeded from a zero TENSOR, never from ``x * 0.0``: a
                    # multiply is vector work, and when the accumulator it seeds is consumed
                    # by matmuls the splitter may put that multiply on the CUBE half, which
                    # has no vector unit. Loading zeros is what the forward does.
                    dq0 = pl.load(zc, [0, 0], [C, BK])
                    dc0 = pl.load(zerov, [0, 0], [BK, 1])
                    # ---- inter-chunk half: dQ's state term, dH_n and dc_prev[n]. Nothing
                    # here touches the key rows, so it is a plain sum over value blocks.
                    for w, (dq_a, dc_a) in pl.range(0, NV, init_values=(dq0, dc0)):
                        vof = w * BV
                        do_w = pl.load(dOmat, [off, vof], [C, BV])
                        srecv = pl.load(Srecv, [dof, vof], [BK, BV])
                        hmat = pl.add(pl.load(Ssnap, [soff + dof, vof], [BK, BV]),
                                      pl.tile.row_expand_mul(srecv, cprev))
                        dq_n = pl.add(dq_a, pl.matmul(do_w, pl.transpose(hmat, 0, 1),
                                                      out_dtype=pl.FP32))
                        dh_n = pl.matmul(pl.transpose(qt_b, 0, 1), do_w, out_dtype=pl.FP32)
                        oh = pl.store(dh_n, [soff + dof, vof], oh)
                        tmp = pl.tile.create([BK, BV], pl.FP32)
                        dc_n = pl.add(dc_a, pl.row_sum(pl.mul(dh_n, srecv), tmp))
                        dq_s, dc_s = pl.yield_(dq_n, dc_n)
                    oc = pl.store(dc_s, [soff + dof, 0], oc)
                    # ---- within-chunk half. Key-row blocks OUTSIDE value blocks: a key
                    # block's dK is finished once its value sum finishes, so it is stored
                    # straight out and never needs a [C, dk] accumulator.
                    for r2, (dq_b,) in pl.range(0, NC, init_values=(dq_s,)):
                        rof2 = r2 * BC
                        tril_2 = pl.load(tril, [0, rof2], [C, BC])
                        kb_r = pl.tile.slice(kb_b, [BC, BK], [rof2, 0])
                        b_r = pl.tile.slice(b_b, [BC, BK], [rof2, 0])
                        dk0 = pl.load(zc, [0, 0], [BC, BK])
                        for w2, (dq_c, dk_c) in pl.range(0, NV, init_values=(dq_b, dk0)):
                            vof2 = w2 * BV
                            do_w2 = pl.load(dOmat, [off, vof2], [C, BV])
                            v_r = pl.load(Vmat, [off + rof2, vof2], [BC, BV])
                            dsc = pl.mul(pl.matmul(do_w2, pl.transpose(v_r, 0, 1),
                                                   out_dtype=pl.FP32), tril_2)
                            dq_d = pl.add(dq_c, pl.matmul(dsc, kb_r, out_dtype=pl.FP32))
                            dk_d = pl.add(dk_c, pl.matmul(pl.transpose(dsc, 0, 1), qt_b,
                                                          out_dtype=pl.FP32))
                            dq_e, dk_e = pl.yield_(dq_d, dk_d)
                        ok = pl.store(pl.div(dk_e, b_r), [off + rof2, dof], ok)
                        dq_f = pl.yield_(dq_e)
                    dq_out = pl.mul(dq_f, b_b)
                    oq = pl.store(dq_out, [off, dof], oq)
                    # dg_cs = dQ*q - dK_o*k needs this head block's dK whole, and the key-row
                    # loop wrote it out in pieces; read it back rather than keeping a [C, BK]
                    # accumulator alive across that loop.
                    dko_b = pl.load(ok, [off, dof], [C, BK])
                    og = pl.store(pl.sub(pl.mul(dq_out, q_b), pl.mul(dko_b, k_b)),
                                  [off, dof], og)
            # ---- pass 2: dV_o. HEAD blocks innermost, so the score block's head-block sum is
            # a tile accumulation and each key-row block's dV is complete when it is stored.
            # The mask is applied once, to the finished sum -- masking is elementwise, hence
            # linear, so no [C, C] score matrix is ever built.
            for n2 in pl.range(0, N):
                of2 = n2 * C
                for r3 in pl.range(0, NC):
                    rf3 = r3 * BC
                    sc0 = pl.load(zc, [0, 0], [C, BC])
                    for j3, (sc_a,) in pl.range(0, NB, init_values=(sc0,)):
                        df3 = j3 * BK
                        b_3 = pl.load(Bs, [of2, df3], [C, BK])
                        qt_3 = pl.mul(pl.load(Q, [of2, df3], [C, BK]), b_3)
                        kb_3 = pl.div(pl.load(Kmat, [of2, df3], [C, BK]), b_3)
                        kbt_3 = pl.tile.slice(pl.transpose(kb_3, 0, 1), [BK, BC], [0, rf3])
                        sc_f = pl.yield_(
                            pl.add(sc_a, pl.matmul(qt_3, kbt_3, out_dtype=pl.FP32)))
                    sc_m = pl.mul(sc_f, pl.load(tril, [0, rf3], [C, BC]))
                    sct = pl.transpose(sc_m, 0, 1)
                    for w3 in pl.range(0, NV):
                        vf3 = w3 * BV
                        do_3 = pl.load(dOmat, [of2, vf3], [C, BV])
                        ov = pl.store(pl.matmul(sct, do_3, out_dtype=pl.FP32),
                                      [of2 + rf3, vf3], ov)
            # ---- pass 3: dS_recv = sum_n dH[n] * c_prev[n]. Kept out of pass 1 so that pass
            # carries no state at all: as a loop carry this is a full [dk, dv] tile live
            # across the whole kernel, which is exactly what A1 removed from the forward.
            osr = dSrecv
            for j2 in pl.range(0, NB):
                df2 = j2 * BK
                for w4 in pl.range(0, NV):
                    vf4 = w4 * BV
                    a0 = pl.load(zero, [df2, vf4], [BK, BV])
                    for n3, (a_acc,) in pl.range(0, N, init_values=(a0,)):
                        sf2 = n3 * DK
                        dh_r = pl.load(oh, [sf2 + df2, vf4], [BK, BV])
                        cp_r = pl.load(Cprev, [sf2 + df2, 0], [BK, 1])
                        a_fin = pl.yield_(pl.add(a_acc, pl.tile.row_expand_mul(dh_r, cp_r)))
                    osr = pl.store(a_fin, [df2, vf4], osr)
            return oq, ok, ov, og, oh, oc, osr
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": SLOT})
        def gla_grad_h(
            self,
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            dH: pl.Tensor[[N * DK, DV], pl.FP32],
            dCp: pl.Tensor[[N * DK, 1], pl.FP32],
            dKo: pl.Tensor[[L, DK], pl.FP32],
            dVo: pl.Tensor[[L, DV], pl.FP32],
            dgcso: pl.Tensor[[L, DK], pl.FP32],
            dStot: pl.Tensor[[DK, DV], pl.FP32],
            dgam: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
            dK: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dV: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dA: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dSl: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCv: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
        ):
            """State-stage adjoints and the gate backward, in three passes.

            Pass 1 RECORDS the reverse recurrence instead of carrying it through the work:
            ``dsloc_n = gamma_n*dsloc_{n+1} + dH_n`` is diagonal in both dims, so the walk
            itself blocks to [BK, BV], and writing each step out frees passes 2 and 3 to order
            their loops by their own reductions -- which a live carry would forbid. Same trick
            as A1 in the forward, run backwards.

            Pass 2 takes dK and dA (value sums, so value blocks are innermost); pass 3 takes
            dV (a head sum, so head blocks are innermost). Both end in a plain store.
            """
            osl = dSl
            ocv = dCv
            okk = dK
            ovv = dV
            oaa = dA
            for j in pl.range(0, NB):
                dof = j * BK
                for w in pl.range(0, NV):
                    vof = w * BV
                    ds0 = pl.load(dStot, [dof, vof], [BK, BV])
                    dc0 = pl.load(dgam, [dof, 0], [BK, 1])
                    for m, (dsloc, dcvec) in pl.range(0, N, init_values=(ds0, dc0)):
                        off = (N - 1 - m) * C
                        soff = (N - 1 - m) * DK
                        la = pl.log(pl.load(A, [off, dof], [C, BK]))
                        gamma = pl.exp(pl.tile.reshape(pl.tile.col_sum(la), [BK, 1]))
                        # Push gamma onto the state ONCE; everything downstream is
                        # broadcast-free. This also detaches the raw iter_arg, which cannot
                        # feed a matmul directly.
                        dsl_p = pl.tile.row_expand_mul(dsloc, gamma)
                        dcv_p = pl.mul(dcvec, gamma)
                        osl = pl.store(dsl_p, [soff + dof, vof], osl)
                        ocv = pl.store(dcv_p, [soff + dof, 0], ocv)
                        dsloc_n = pl.add(dsl_p, pl.load(dH, [soff + dof, vof], [BK, BV]))
                        dcvec_n = pl.add(dcv_p, pl.load(dCp, [soff + dof, 0], [BK, 1]))
                        dsl_f, dcv_f = pl.yield_(dsloc_n, dcvec_n)
            # ---- pass 2: dK's state half and the gate gradient.
            #     dK_h = (v @ (gamma*dS)^T) / b, with the 1/b applied after the value sum.
            for n in pl.range(0, N):
                off = n * C
                soff = n * DK
                for j2 in pl.range(0, NB):
                    dof2 = j2 * BK
                    k_b = pl.load(Kmat, [off, dof2], [C, BK])
                    a_b = pl.load(A, [off, dof2], [C, BK])
                    b_b = pl.load(Bs, [off, dof2], [C, BK])
                    cprev = pl.load(Cprev, [soff + dof2, 0], [BK, 1])
                    dk0 = pl.load(zc, [0, 0], [C, BK])
                    c10 = pl.load(zerov, [0, 0], [BK, 1])
                    for w2, (dk_a, c1_a) in pl.range(0, NV, init_values=(dk0, c10)):
                        vf2 = w2 * BV
                        dsl_p = pl.load(osl, [soff + dof2, vf2], [BK, BV])
                        v_w = pl.load(Vmat, [off, vf2], [C, BV])
                        sprev = pl.load(Ssnap, [soff + dof2, vf2], [BK, BV])
                        dk_n = pl.add(dk_a, pl.matmul(v_w, pl.transpose(dsl_p, 0, 1),
                                                      out_dtype=pl.FP32))
                        tmp = pl.tile.create([BK, BV], pl.FP32)
                        c1_n = pl.add(c1_a, pl.row_sum(pl.mul(dsl_p, sprev), tmp))
                        dk_s, c1_s = pl.yield_(dk_n, c1_n)
                    dk_h = pl.div(dk_s, b_b)
                    dkk = pl.mul(dk_h, k_b)
                    c2 = pl.mul(pl.load(ocv, [soff + dof2, 0], [BK, 1]), cprev)
                    corr = pl.add(pl.add(pl.tile.reshape(c1_s, [1, BK]),
                                         pl.tile.reshape(c2, [1, BK])),
                                  pl.tile.col_sum(dkk))
                    dgcs = pl.sub(pl.load(dgcso, [off, dof2], [C, BK]), dkk)
                    # The per-chunk REVERSE cumulative sum is a matmul by an upper-triangular
                    # ones matrix, not a scan -- and it blocks over its contraction axis for
                    # the same reason tril does: ``triu[:, r]`` carries the zeros.
                    zr = pl.load(zc, [0, 0], [C, BK])
                    for r2, (rc_acc,) in pl.range(0, NC, init_values=(zr,)):
                        rf2 = r2 * BC
                        triu_r = pl.load(triu, [0, rf2], [C, BC])
                        dg_r = pl.tile.slice(dgcs, [BC, BK], [rf2, 0])
                        rc_fin = pl.yield_(
                            pl.add(rc_acc, pl.matmul(triu_r, dg_r, out_dtype=pl.FP32)))
                    oaa = pl.store(pl.div(pl.tile.col_expand_add(rc_fin, corr), a_b),
                                   [off, dof2], oaa)
                    okk = pl.store(pl.add(pl.load(dKo, [off, dof2], [C, BK]), dk_h),
                                   [off, dof2], okk)
            # ---- pass 3: dV = dV_o + sum over HEAD blocks of (k/b) @ (gamma*dS). Head blocks
            # innermost, so the sum is a tile and the write is a store: dV_o seeds it, and
            # nothing accumulates through GM.
            for n2 in pl.range(0, N):
                of2 = n2 * C
                sf2 = n2 * DK
                for w3 in pl.range(0, NV):
                    vf3 = w3 * BV
                    dv0 = pl.load(dVo, [of2, vf3], [C, BV])
                    for j3, (dv_a,) in pl.range(0, NB, init_values=(dv0,)):
                        df3 = j3 * BK
                        kb_3 = pl.div(pl.load(Kmat, [of2, df3], [C, BK]),
                                      pl.load(Bs, [of2, df3], [C, BK]))
                        dsl_3 = pl.load(osl, [sf2 + df3, vf3], [BK, BV])
                        dv_f = pl.yield_(
                            pl.add(dv_a, pl.matmul(kb_3, dsl_3, out_dtype=pl.FP32)))
                    ovv = pl.store(dv_f, [of2, vf3], ovv)
            return okk, ovv, oaa, osl, ocv
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_recompute(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Cprev: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Bs: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.gla_recompute(A, Kmat, Vmat, tril, zero, onev, zc, Ssnap, Cprev, Bs,
                                      Stot)
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_grad_o(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            dOmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
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
            return self.gla_grad_o(Q, Kmat, Vmat, dOmat, tril, Ssnap, Cprev, Bs, Srecv, zero,
                                   zc, zerov, dQ, dKo, dVo, dgcso, dH, dCp, dSrecv)
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_grad_h(
            self,
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            dH: pl.Tensor[[N * DK, DV], pl.FP32],
            dCp: pl.Tensor[[N * DK, 1], pl.FP32],
            dKo: pl.Tensor[[L, DK], pl.FP32],
            dVo: pl.Tensor[[L, DV], pl.FP32],
            dgcso: pl.Tensor[[L, DK], pl.FP32],
            dStot: pl.Tensor[[DK, DV], pl.FP32],
            dgam: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
            dK: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dV: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dA: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dSl: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCv: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DV], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
        ]:
            return self.gla_grad_h(Kmat, Vmat, A, triu, Ssnap, Cprev, Bs, dH, dCp, dKo, dVo,
                                   dgcso, dStot, dgam, zc, zerov, dK, dV, dA, dSl, dCv)
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
            zc: pl.Tensor[[C, ZW], pl.FP32],
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
            dgcso = pl.create_tensor([P, L, dk], dtype=pl.FP32)
            # The state-adjoint walk, recorded per chunk instead of carried (see gla_grad_h).
            dSl = pl.create_tensor([P, N * dk, dv], dtype=pl.FP32)
            dCv = pl.create_tensor([P, N * dk, 1], dtype=pl.FP32)
            dVo = pl.create_tensor([P, L, dv], dtype=pl.FP32)
            Bs = pl.create_tensor([P, L, dk], dtype=pl.FP32)

            # `sl_r` (S_total) and `dsr` (dS_recv) are unpacked but unused, and pypto's
            # UnusedVariableCheck says so: both exist only to feed the rings, which P=1 does
            # not have. They are kept — the tuple element order maps positionally onto the
            # callee's Out params, so an element cannot be dropped from the middle, and
            # keeping the kernels identical to the P>1 pair is worth three warnings.
            for r in pl.range(P):
                snap, cp, bs, sl_r = self.chip_recompute(
                    A[r], Kmat[r], Vmat[r], tril, zero, onev, zc,
                    Ssnap[r], Cprev[r], Bs[r], Stot[r], device=r)
                dq_r, dko, dvo, dgo, dh, dcp, dsr = self.chip_grad_o(
                    Qmat[r], Kmat[r], Vmat[r], dOmat[r], tril, snap, cp, bs, zero, zero,
                    zc, zerov, dQ[r], dKo[r], dVo[r], dgcso[r], dH[r], dCp[r], dSrecv[r],
                    device=r)
                self.chip_grad_h(
                    Kmat[r], Vmat[r], A[r], triu, snap, cp, bs, dh, dcp, dko, dvo, dgo,
                    zero, zerov, zc, zerov, dK[r], dV[r], dA[r], dSl[r], dCv[r], device=r)
            return dQ, dK, dV, dA

    return FusedBackwardP1Program


def build_fused_backward_program(L: int, C: int, dk: int, dv: int, K: int, P: int,
                                 nb: int = 1, nv: int = 1, slot: int = 4, nc: int = 1):
    """Build the fully-fused ``recompute + ring + grad_o + reverse-ring + grad_h`` program.

    Args:
        L: Tokens per device. C: chunk size (``L % C == 0``, ``N = L // C``).
        dk, dv: key/query and value dims. K: how many pieces the boundary exchange is cut
            into (``dk % K == 0``).
        P: ranks / devices. ``P == 1`` builds the native single-rank program (no rings).
        nb, nv, nc: how many pieces the head dim, the value dim and the chunk's key-row axis
            are cut into. slot: the cube<->vector pipe ring depth. Chosen by
            :func:`compile_fused_backward`, which keeps the cheapest setting that fits.

    Returns:
        A ``@pl.program`` whose ``host_orch`` takes ``(Qmat, Kmat, Vmat, A, dOmat, gammas,
        tril, triu, zero, zerov, onev)`` and writes ``(dQ, dK, dV, dA)`` ``[P, L, ...]``.
    """
    assert dk % K == 0, f"dk ({dk}) must be divisible by K ({K})"
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    if P == 1:
        return _build_p1_backward_program(L, C, dk, dv, nb, nv, slot, nc)

    BLOCK = dk // K
    N = L // C
    DK, DV = dk, dv
    NB, BK, NV, BV, NC, BC, SLOT = nb, dk // nb, nv, dv // nv, nc, C // nc, slot
    # One zero tile source for every accumulator seed. The widest seed is [C, BK] or
    # [C, BC], so it has to be at least as wide as both the head block and the chunk.
    ZW = max(C, dk)

    @pl.program
    class FusedBackwardProgram:
        # ---- phase 1: forward recompute with per-chunk snapshots ----
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": SLOT})
        def gla_recompute(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Cprev: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Bs: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            """Forward chunk scan, recording everything the two adjoint kernels re-read.

            Four outputs: the per-chunk snapshot ``Ssnap[n] = S_n(0)``, the running decay
            ``Cprev[n] = prod_{m<n} gamma_m``, the WITHIN-chunk decay ``Bs = exp(tril @ log A)``
            and the end-of-slice state ``Stot`` that feeds the ring.

            ``Bs`` is recorded rather than recomputed. All three kernels need it, it costs two
            matmuls per (chunk, head block) to build, and once it is a plain load the adjoint
            kernels can order their loops by what their REDUCTIONS want instead of by what the
            decay costs -- which is what lets dV be stored rather than accumulated.

            Head-dim block OUTSIDE the chunk scan, exactly as the forward's stage1: nothing
            here contracts over the head dim (``kb^T @ v`` has it as the OUTPUT row dim), so
            block j's state depends only on block j for the whole slice. That makes the live
            carry [BK, BV] instead of [dk, dv] -- three copies of a [128, 128] carry are
            196608 B of a 188416 B buffer, which no amount of inner blocking can move.
            """
            snaps = Ssnap
            cp = Cprev
            bs = Bs
            tot = Stot
            for j in pl.range(0, NB):
                dof = j * BK
                for w in pl.range(0, NV):
                    vof = w * BV
                    s0 = pl.load(zero, [dof, vof], [BK, BV])
                    c0 = pl.load(onev, [dof, 0], [BK, 1])
                    for n, (s_run, c_run) in pl.range(0, N, init_values=(s0, c0)):
                        off = n * C
                        soff = n * DK
                        # Snapshot BEFORE this chunk's update: that is exactly S_n(0).
                        snaps = pl.store(s_run, [soff + dof, vof], snaps)
                        cp = pl.store(c_run, [soff + dof, 0], cp)
                        k_b = pl.load(Kmat, [off, dof], [C, BK])
                        la_b = pl.log(pl.load(A, [off, dof], [C, BK]))
                        # Decay, blocked over the KEY-ROW (contraction) axis: ``tril[:, r]``
                        # already carries the causal zeros, so each block is an independent
                        # product, no scan is needed, and the MAC count is unchanged.
                        zb = pl.load(zc, [0, 0], [C, BK])
                        for r, (b_acc,) in pl.range(0, NC, init_values=(zb,)):
                            rof = r * BC
                            tril_r = pl.load(tril, [0, rof], [C, BC])
                            la_r = pl.log(pl.load(A, [off + rof, dof], [BC, BK]))
                            b_fin = pl.yield_(
                                pl.add(b_acc, pl.matmul(tril_r, la_r, out_dtype=pl.FP32)))
                        b_b = pl.exp(b_fin)
                        # Written once per value block with the same numbers, like Cprev: a
                        # device conditional to write it only at w == 0 costs more than the
                        # store does.
                        bs = pl.store(b_b, [off, dof], bs)
                        gamma_b = pl.exp(pl.tile.reshape(pl.tile.col_sum(la_b), [BK, 1]))
                        kbt_b = pl.transpose(pl.div(k_b, b_b), 0, 1)
                        # (K/b)^T @ V contracts over the chunk rows too, so it blocks the same
                        # way and for the same reason.
                        zkv = pl.load(zero, [dof, vof], [BK, BV])
                        for r2, (kv_acc,) in pl.range(0, NC, init_values=(zkv,)):
                            rof2 = r2 * BC
                            v_r = pl.load(Vmat, [off + rof2, vof], [BC, BV],
                                          target_memory=pl.MemorySpace.Mat)
                            kbt_r = pl.tile.slice(kbt_b, [BK, BC], [0, rof2])
                            kv_b = pl.yield_(
                                pl.add(kv_acc, pl.matmul(kbt_r, v_r, out_dtype=pl.FP32)))
                        # S = gamma * (S + (K/b)^T @ V), exactly as the reference factors it.
                        s_new = pl.tile.row_expand_mul(pl.add(s_run, kv_b), gamma_b)
                        c_new = pl.mul(c_run, gamma_b)
                        s_fin, c_fin = pl.yield_(s_new, c_new)
                    tot = pl.store(s_fin, [dof, vof], tot)
            return snaps, cp, bs, tot
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_recompute(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            onev: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            Ssnap: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            Cprev: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            Bs: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[DK, DV], pl.FP32],
        ]:
            return self.gla_recompute(A, Kmat, Vmat, tril, zero, onev, zc, Ssnap, Cprev, Bs,
                                      Stot)
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": SLOT})
        def gla_grad_o(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            dOmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
            dQ: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dKo: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dVo: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dgcso: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dH: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCp: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
            dSrecv: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
        ):
            """Output-stage adjoints, in three passes over the chunks::

                H_n     = S_prev[n] + c_prev[n]*S_recv     reconstruct's inter-chunk history
                dQt     = dO @ H_n^T  +  dsc @ (k/b)
                dH_n    = (q*b)^T @ dO                     -> the state stage's dS_prev[n]
                dsc     = (dO @ v^T) * tril                intra-chunk masked attention
                dV_o    = scores^T @ dO ,  dK_o = (dsc^T @ (q*b)) / b
                dg_cs_o = dQ*q - dK_o*k                    log-domain gate grad

            Three passes because the outputs reduce over DIFFERENT axes and this kernel blocks
            both: dQ and dK sum over value blocks, dV sums over head blocks. One loop nest
            would have to hold one of those sums across its outer loop -- which is the very
            tile being split -- so each pass orders its loops for its own reduction, and every
            write is a plain store.

            The within-chunk decay is READ from ``Bs`` rather than recomputed; that is what
            makes a second and third pass cheap enough to be worth having.
            """
            oq = dQ
            ok = dKo
            ov = dVo
            og = dgcso
            oh = dH
            oc = dCp
            # ---- pass 1: dQ, dK_o, dg_cs, dH, dc_prev. Head blocks outside, so the value
            # sums that dQ and dK need land in tiles.
            for n in pl.range(0, N):
                off = n * C
                soff = n * DK
                for j in pl.range(0, NB):
                    dof = j * BK
                    q_b = pl.load(Q, [off, dof], [C, BK])
                    k_b = pl.load(Kmat, [off, dof], [C, BK])
                    b_b = pl.load(Bs, [off, dof], [C, BK])
                    qt_b = pl.mul(q_b, b_b)
                    kb_b = pl.div(k_b, b_b)
                    cprev = pl.load(Cprev, [soff + dof, 0], [BK, 1])
                    # Accumulators are seeded from a zero TENSOR, never from ``x * 0.0``: a
                    # multiply is vector work, and when the accumulator it seeds is consumed
                    # by matmuls the splitter may put that multiply on the CUBE half, which
                    # has no vector unit. Loading zeros is what the forward does.
                    dq0 = pl.load(zc, [0, 0], [C, BK])
                    dc0 = pl.load(zerov, [0, 0], [BK, 1])
                    # ---- inter-chunk half: dQ's state term, dH_n and dc_prev[n]. Nothing
                    # here touches the key rows, so it is a plain sum over value blocks.
                    for w, (dq_a, dc_a) in pl.range(0, NV, init_values=(dq0, dc0)):
                        vof = w * BV
                        do_w = pl.load(dOmat, [off, vof], [C, BV])
                        srecv = pl.load(Srecv, [dof, vof], [BK, BV])
                        hmat = pl.add(pl.load(Ssnap, [soff + dof, vof], [BK, BV]),
                                      pl.tile.row_expand_mul(srecv, cprev))
                        dq_n = pl.add(dq_a, pl.matmul(do_w, pl.transpose(hmat, 0, 1),
                                                      out_dtype=pl.FP32))
                        dh_n = pl.matmul(pl.transpose(qt_b, 0, 1), do_w, out_dtype=pl.FP32)
                        oh = pl.store(dh_n, [soff + dof, vof], oh)
                        tmp = pl.tile.create([BK, BV], pl.FP32)
                        dc_n = pl.add(dc_a, pl.row_sum(pl.mul(dh_n, srecv), tmp))
                        dq_s, dc_s = pl.yield_(dq_n, dc_n)
                    oc = pl.store(dc_s, [soff + dof, 0], oc)
                    # ---- within-chunk half. Key-row blocks OUTSIDE value blocks: a key
                    # block's dK is finished once its value sum finishes, so it is stored
                    # straight out and never needs a [C, dk] accumulator.
                    for r2, (dq_b,) in pl.range(0, NC, init_values=(dq_s,)):
                        rof2 = r2 * BC
                        tril_2 = pl.load(tril, [0, rof2], [C, BC])
                        kb_r = pl.tile.slice(kb_b, [BC, BK], [rof2, 0])
                        b_r = pl.tile.slice(b_b, [BC, BK], [rof2, 0])
                        dk0 = pl.load(zc, [0, 0], [BC, BK])
                        for w2, (dq_c, dk_c) in pl.range(0, NV, init_values=(dq_b, dk0)):
                            vof2 = w2 * BV
                            do_w2 = pl.load(dOmat, [off, vof2], [C, BV])
                            v_r = pl.load(Vmat, [off + rof2, vof2], [BC, BV])
                            dsc = pl.mul(pl.matmul(do_w2, pl.transpose(v_r, 0, 1),
                                                   out_dtype=pl.FP32), tril_2)
                            dq_d = pl.add(dq_c, pl.matmul(dsc, kb_r, out_dtype=pl.FP32))
                            dk_d = pl.add(dk_c, pl.matmul(pl.transpose(dsc, 0, 1), qt_b,
                                                          out_dtype=pl.FP32))
                            dq_e, dk_e = pl.yield_(dq_d, dk_d)
                        ok = pl.store(pl.div(dk_e, b_r), [off + rof2, dof], ok)
                        dq_f = pl.yield_(dq_e)
                    dq_out = pl.mul(dq_f, b_b)
                    oq = pl.store(dq_out, [off, dof], oq)
                    # dg_cs = dQ*q - dK_o*k needs this head block's dK whole, and the key-row
                    # loop wrote it out in pieces; read it back rather than keeping a [C, BK]
                    # accumulator alive across that loop.
                    dko_b = pl.load(ok, [off, dof], [C, BK])
                    og = pl.store(pl.sub(pl.mul(dq_out, q_b), pl.mul(dko_b, k_b)),
                                  [off, dof], og)
            # ---- pass 2: dV_o. HEAD blocks innermost, so the score block's head-block sum is
            # a tile accumulation and each key-row block's dV is complete when it is stored.
            # The mask is applied once, to the finished sum -- masking is elementwise, hence
            # linear, so no [C, C] score matrix is ever built.
            for n2 in pl.range(0, N):
                of2 = n2 * C
                for r3 in pl.range(0, NC):
                    rf3 = r3 * BC
                    sc0 = pl.load(zc, [0, 0], [C, BC])
                    for j3, (sc_a,) in pl.range(0, NB, init_values=(sc0,)):
                        df3 = j3 * BK
                        b_3 = pl.load(Bs, [of2, df3], [C, BK])
                        qt_3 = pl.mul(pl.load(Q, [of2, df3], [C, BK]), b_3)
                        kb_3 = pl.div(pl.load(Kmat, [of2, df3], [C, BK]), b_3)
                        kbt_3 = pl.tile.slice(pl.transpose(kb_3, 0, 1), [BK, BC], [0, rf3])
                        sc_f = pl.yield_(
                            pl.add(sc_a, pl.matmul(qt_3, kbt_3, out_dtype=pl.FP32)))
                    sc_m = pl.mul(sc_f, pl.load(tril, [0, rf3], [C, BC]))
                    sct = pl.transpose(sc_m, 0, 1)
                    for w3 in pl.range(0, NV):
                        vf3 = w3 * BV
                        do_3 = pl.load(dOmat, [of2, vf3], [C, BV])
                        ov = pl.store(pl.matmul(sct, do_3, out_dtype=pl.FP32),
                                      [of2 + rf3, vf3], ov)
            # ---- pass 3: dS_recv = sum_n dH[n] * c_prev[n]. Kept out of pass 1 so that pass
            # carries no state at all: as a loop carry this is a full [dk, dv] tile live
            # across the whole kernel, which is exactly what A1 removed from the forward.
            osr = dSrecv
            for j2 in pl.range(0, NB):
                df2 = j2 * BK
                for w4 in pl.range(0, NV):
                    vf4 = w4 * BV
                    a0 = pl.load(zero, [df2, vf4], [BK, BV])
                    for n3, (a_acc,) in pl.range(0, N, init_values=(a0,)):
                        sf2 = n3 * DK
                        dh_r = pl.load(oh, [sf2 + df2, vf4], [BK, BV])
                        cp_r = pl.load(Cprev, [sf2 + df2, 0], [BK, 1])
                        a_fin = pl.yield_(pl.add(a_acc, pl.tile.row_expand_mul(dh_r, cp_r)))
                    osr = pl.store(a_fin, [df2, vf4], osr)
            return oq, ok, ov, og, oh, oc, osr
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_grad_o(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            dOmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
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
            return self.gla_grad_o(Q, Kmat, Vmat, dOmat, tril, Ssnap, Cprev, Bs, Srecv, zero,
                                   zc, zerov, dQ, dKo, dVo, dgcso, dH, dCp, dSrecv)
        @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": SLOT})
        def gla_grad_h(
            self,
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            dH: pl.Tensor[[N * DK, DV], pl.FP32],
            dCp: pl.Tensor[[N * DK, 1], pl.FP32],
            dKo: pl.Tensor[[L, DK], pl.FP32],
            dVo: pl.Tensor[[L, DV], pl.FP32],
            dgcso: pl.Tensor[[L, DK], pl.FP32],
            dStot: pl.Tensor[[DK, DV], pl.FP32],
            dgam: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
            dK: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dV: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dA: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dSl: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCv: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
        ):
            """State-stage adjoints and the gate backward, in three passes.

            Pass 1 RECORDS the reverse recurrence instead of carrying it through the work:
            ``dsloc_n = gamma_n*dsloc_{n+1} + dH_n`` is diagonal in both dims, so the walk
            itself blocks to [BK, BV], and writing each step out frees passes 2 and 3 to order
            their loops by their own reductions -- which a live carry would forbid. Same trick
            as A1 in the forward, run backwards.

            Pass 2 takes dK and dA (value sums, so value blocks are innermost); pass 3 takes
            dV (a head sum, so head blocks are innermost). Both end in a plain store.
            """
            osl = dSl
            ocv = dCv
            okk = dK
            ovv = dV
            oaa = dA
            for j in pl.range(0, NB):
                dof = j * BK
                for w in pl.range(0, NV):
                    vof = w * BV
                    ds0 = pl.load(dStot, [dof, vof], [BK, BV])
                    dc0 = pl.load(dgam, [dof, 0], [BK, 1])
                    for m, (dsloc, dcvec) in pl.range(0, N, init_values=(ds0, dc0)):
                        off = (N - 1 - m) * C
                        soff = (N - 1 - m) * DK
                        la = pl.log(pl.load(A, [off, dof], [C, BK]))
                        gamma = pl.exp(pl.tile.reshape(pl.tile.col_sum(la), [BK, 1]))
                        # Push gamma onto the state ONCE; everything downstream is
                        # broadcast-free. This also detaches the raw iter_arg, which cannot
                        # feed a matmul directly.
                        dsl_p = pl.tile.row_expand_mul(dsloc, gamma)
                        dcv_p = pl.mul(dcvec, gamma)
                        osl = pl.store(dsl_p, [soff + dof, vof], osl)
                        ocv = pl.store(dcv_p, [soff + dof, 0], ocv)
                        dsloc_n = pl.add(dsl_p, pl.load(dH, [soff + dof, vof], [BK, BV]))
                        dcvec_n = pl.add(dcv_p, pl.load(dCp, [soff + dof, 0], [BK, 1]))
                        dsl_f, dcv_f = pl.yield_(dsloc_n, dcvec_n)
            # ---- pass 2: dK's state half and the gate gradient.
            #     dK_h = (v @ (gamma*dS)^T) / b, with the 1/b applied after the value sum.
            for n in pl.range(0, N):
                off = n * C
                soff = n * DK
                for j2 in pl.range(0, NB):
                    dof2 = j2 * BK
                    k_b = pl.load(Kmat, [off, dof2], [C, BK])
                    a_b = pl.load(A, [off, dof2], [C, BK])
                    b_b = pl.load(Bs, [off, dof2], [C, BK])
                    cprev = pl.load(Cprev, [soff + dof2, 0], [BK, 1])
                    dk0 = pl.load(zc, [0, 0], [C, BK])
                    c10 = pl.load(zerov, [0, 0], [BK, 1])
                    for w2, (dk_a, c1_a) in pl.range(0, NV, init_values=(dk0, c10)):
                        vf2 = w2 * BV
                        dsl_p = pl.load(osl, [soff + dof2, vf2], [BK, BV])
                        v_w = pl.load(Vmat, [off, vf2], [C, BV])
                        sprev = pl.load(Ssnap, [soff + dof2, vf2], [BK, BV])
                        dk_n = pl.add(dk_a, pl.matmul(v_w, pl.transpose(dsl_p, 0, 1),
                                                      out_dtype=pl.FP32))
                        tmp = pl.tile.create([BK, BV], pl.FP32)
                        c1_n = pl.add(c1_a, pl.row_sum(pl.mul(dsl_p, sprev), tmp))
                        dk_s, c1_s = pl.yield_(dk_n, c1_n)
                    dk_h = pl.div(dk_s, b_b)
                    dkk = pl.mul(dk_h, k_b)
                    c2 = pl.mul(pl.load(ocv, [soff + dof2, 0], [BK, 1]), cprev)
                    corr = pl.add(pl.add(pl.tile.reshape(c1_s, [1, BK]),
                                         pl.tile.reshape(c2, [1, BK])),
                                  pl.tile.col_sum(dkk))
                    dgcs = pl.sub(pl.load(dgcso, [off, dof2], [C, BK]), dkk)
                    # The per-chunk REVERSE cumulative sum is a matmul by an upper-triangular
                    # ones matrix, not a scan -- and it blocks over its contraction axis for
                    # the same reason tril does: ``triu[:, r]`` carries the zeros.
                    zr = pl.load(zc, [0, 0], [C, BK])
                    for r2, (rc_acc,) in pl.range(0, NC, init_values=(zr,)):
                        rf2 = r2 * BC
                        triu_r = pl.load(triu, [0, rf2], [C, BC])
                        dg_r = pl.tile.slice(dgcs, [BC, BK], [rf2, 0])
                        rc_fin = pl.yield_(
                            pl.add(rc_acc, pl.matmul(triu_r, dg_r, out_dtype=pl.FP32)))
                    oaa = pl.store(pl.div(pl.tile.col_expand_add(rc_fin, corr), a_b),
                                   [off, dof2], oaa)
                    okk = pl.store(pl.add(pl.load(dKo, [off, dof2], [C, BK]), dk_h),
                                   [off, dof2], okk)
            # ---- pass 3: dV = dV_o + sum over HEAD blocks of (k/b) @ (gamma*dS). Head blocks
            # innermost, so the sum is a tile and the write is a store: dV_o seeds it, and
            # nothing accumulates through GM.
            for n2 in pl.range(0, N):
                of2 = n2 * C
                sf2 = n2 * DK
                for w3 in pl.range(0, NV):
                    vf3 = w3 * BV
                    dv0 = pl.load(dVo, [of2, vf3], [C, BV])
                    for j3, (dv_a,) in pl.range(0, NB, init_values=(dv0,)):
                        df3 = j3 * BK
                        kb_3 = pl.div(pl.load(Kmat, [of2, df3], [C, BK]),
                                      pl.load(Bs, [of2, df3], [C, BK]))
                        dsl_3 = pl.load(osl, [sf2 + df3, vf3], [BK, BV])
                        dv_f = pl.yield_(
                            pl.add(dv_a, pl.matmul(kb_3, dsl_3, out_dtype=pl.FP32)))
                    ovv = pl.store(dv_f, [of2, vf3], ovv)
            return okk, ovv, oaa, osl, ocv
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_grad_h(
            self,
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            triu: pl.Tensor[[C, C], pl.FP32],
            Ssnap: pl.Tensor[[N * DK, DV], pl.FP32],
            Cprev: pl.Tensor[[N * DK, 1], pl.FP32],
            Bs: pl.Tensor[[L, DK], pl.FP32],
            dH: pl.Tensor[[N * DK, DV], pl.FP32],
            dCp: pl.Tensor[[N * DK, 1], pl.FP32],
            dKo: pl.Tensor[[L, DK], pl.FP32],
            dVo: pl.Tensor[[L, DV], pl.FP32],
            dgcso: pl.Tensor[[L, DK], pl.FP32],
            dStot: pl.Tensor[[DK, DV], pl.FP32],
            dgam: pl.Tensor[[DK, 1], pl.FP32],
            zc: pl.Tensor[[C, ZW], pl.FP32],
            zerov: pl.Tensor[[DK, 1], pl.FP32],
            dK: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dV: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
            dA: pl.Out[pl.Tensor[[L, DK], pl.FP32]],
            dSl: pl.Out[pl.Tensor[[N * DK, DV], pl.FP32]],
            dCv: pl.Out[pl.Tensor[[N * DK, 1], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[L, DV], pl.FP32],
            pl.Tensor[[L, DK], pl.FP32],
            pl.Tensor[[N * DK, DV], pl.FP32],
            pl.Tensor[[N * DK, 1], pl.FP32],
        ]:
            return self.gla_grad_h(Kmat, Vmat, A, triu, Ssnap, Cprev, Bs, dH, dCp, dKo, dVo,
                                   dgcso, dStot, dgam, zc, zerov, dK, dV, dA, dSl, dCv)
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
            zc: pl.Tensor[[C, ZW], pl.FP32],
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
            dgcso = pl.create_tensor([P, L, dk], dtype=pl.FP32)
            dStot = pl.create_tensor([P, dk, dv], dtype=pl.FP32)
            dgam = pl.create_tensor([P, dk, 1], dtype=pl.FP32)
            # The state-adjoint walk, recorded per chunk instead of carried (see gla_grad_h).
            dSl = pl.create_tensor([P, N * dk, dv], dtype=pl.FP32)
            dCv = pl.create_tensor([P, N * dk, 1], dtype=pl.FP32)
            dVo = pl.create_tensor([P, L, dv], dtype=pl.FP32)
            Bs = pl.create_tensor([P, L, dk], dtype=pl.FP32)

            for r in pl.range(P):
                fdst = pld.window(fdst_buf, [dk, dv], dtype=pl.FP32)
                fsig = pld.window(fsig_buf, [K, 1], dtype=pl.INT32)
                bdst = pld.window(bdst_buf, [dk, dv], dtype=pl.FP32)
                bsig = pld.window(bsig_buf, [K, 1], dtype=pl.INT32)

                snap, cp, bs, sl_r = self.chip_recompute(
                    A[r], Kmat[r], Vmat[r], tril, zero, onev, zc,
                    Ssnap[r], Cprev[r], Bs[r], Stot[r], device=r)

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
                    Qmat[r], Kmat[r], Vmat[r], dOmat[r], tril, snap, cp, bs, boundary, zero,
                    zc, zerov, dQ[r], dKo[r], dVo[r], dgcso[r], dH[r], dCp[r], dSrecv[r],
                    device=r)

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
                    Kmat[r], Vmat[r], A[r], triu, snap, cp, bs, dh, dcp, dko, dvo, dgo,
                    dst_r, dgam_r, zc, zerov, dK[r], dV[r], dA[r], dSl[r], dCv[r], device=r)
            return dQ, dK, dV, dA

    return FusedBackwardProgram
