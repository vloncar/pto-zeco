"""Fully-fused distributed PyPTO ZeCO forward: stage1 + AllScan-ring + stage2 in ONE program.

This is the "entire forward in PyPTO" — no ``@pl.jit`` hybrid, no host round-trip. Per
rank ``r`` (device ``r``) the single ``host_orch`` runs three phases:

1. **stage1** (InCore) — local end-of-slice state ``S_total`` (from ``S = 0``), fed to the
   ring. Identical to :mod:`.dist_program`'s ``gla_stage1``.
2. **AllScan ring** (InCore first/middle/last) — the exclusive-prefix boundary scan
   ``out[p] = S_local[p] + gamma[p]*out[p-1]``. Each rank receives ``out[p-1]`` into its
   window ``dst``; the middle/last steps **also store that received value into ``S_recv``**
   and return it, so stage2 reads the boundary state device-locally (synchronized by the
   ring's notify/wait — reading ``outputs[r-1]`` from GM would be an unsynchronized
   cross-device race).
3. **stage2** (InCore) — ``O[r]`` from ``Q,K,V,A`` and ``S_recv[r]`` (zero for rank 0), the
   chunk recurrence initialised from the boundary. This is the wide matmul-DAG kernel that
   historically only survived ``@pl.jit``; running it as a distributed chip kernel here is
   the point of the fusion.

stage2 is dispatched **once, after the ``if/elif/else`` merge**, consuming the boundary phi
(``zero`` for rank 0 vs the ring's returned ``out[r-1]`` for middle/last). That cross-branch phi
used to miscompile the host orchestrator — ``ir.compile`` clean, then ``NameError:
zero__ssa_v0 not defined`` in the generated ``host_orch.py`` at ``prepare()`` — which forced a
dispatch-inside-every-branch workaround. Fixed upstream by pypto PR #2183 (merged 2026-07-30),
so the workaround is gone; see ``issues/pypto-crossbranch-phi-nameerror/``.

**P == 1** is a native path (:func:`_build_p1_forward_program`): a single rank has no
communication at all, so there is no ring and stage1 is dead (its ``S_total`` only feeds the
ring). The P=1 program is therefore just stage2 from a zero boundary. Its ``host_orch`` keeps
the uniform ``[P, ...]`` entry and stays a distributed program (same ``prepare``/``rt``/
``close`` run path as P>1). This relies on the P=1 ``device=r`` unroll fix (pypto ``1a18fb26``:
the single-trip ``for r in pl.range(1)`` folds ``r`` → 0 in the ``device=`` attr; before that
fix codegen emitted an unbound ``r__idx_v0`` → NameError). Use :func:`run_fused_forward`, which
also falls back to a non-distributed single-device run if a config compiles P=1 without a HOST
orchestrator (entry then = stage2's own signature ``(Q, Kmat, Vmat, A, tril, mask, ones_cc,
ones_cdv, Srecv, O)``).

**P=1 and P>1 MUST be built by separate factory functions** (not one class with an
``if P == 1:`` branch): conditionally-defined methods in a ``@pl.program`` class body are
silently NOT registered — that drops the ring functions and collapses the whole program down
to just stage2 (ranks > 0 then never receive their boundary → garbage). Learned the hard way.

Runs on a2a3sim (the earlier CPU-sim cross-core pipe deadlock on these chunk kernels was
fixed upstream).
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld


def run_fused_forward(compiled, Q, K, V, A, gammas, tril, mask, ones_cc, ones_cdv, zero, O,
                      *, platform, device_ids):
    """Run a compiled fused-forward program, hiding the P==1 vs P>1 run-API split.

    P>1 compiles to a distributed program (``prepare``/``rt``/``close``, share-memory
    tensors). P==1 has no communication, so pypto compiles it to a plain single-device
    ``CompiledProgram`` whose entry is stage2's own signature
    ``(Q, Kmat, Vmat, A, tril, mask, ones_cc, ones_cdv, Srecv, O)`` — run directly with the
    rank-0 slices and ``zero`` as ``Srecv``. Writes results in place into ``O`` (``[P, L, dv]``).
    """
    if hasattr(compiled, "prepare"):
        def sm(t):
            return t if t.is_shared() else t.clone().share_memory_()
        h = [sm(t) for t in (Q, K, V, A, gammas, tril, mask, ones_cc, ones_cdv, zero)]
        h_O = sm(O)
        rt = compiled.prepare()
        try:
            rt(*h, h_O)
        finally:
            rt.close()
        if h_O is not O:
            O.copy_(h_O)
        return O
    # P==1: non-distributed single-device program, entry = stage2 signature (rank-0 slices).
    from pypto.runtime.runner import RunConfig
    compiled(Q[0], K[0], V[0], A[0], tril, mask, ones_cc, ones_cdv, zero, O[0],
             config=RunConfig(platform=platform, device_id=device_ids[0]))
    return O


def _build_p1_forward_program(L: int, C: int, dk: int, dv: int, K: int):
    """P == 1 native path: stage2 from a zero boundary (no ring; stage1 is dead).

    Compiles to a non-distributed single-device program (entry = the stage2 signature, since
    pypto DCE's the dead stage1 and inlines the single chip dispatch). Kept as its own factory
    so every method is a top-level ``@pl.program`` class member (conditional class-body defs
    are silently dropped)."""
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    N = L // C
    P, DK, DV = 1, dk, dv

    @pl.program
    class FusedForwardP1Program:
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage2(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            mask: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            """O[n] = (Q_n*b_n)@S_run + ((Q_n*b_n)@(K_n/b_n)^T ⊙ mask)@V_n; carry S from Srecv."""
            tril_t = pl.load(tril, [0, 0], [C, C])
            mask_t = pl.load(mask, [0, 0], [C, C])
            ones_cc_t = pl.load(ones_cc, [0, 0], [C, C])
            ones_cdv_t = pl.load(ones_cdv, [0, 0], [C, DV])
            s_init = pl.load(Srecv, [0, 0], [DK, DV])
            out = O
            for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
                off = n * C
                q = pl.load(Q, [off, 0], [C, DK])
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                g_row_full = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
                g_full = pl.exp(pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
                qt = pl.mul(q, b)
                kb = pl.div(k, b)
                scores = pl.mul(pl.matmul(qt, pl.transpose(kb, 0, 1), out_dtype=pl.FP32), mask_t)
                o_intra = pl.matmul(scores, v, out_dtype=pl.FP32)
                s_run_v = pl.mul(s_run, 1.0)  # detach carry for matmul (raw iter_arg stays in vec)
                o_inter = pl.matmul(qt, s_run_v, out_dtype=pl.FP32)
                o_n = pl.add(o_inter, o_intra)
                out = pl.store(o_n, [off, 0], out)
                kbar = pl.mul(kb, g_row_full)
                kv = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
                s_scaled = pl.mul(s_run, g_full)
                s_new = pl.add(s_scaled, kv)
                s_run = pl.yield_(s_new)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_stage2(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            mask: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            return self.gla_stage2(Q, Kmat, Vmat, A, tril, mask, ones_cc, ones_cdv, Srecv, O)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            Qmat: pl.Tensor[[P, L, dk], pl.FP32],
            Kmat: pl.Tensor[[P, L, dk], pl.FP32],
            Vmat: pl.Tensor[[P, L, dv], pl.FP32],
            A: pl.Tensor[[P, L, dk], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            mask: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, dv], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            O: pl.Out[pl.Tensor[[P, L, dv], pl.FP32]],
        ) -> pl.Tensor[[P, L, dv], pl.FP32]:
            """P == 1: single rank, S_recv = 0 (no boundary). ``gammas`` is unused."""
            for r in pl.range(P):
                Q_r = Qmat[r]
                K_r = Kmat[r]
                V_r = Vmat[r]
                A_r = A[r]
                O_r = O[r]
                self.chip_orch_stage2(
                    Q_r, K_r, V_r, A_r, tril, mask, ones_cc, ones_cdv, zero, O_r, device=r)
            return O

    return FusedForwardP1Program


def build_fused_forward_program(L: int, C: int, dk: int, dv: int, K: int, P: int):
    """Build the fully-fused ``stage1 + ring + stage2`` distributed program.

    Args:
        L: Tokens per device. C: chunk size (``L % C == 0``, ``N = L // C``).
        dk, dv: key/query and value dims. K: ring pipeline depth (``dk % K == 0``).
        P: ranks / devices. ``P == 1`` builds the native single-rank program (stage2 only).

    Returns:
        A ``@pl.program`` whose ``host_orch`` takes
        ``(Qmat, Kmat, Vmat, A, gammas, tril, mask, ones_cc, ones_cdv, zero)`` and writes
        ``O`` ``[P, L, dv]`` (the per-rank ZeCO outputs). Run via :func:`run_fused_forward`.
    """
    assert dk % K == 0, f"dk ({dk}) must be divisible by K ({K})"
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    if P == 1:
        return _build_p1_forward_program(L, C, dk, dv, K)

    BLOCK = dk // K
    N = L // C
    DK, DV = dk, dv

    @pl.program
    class FusedForwardProgram:
        # ---- phase 1: chunk-recurrent GLA stage1 (local S_total, from S=0) ----
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
            # NOTE (F2, root-caused + FIXED 2026-07-31): this kernel used to be why the fused
            # forward was wrong for N>2, but nothing written here was ever at fault. On a2a3 the
            # two directions of a bidirectional cube<->vector TPipe indexed the SAME GM slots
            # (the per-direction entryOffset the ISA reference specifies is never applied), so
            # the cube's matmul results and the vector's operands overwrote each other. The
            # loop-carried state was simply the visible victim.
            # Fixed by two upstream-bound diffs (carried locally):
            #   pto-isa: set the V2C entry offset in the TPipe ctor for DIR_BOTH
            #   pypto:   size the GM pipe workspace for BOTH rings, honour slot_num
            # With those, stock ring depth, the fused forward is 12/12 at C=16 and 12/12 at
            # C=32. See allscan/issues/pypto-incore-loop-cube-vector-race/ROOT-CAUSE.md.
            # (The post-loop carry read below is fine and always was: swapping it for a
            # per-iteration store left the old failures bit-identical.)
            return pl.store(s_fin, [0, 0], Stot)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_stage1(
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

        # ---- phase 3: chunk-recurrent GLA stage2 (output from boundary S_recv) ----
        @pl.function(type=pl.FunctionType.InCore)
        def gla_stage2(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            mask: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            """O[n] = (Q_n*b_n)@S_run + ((Q_n*b_n)@(K_n/b_n)^T ⊙ mask)@V_n; carry S from Srecv."""
            tril_t = pl.load(tril, [0, 0], [C, C])
            mask_t = pl.load(mask, [0, 0], [C, C])
            ones_cc_t = pl.load(ones_cc, [0, 0], [C, C])
            ones_cdv_t = pl.load(ones_cdv, [0, 0], [C, DV])
            s_init = pl.load(Srecv, [0, 0], [DK, DV])
            out = O
            for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
                off = n * C
                q = pl.load(Q, [off, 0], [C, DK])
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                g_row_full = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
                g_full = pl.exp(pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
                qt = pl.mul(q, b)
                kb = pl.div(k, b)
                scores = pl.mul(pl.matmul(qt, pl.transpose(kb, 0, 1), out_dtype=pl.FP32), mask_t)
                o_intra = pl.matmul(scores, v, out_dtype=pl.FP32)
                s_run_v = pl.mul(s_run, 1.0)  # detach carry for matmul (raw iter_arg stays in vec)
                o_inter = pl.matmul(qt, s_run_v, out_dtype=pl.FP32)
                o_n = pl.add(o_inter, o_intra)
                out = pl.store(o_n, [off, 0], out)
                kbar = pl.mul(kb, g_row_full)
                kv = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
                s_scaled = pl.mul(s_run, g_full)
                s_new = pl.add(s_scaled, kv)
                s_run = pl.yield_(s_new)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_stage2(
            self,
            Q: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            A: pl.Tensor[[L, DK], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            mask: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            Srecv: pl.Tensor[[DK, DV], pl.FP32],
            O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
        ) -> pl.Tensor[[L, DV], pl.FP32]:
            return self.gla_stage2(Q, Kmat, Vmat, A, tril, mask, ones_cc, ones_cdv, Srecv, O)

        # ---- phase 2: AllScan ring (first/middle/last), emitting S_recv ----
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
                pld.system.notify(target=signal, peer=peer_next, offsets=[kk, 0], value=1, op=pld.NotifyOp.AtomicAdd)
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
            # S_recv is the SOLE Out (out[r-1], the boundary for stage2); out[r] is sent to
            # r+1 via remote_store, never read locally, so no S_out store here.
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
                pld.system.notify(target=signal, peer=peer_next, offsets=[kk, 0], value=1, op=pld.NotifyOp.AtomicAdd)
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

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            Qmat: pl.Tensor[[P, L, dk], pl.FP32],
            Kmat: pl.Tensor[[P, L, dk], pl.FP32],
            Vmat: pl.Tensor[[P, L, dv], pl.FP32],
            A: pl.Tensor[[P, L, dk], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            mask: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, dv], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            O: pl.Out[pl.Tensor[[P, L, dv], pl.FP32]],
        ) -> pl.Tensor[[P, L, dv], pl.FP32]:
            """Per rank r on device r: stage1 -> ring (first/middle/last, emitting S_recv)
            -> stage2 reading S_recv (zero for rank 0), dispatched once after the merge."""
            dst_buf = pld.alloc_window_buffer(dk * dv * 4)
            signal_buf = pld.alloc_window_buffer(K * 4)
            S_local = pl.create_tensor([P, dk, dv], dtype=pl.FP32)     # stage1 local S_total (ring input)
            S_out_all = pl.create_tensor([P, dk, dv], dtype=pl.FP32)   # ring inclusive-scan out[r] (sent)
            S_recv_all = pl.create_tensor([P, dk, dv], dtype=pl.FP32)  # received boundary out[r-1] per rank

            for r in pl.range(P):
                # Slice this rank's inputs once, before the if/elif/else.
                Q_r = Qmat[r]
                K_r = Kmat[r]
                V_r = Vmat[r]
                A_r = A[r]
                gamma_r = gammas[r]
                O_r = O[r]
                S_local_r = S_local[r]
                S_out_r = S_out_all[r]
                S_recv_r = S_recv_all[r]
                dst = pld.window(dst_buf, [dk, dv], dtype=pl.FP32)
                signal = pld.window(signal_buf, [K, 1], dtype=pl.INT32)

                sl_r = self.chip_orch_stage1(
                    A_r, K_r, V_r, tril, ones_cc, ones_cdv, zero, S_local_r, device=r)

                # The ring picks this rank's boundary: rank 0 has none (S_recv = 0), the
                # others take the received out[r-1]. stage2 then consumes the merged value
                # ONCE, after the if/elif/else — the natural way to write this.
                # (Until pypto PR #2183 this cross-branch phi — an input tensor `zero` in one
                # branch vs an Orchestration SSA result in the others — miscompiled the host
                # orchestrator into an out-of-scope name: `NameError: zero__ssa_v0 not
                # defined` at prepare(). That fix is merged, so the workaround of dispatching
                # stage2 inside every branch is gone. See issues/pypto-crossbranch-phi-nameerror/.)
                if r == 0:
                    self.chip_orch_first(sl_r, S_out_r, dst, signal, r + 1, device=r)
                    boundary = zero
                elif r == P - 1:
                    boundary = self.chip_orch_last(sl_r, gamma_r, S_recv_r, dst, signal, device=r)
                else:
                    boundary = self.chip_orch_middle(
                        sl_r, gamma_r, S_recv_r, dst, signal, r + 1, device=r)

                self.chip_orch_stage2(
                    Q_r, K_r, V_r, A_r, tril, mask, ones_cc, ones_cdv, boundary, O_r, device=r)
            return O

    return FusedForwardProgram
