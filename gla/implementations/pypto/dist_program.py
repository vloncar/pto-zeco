"""Fused distributed PyPTO program: GLA ``stage1`` + AllScan ring in one host_orch.

This is the communication half of the ZeCO forward. Per rank ``r`` (on device
``r``) the host orchestrator runs two phases:

1. **stage1** — the chunk-recurrent local end-of-slice state ``S_total`` (from
   ``S = 0``), computed on-device as an ``InCore`` tile kernel (identical math to
   :func:`gla.implementations.pypto.program._stage1_kernel`, but expressed as a
   ``@pl.function(type=InCore)`` method so it runs as a *distributed chip kernel*
   rather than a ``@pl.jit`` dispatch). Written into a ``pl.create_tensor``
   intermediate, whose returned ref is threaded into phase 2.
2. **AllScan ring** — the exclusive-prefix boundary scan
   ``out[p] = S_local[p] + gamma[p] * out[p-1]`` (verbatim from
   :func:`allscan.implementations.pypto.program.build_allscan_program`), reading
   the freshly-computed ``S_local`` and producing ``outputs`` (the inclusive
   ring scan). The caller derives ``S_recv[r] = outputs[r-1]`` by a host shift.

**Why stage1 is fused here but stage2 is NOT** (see :mod:`.impl`): folding stage1
into the distributed program removes the only ``@pl.jit`` dispatch that would run
*before* the ``DistributedWorker.prepare()`` fork — which is exactly the
coexistence that segfaults at ``P > 1`` (jit-dispatch-then-prepare). stage2, by
contrast, is a wide matmul-DAG kernel that **hangs as a distributed chip kernel**
(AICore ``507018`` device-drain timeout) and only survives the ``@pl.jit``
CORE_GROUP path, so it stays on jit and runs *after* this worker closes (the
safe ``prepare -> close -> jit`` order). See :mod:`.program` for the
sim-vs-hardware / loop-carry notes on the chunk kernels.

Runs on both a2a3sim and a2a3 (the earlier a2a3sim deadlock on the chunk-recurrent
stage1 body was fixed upstream in the CPU-sim cross-core pipe model).
"""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld


def build_stage1ring_program(L: int, C: int, dk: int, dv: int, K: int, P: int):
    """Build the fused ``stage1 + AllScan-ring`` distributed program for these shapes.

    Args:
        L: Tokens per device.
        C: Chunk size (``L`` divisible by ``C``); ``N = L // C`` chunks.
        dk: Key/query dimension.
        dv: Value dimension.
        K: Ring pipeline depth (``dk`` divisible by ``K``); ``BLOCK = dk // K``.
        P: Number of ranks / devices.

    Returns:
        A ``@pl.program`` class whose ``host_orch`` takes
        ``(A, K, V, gammas, tril, ones_cc, ones_cdv, zero)`` and writes
        ``outputs`` ``[P, dk, dv]`` — the inclusive ring scan of the per-rank
        local states (the caller shifts it to ``S_recv``).
    """
    assert dk % K == 0, f"dk ({dk}) must be divisible by K ({K})"
    assert L % C == 0, f"L ({L}) must be divisible by C ({C})"
    BLOCK = dk // K
    N = L // C
    DK, DV = dk, dv

    @pl.program
    class Stage1RingProgram:
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
            """S <- (S ⊙ gamma_n) + (K_n*(gamma_n/b_n))^T @ V_n, carried over N chunks."""
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
            """Orchestration wrapper dispatching :meth:`gla_stage1` on device ``r``."""
            return self.gla_stage1(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Stot)

        # ---- phase 2: AllScan ring (verbatim from allscan program.py) ----
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
                pld.system.fence()
                pld.system.notify(target=signal, peer=peer_next, offsets=[kk, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            return S_out

        @pl.function(type=pl.FunctionType.InCore)
        def allscan_middle_step(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                pld.system.wait(signal=signal, offsets=[kk, 0], expected=1, cmp=pld.WaitCmp.Ge)
                S_recv_k = pl.load(dst, [offset_k, 0], [BLOCK, dv])
                S_local_k = pl.load(S_local, [offset_k, 0], [BLOCK, dv])
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK, 1])
                scaled_recv_k = pl.tile.row_expand_mul(S_recv_k, gamma_k)
                S_send_k = pl.tile.add(S_local_k, scaled_recv_k)
                S_out = pl.store(S_send_k, [offset_k, 0], S_out)
                pld.tile.remote_store(S_send_k, target=dst, peer=peer_next, offsets=[offset_k, 0])
                pld.system.fence()
                pld.system.notify(target=signal, peer=peer_next, offsets=[kk, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            return S_out

        @pl.function(type=pl.FunctionType.InCore)
        def allscan_last_step(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            for kk in pl.range(K):
                offset_k = kk * BLOCK
                pld.system.wait(signal=signal, offsets=[kk, 0], expected=1, cmp=pld.WaitCmp.Ge)
                S_recv_k = pl.load(dst, [offset_k, 0], [BLOCK, dv])
                S_local_k = pl.load(S_local, [offset_k, 0], [BLOCK, dv])
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK, 1])
                scaled_recv_k = pl.tile.row_expand_mul(S_recv_k, gamma_k)
                S_send_k = pl.tile.add(S_local_k, scaled_recv_k)
                S_out = pl.store(S_send_k, [offset_k, 0], S_out)
            return S_out

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
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            return self.allscan_middle_step(S_local, gamma, S_out, dst, signal, peer_next)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_last(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            return self.allscan_last_step(S_local, gamma, S_out, dst, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            A: pl.Tensor[[P, L, dk], pl.FP32],
            Kmat: pl.Tensor[[P, L, dk], pl.FP32],
            Vmat: pl.Tensor[[P, L, dv], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, dv], pl.FP32],
            zero: pl.Tensor[[dk, dv], pl.FP32],
            outputs: pl.Out[pl.Tensor[[P, dk, dv], pl.FP32]],
        ) -> pl.Tensor[[P, dk, dv], pl.FP32]:
            """Per rank ``r`` on device ``r``: stage1 -> local ``S``; then the ring
            step (first/middle/last) reading that ``S`` and writing ``outputs[r]``.

            The stage1 result feeds the ring via its returned ref (``sl_r``), so the
            compiler orders phase 1 before phase 2 on each device. ``S_local`` is a
            ``pl.create_tensor`` intermediate (never leaves the device).
            """
            dst_buf = pld.alloc_window_buffer(dk * dv * 4)
            signal_buf = pld.alloc_window_buffer(K * 4)
            S_local = pl.create_tensor([P, dk, dv], dtype=pl.FP32)

            for r in pl.range(P):
                # Pre-slice unconditionally (slices inside conditionals are not hoisted
                # by the code generator — see allscan program.py).
                A_r = A[r]
                K_r = Kmat[r]
                V_r = Vmat[r]
                gamma_r = gammas[r]
                output_r = outputs[r]
                S_local_r = S_local[r]
                dst = pld.window(dst_buf, [dk, dv], dtype=pl.FP32)
                signal = pld.window(signal_buf, [K, 1], dtype=pl.INT32)

                sl_r = self.chip_orch_stage1(
                    A_r, K_r, V_r, tril, ones_cc, ones_cdv, zero, S_local_r, device=r)

                if r == 0:
                    self.chip_orch_first(sl_r, output_r, dst, signal, r + 1, device=r)
                elif r == P - 1:
                    self.chip_orch_last(sl_r, gamma_r, output_r, dst, signal, device=r)
                else:
                    self.chip_orch_middle(sl_r, gamma_r, output_r, dst, signal, r + 1, device=r)
            return outputs

    return Stage1RingProgram
