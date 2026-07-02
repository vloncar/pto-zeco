import pypto.language as pl
import pypto.language.distributed as pld

def build_allscan_program(dk: int, dv: int, K: int, P: int):
    """
    Builds the PyPTO program for All-Scan collective communication.

    dk: Hidden dimension size (keys)
    dv: Hidden dimension size (values)
    K: Pipeline depth (number of blocks)
    P: Number of devices (ranks)
    """
    assert dk % K == 0, f"dk ({dk}) must be divisible by K ({K})"
    BLOCK_SIZE = dk // K

    @pl.program
    class AllScanProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def allscan_first_step(
            self,
            S_local: pl.Tensor[[dk, dv], pl.FP32],
            S_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_next: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[dk, dv], pl.FP32]:
            """Rank 0 (source): emit local state and push it to the next rank.

            Args:
                S_local: This rank's local state, ``[dk, dv]``.
                S_out: Output tensor for this rank's scan result, ``[dk, dv]``
                    (out[0] == S_local[0]); written block by block.
                dst: The *next* rank's recv window (a symmetric HCCL buffer);
                    ``remote_store`` targets it, ``[dk, dv]``.
                signal: The next rank's per-block signal window, ``[K, 1]``;
                    bumped once per block so the peer's wait can proceed.
                peer_next: Rank id of the next ring participant (``rank + 1``).

            Returns:
                The updated ``S_out`` ref (SSA store chain).
            """
            # Rank 0: Send its local state blocks to the next rank
            for k in pl.range(K):
                offset_k = k * BLOCK_SIZE
                S_send_k = pl.load(S_local, [offset_k, 0], [BLOCK_SIZE, dv])

                # Store local output — capture return value (SSA: store returns updated ref)
                S_out = pl.store(S_send_k, [offset_k, 0], S_out)

                # Push block to peer
                pld.tile.remote_store(S_send_k, target=dst, peer=peer_next, offsets=[offset_k, 0])

                # Memory-ordering barrier: drain the MTE3 store pipe so the data
                # is globally visible before the signal lands (Ascend 910B NoC is
                # weakly ordered; without this the peer can observe the notify
                # before the remote_store data — the producer-side race).
                pld.system.fence()

                # Notify peer (AtomicAdd matches the simpler reference kernel and
                # is forward-compatible with epoch-based batching).
                pld.system.notify(
                    target=signal,
                    peer=peer_next,
                    offsets=[k, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
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
            """Middle ranks: recv from prev, fuse ``S_local + gamma*recv``, send on.

            Args:
                S_local: This rank's local state, ``[dk, dv]``.
                gamma: This rank's decay factor, ``[dk, 1]``; broadcast-multiplies
                    the received block across ``dv`` columns.
                S_out: Output tensor for this rank's scan result, ``[dk, dv]``.
                dst: Dual-purpose window — this rank *reads* its own recv slot
                    (the block from the previous rank) and *writes* the next
                    rank's recv slot, ``[dk, dv]``.
                signal: Per-block signal window, ``[K, 1]``; this rank waits on
                    its own slot, then bumps the next rank's.
                peer_next: Rank id of the next ring participant (``rank + 1``).

            Returns:
                The updated ``S_out`` ref (SSA store chain).
            """
            # Middle Ranks: Receive block from prev, update, send to next
            for k in pl.range(K):
                offset_k = k * BLOCK_SIZE

                # Wait for previous rank to write to our buffer
                pld.system.wait(
                    signal=signal,
                    offsets=[k, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

                S_recv_k = pl.load(dst, [offset_k, 0], [BLOCK_SIZE, dv])
                S_local_k = pl.load(S_local, [offset_k, 0], [BLOCK_SIZE, dv])
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK_SIZE, 1])

                scaled_recv_k = pl.tile.row_expand_mul(S_recv_k, gamma_k)
                S_send_k = pl.tile.add(S_local_k, scaled_recv_k)

                S_out = pl.store(S_send_k, [offset_k, 0], S_out)

                pld.tile.remote_store(S_send_k, target=dst, peer=peer_next, offsets=[offset_k, 0])

                # See allscan_first_step: fence before notify so data is visible
                # before the signal on the weakly-ordered NoC.
                pld.system.fence()

                pld.system.notify(
                    target=signal,
                    peer=peer_next,
                    offsets=[k, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
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
            """Rank P-1 (terminal): recv from prev, fuse, terminate (no send).

            Args:
                S_local: This rank's local state, ``[dk, dv]``.
                gamma: This rank's decay factor, ``[dk, 1]``.
                S_out: Output tensor for this rank's scan result, ``[dk, dv]``.
                dst: This rank's own recv window (the block from the previous
                    rank), read-only here, ``[dk, dv]``.
                signal: This rank's per-block signal window, ``[K, 1]``; waited on.

            Returns:
                The updated ``S_out`` ref (SSA store chain).
            """
            # Last Rank: Receive block from prev, update, NO send
            for k in pl.range(K):
                offset_k = k * BLOCK_SIZE

                pld.system.wait(
                    signal=signal,
                    offsets=[k, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

                S_recv_k = pl.load(dst, [offset_k, 0], [BLOCK_SIZE, dv])
                S_local_k = pl.load(S_local, [offset_k, 0], [BLOCK_SIZE, dv])
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK_SIZE, 1])

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
            """Orchestration wrapper dispatching :meth:`allscan_first_step` on one
            chip. Same args as that kernel; the target device is bound by the
            ``device=`` kwarg at the :meth:`host_orch` call site."""
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
            """Orchestration wrapper dispatching :meth:`allscan_middle_step` on one
            chip (device bound via ``device=`` at the :meth:`host_orch` call site)."""
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
            """Orchestration wrapper dispatching :meth:`allscan_last_step` on one
            chip (device bound via ``device=`` at the :meth:`host_orch` call site)."""
            return self.allscan_last_step(S_local, gamma, S_out, dst, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            S_locals: pl.Tensor[[P, dk, dv], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            outputs: pl.Out[pl.Tensor[[P, dk, dv], pl.FP32]],
        ) -> pl.Tensor[[P, dk, dv], pl.FP32]:
            """Host orchestrator: allocate the shared window and dispatch one chip
            kernel per rank (first / middle / last), wiring each to device ``r``.

            Args:
                S_locals: All ranks' local state, ``[P, dk, dv]``; ``S_locals[r]``
                    goes to device ``r``.
                gammas: All ranks' decay factors, ``[P, dk, 1]``.
                outputs: All ranks' scan outputs, ``[P, dk, dv]``; written in place.

            Returns:
                ``outputs``.
            """
            dst_buf = pld.alloc_window_buffer(dk * dv * 4)
            signal_buf = pld.alloc_window_buffer(K * 4)

            for r in pl.range(P):
                # Pre-slice unconditionally so the compiler emits slice assignments
                # before the if/elif/else — slices inside conditionals are not
                # hoisted by the code generator and would produce a KeyError.
                S_local_r = S_locals[r]
                gamma_r = gammas[r]
                output_r = outputs[r]
                dst = pld.window(dst_buf, [dk, dv], dtype=pl.FP32)
                signal = pld.window(signal_buf, [K, 1], dtype=pl.INT32)

                if r == 0:
                    self.chip_orch_first(S_local_r, output_r, dst, signal, r + 1, device=r)
                elif r == P - 1:
                    self.chip_orch_last(S_local_r, gamma_r, output_r, dst, signal, device=r)
                else:
                    self.chip_orch_middle(S_local_r, gamma_r, output_r, dst, signal, r + 1, device=r)
            return outputs

    return AllScanProgram
