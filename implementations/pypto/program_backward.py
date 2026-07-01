import pypto.language as pl
import pypto.language.distributed as pld


def build_allscan_backward_program(dk: int, dv: int, K: int, P: int):
    """
    Builds the PyPTO program for the All-Scan backward pass.

    Backward of the forward scan  out[p] = S_local[p] + gamma[p] (*) out[p-1].
    Given the upstream gradient g_out[p] = dL/dout[p], the adjoint d[p] is a
    *reverse* scan with gamma shifted by one:

        d[P-1] = g_out[P-1]
        d[p]   = g_out[p] + gamma[p+1] (*) d[p+1]      (p = P-2 .. 0)

    and the input gradients are local:

        dS[p]     = d[p]                                (all p)
        dgamma[p] = rowsum_dv( d[p] (*) out[p-1] )      (p = 1 .. P-1) -> [dk,1]
        dgamma[0] = 0                                   (gamma[0] is unused)

    The forward ring flows rank -> rank+1; the adjoint flows rank -> rank-1.
    Each rank forwards the message m = gamma[p] (*) d[p] into the previous rank's
    recv slot; the receiver adds its own g_out to form d. ``out_prev`` (= out[p-1],
    the block a rank received during the forward pass) is passed in so the dgamma
    reduction is fully local. Roles by rank:
      rank P-1 : source   — d = g_out, no recv; send m to P-2, reduce dgamma.
      rank 1..P-2 : middle — recv m, d = g_out + m; send m to prev, reduce dgamma.
      rank 0   : terminal — recv m, d = g_out + m; no send, dgamma[0] = 0.
    """
    assert dk % K == 0, f"dk ({dk}) must be divisible by K ({K})"
    BLOCK_SIZE = dk // K

    @pl.program
    class AllScanBackwardProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def allscan_bwd_source_step(
            self,
            g_out: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            out_prev: pl.Tensor[[dk, dv], pl.FP32],
            dS_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgamma_out: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ):
            # Rank P-1: d = g_out (no incoming message).
            for k in pl.range(K):
                offset_k = k * BLOCK_SIZE
                d_k = pl.load(g_out, [offset_k, 0], [BLOCK_SIZE, dv])

                # Local dS = d.
                dS_out = pl.store(d_k, [offset_k, 0], dS_out)

                # dgamma = rowsum_dv(d (*) out_prev).
                out_prev_k = pl.load(out_prev, [offset_k, 0], [BLOCK_SIZE, dv])
                prod_k = pl.tile.mul(d_k, out_prev_k)
                tmp_k = pl.tile.create([BLOCK_SIZE, dv], pl.FP32)
                dgamma_k = pl.row_sum(prod_k, tmp_k)
                dgamma_out = pl.store(dgamma_k, [offset_k, 0], dgamma_out)

                # Message m = gamma (*) d, pushed to the previous rank.
                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK_SIZE, 1])
                msg_k = pl.tile.row_expand_mul(d_k, gamma_k)
                pld.tile.remote_store(msg_k, target=dst, peer=peer_prev, offsets=[offset_k, 0])

                # Drain the store pipe so the data is globally visible before the
                # signal lands (weakly-ordered NoC; see forward program).
                pld.system.fence()
                pld.system.notify(
                    target=signal,
                    peer=peer_prev,
                    offsets=[k, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
            return dS_out, dgamma_out

        @pl.function(type=pl.FunctionType.InCore)
        def allscan_bwd_middle_step(
            self,
            g_out: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            out_prev: pl.Tensor[[dk, dv], pl.FP32],
            dS_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgamma_out: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ):
            # Middle ranks: receive m from p+1, form d = g_out + m, forward m to p-1.
            for k in pl.range(K):
                offset_k = k * BLOCK_SIZE

                pld.system.wait(
                    signal=signal,
                    offsets=[k, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

                msg_k = pl.load(dst, [offset_k, 0], [BLOCK_SIZE, dv])
                g_out_k = pl.load(g_out, [offset_k, 0], [BLOCK_SIZE, dv])
                d_k = pl.tile.add(g_out_k, msg_k)

                dS_out = pl.store(d_k, [offset_k, 0], dS_out)

                out_prev_k = pl.load(out_prev, [offset_k, 0], [BLOCK_SIZE, dv])
                prod_k = pl.tile.mul(d_k, out_prev_k)
                tmp_k = pl.tile.create([BLOCK_SIZE, dv], pl.FP32)
                dgamma_k = pl.row_sum(prod_k, tmp_k)
                dgamma_out = pl.store(dgamma_k, [offset_k, 0], dgamma_out)

                gamma_k = pl.load(gamma, [offset_k, 0], [BLOCK_SIZE, 1])
                msg_out_k = pl.tile.row_expand_mul(d_k, gamma_k)
                pld.tile.remote_store(msg_out_k, target=dst, peer=peer_prev, offsets=[offset_k, 0])

                pld.system.fence()
                pld.system.notify(
                    target=signal,
                    peer=peer_prev,
                    offsets=[k, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
            return dS_out, dgamma_out

        @pl.function(type=pl.FunctionType.InCore)
        def allscan_bwd_terminal_step(
            self,
            g_out: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            out_prev: pl.Tensor[[dk, dv], pl.FP32],
            dS_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgamma_out: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ):
            # Rank 0: receive m from rank 1, form d = g_out + m. gamma[0] is unused,
            # so dgamma[0] is left untouched (the host zeroes the dgamma buffer
            # before dispatch, so dgamma[0] == 0); no outgoing message.
            for k in pl.range(K):
                offset_k = k * BLOCK_SIZE

                pld.system.wait(
                    signal=signal,
                    offsets=[k, 0],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )

                msg_k = pl.load(dst, [offset_k, 0], [BLOCK_SIZE, dv])
                g_out_k = pl.load(g_out, [offset_k, 0], [BLOCK_SIZE, dv])
                d_k = pl.tile.add(g_out_k, msg_k)

                dS_out = pl.store(d_k, [offset_k, 0], dS_out)

                # dgamma[0] is left untouched (gamma[0] is unused); the host
                # zeroes the dgamma buffer before dispatch, so dgamma[0] == 0.
            return dS_out, dgamma_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_bwd_source(
            self,
            g_out: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            out_prev: pl.Tensor[[dk, dv], pl.FP32],
            dS_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgamma_out: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ):
            return self.allscan_bwd_source_step(
                g_out, gamma, out_prev, dS_out, dgamma_out, dst, signal, peer_prev
            )

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_bwd_middle(
            self,
            g_out: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            out_prev: pl.Tensor[[dk, dv], pl.FP32],
            dS_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgamma_out: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
            peer_prev: pl.Scalar[pl.INT32],
        ):
            return self.allscan_bwd_middle_step(
                g_out, gamma, out_prev, dS_out, dgamma_out, dst, signal, peer_prev
            )

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_bwd_terminal(
            self,
            g_out: pl.Tensor[[dk, dv], pl.FP32],
            gamma: pl.Tensor[[dk, 1], pl.FP32],
            out_prev: pl.Tensor[[dk, dv], pl.FP32],
            dS_out: pl.Out[pl.Tensor[[dk, dv], pl.FP32]],
            dgamma_out: pl.Out[pl.Tensor[[dk, 1], pl.FP32]],
            dst: pld.DistributedTensor[[dk, dv], pl.FP32],
            signal: pld.DistributedTensor[[K, 1], pl.INT32],
        ):
            return self.allscan_bwd_terminal_step(
                g_out, gamma, out_prev, dS_out, dgamma_out, dst, signal
            )

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            g_outs: pl.Tensor[[P, dk, dv], pl.FP32],
            gammas: pl.Tensor[[P, dk, 1], pl.FP32],
            out_prevs: pl.Tensor[[P, dk, dv], pl.FP32],
            dS: pl.Out[pl.Tensor[[P, dk, dv], pl.FP32]],
            dgamma: pl.Out[pl.Tensor[[P, dk, 1], pl.FP32]],
        ):
            dst_buf = pld.alloc_window_buffer(dk * dv * 4)
            signal_buf = pld.alloc_window_buffer(K * 4)

            for r in pl.range(P):
                g_out_r = g_outs[r]
                gamma_r = gammas[r]
                out_prev_r = out_prevs[r]
                dS_r = dS[r]
                dgamma_r = dgamma[r]
                dst = pld.window(dst_buf, [dk, dv], dtype=pl.FP32)
                signal = pld.window(signal_buf, [K, 1], dtype=pl.INT32)

                if r == P - 1:
                    self.chip_orch_bwd_source(
                        g_out_r, gamma_r, out_prev_r, dS_r, dgamma_r, dst, signal, r - 1, device=r
                    )
                elif r == 0:
                    self.chip_orch_bwd_terminal(
                        g_out_r, gamma_r, out_prev_r, dS_r, dgamma_r, dst, signal, device=r
                    )
                else:
                    self.chip_orch_bwd_middle(
                        g_out_r, gamma_r, out_prev_r, dS_r, dgamma_r, dst, signal, r - 1, device=r
                    )
            return dS, dgamma

    return AllScanBackwardProgram
