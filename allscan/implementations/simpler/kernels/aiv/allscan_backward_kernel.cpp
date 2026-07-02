/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * AllScan backward kernel — direct PTO-runtime ("simpler") port.
 *
 * Backward of the forward scan  out[p] = S_local[p] + gamma[p] (*) out[p-1].
 * Given the upstream gradient g_out[p] = dL/dout[p], the adjoint d[p] = dL/dout[p]
 * (total, including the downstream chain) is a *reverse* scan with gamma shifted
 * by one:
 *
 *     d[P-1] = g_out[P-1]
 *     d[p]   = g_out[p] + gamma[p+1] (*) d[p+1]      (p = P-2 .. 0)
 *
 * from which the input gradients are local:
 *
 *     dS_local[p] = d[p]                              (all p)
 *     dgamma[p]   = rowsum_dv( d[p] (*) out[p-1] )    (p = 1 .. P-1) -> [dk,1]
 *     dgamma[0]   = 0                                 (gamma[0] is unused)
 *
 * The forward ring flows rank -> rank+1; the adjoint flows the other way,
 * rank -> rank-1.  Each rank forwards the *message* m = gamma[p] (*) d[p] into
 * the previous rank's recv slot; the receiver adds its own g_out to form d, so
 *     d[p-1] = g_out[p-1] + m.
 * ``out[p-1]`` (needed for dgamma[p]) is passed in as ``out_prev`` — the block
 * this rank received during the forward pass, so the dgamma reduction is fully
 * local.  One uniform kernel runs on every rank; behaviour selects from rankId:
 *   rank P-1 : source — d = g_out, no recv; push m to P-2, reduce dgamma.
 *   rank 1..P-2 : recv m, d = g_out + m; push m to prev, reduce dgamma.
 *   rank 0   : terminal — recv m, d = g_out + m; no push, dgamma stays 0.
 *
 * Synchronisation mirrors the forward kernel: symmetric HCCL window, remote
 * TSTORE + TNOTIFY(AtomicAdd) into the peer's recv+signal slot, receiver TWAITs
 * on signal >= epoch (window zeroed at alloc, epoch is the 1-based run index).
 *
 * args layout (see allscan_backward_orch.cpp):
 *   tensor(0) g_out    [dk, dv]            INPUT
 *   tensor(1) gamma    [dk, 1]             INPUT   (this rank's gamma[p])
 *   tensor(2) out_prev [dk, dv]            INPUT   (out[p-1]; zeros for rank 0)
 *   tensor(3) dS       [dk, dv]            OUTPUT_EXISTING  (= d[p])
 *   tensor(4) dgamma   [dk, 1]             OUTPUT_EXISTING
 *   tensor(5) scratch  recv[dk*dv] + sig[K] INOUT  (HCCL window)
 *   scalar(0) dk
 *   scalar(1) dv
 *   scalar(2) K        (pipeline depth / number of blocks)
 *   scalar(3) nranks
 *   scalar(4) epoch    (1-based run index; expected signal count this run)
 *   scalar(5) CommContext device pointer
 */

#include <cstdint>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr int kMaxRows = 128;
static constexpr int kMaxCols = 128;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *g_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *gamma_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_prev_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *dS_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *dgamma_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    int dk = static_cast<int>(args[6]);
    int dv = static_cast<int>(args[7]);
    int K = static_cast<int>(args[8]);
    int nranks = static_cast<int>(args[9]);
    int32_t epoch = static_cast<int32_t>(args[10]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[11]);

    __gm__ float *g_out = reinterpret_cast<__gm__ float *>(g_out_tensor->buffer.addr) + g_out_tensor->start_offset;
    __gm__ float *gamma = reinterpret_cast<__gm__ float *>(gamma_tensor->buffer.addr) + gamma_tensor->start_offset;
    __gm__ float *out_prev =
        reinterpret_cast<__gm__ float *>(out_prev_tensor->buffer.addr) + out_prev_tensor->start_offset;
    __gm__ float *dS = reinterpret_cast<__gm__ float *>(dS_tensor->buffer.addr) + dS_tensor->start_offset;
    __gm__ float *dgamma = reinterpret_cast<__gm__ float *>(dgamma_tensor->buffer.addr) + dgamma_tensor->start_offset;
    // scratch = recv region (dk*dv floats) followed by K int32 signal slots.
    __gm__ float *recv = reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;
    __gm__ int32_t *signal = reinterpret_cast<__gm__ int32_t *>(recv + dk * dv);

    int my_rank = static_cast<int>(commCtx->rankId);
    bool is_source = (my_rank == nranks - 1);  // reverse-ring head: no incoming
    bool is_terminal = (my_rank == 0);         // reverse-ring tail: no outgoing, dgamma == 0
    int block = dk / K;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GlobalF = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileF = pto::Tile<pto::TileType::Vec, float, kMaxRows, kMaxCols, pto::BLayout::RowMajor, -1, -1>;
    // Column-vector tiles ([block, 1]) need 32-byte (8-float) aligned cols.
    using ColTile = pto::Tile<pto::TileType::Vec, float, kMaxRows, 8, pto::BLayout::RowMajor, -1, -1>;

    // UB is 192KB = three [128,128] float tiles exactly, so tiles are reused by
    // lifetime: recvTile holds the incoming message, then out_prev, then
    // d (*) out_prev; sTile holds g_out then d; tmpTile aliases sTile's region
    // (sTile is dead once the dgamma product is formed). gamma/dgamma column
    // tiles are tiny and share the third region.
    TileF recvTile(block, dv);
    TileF sTile(block, dv);
    TileF tmpTile(block, dv);
    ColTile gammaTile(block, 1);
    ColTile dgammaTile(block, 1);
    TASSIGN(recvTile, 0x0);
    TASSIGN(sTile, 0x10000);
    TASSIGN(tmpTile, 0x10000);  // reuses sTile region (disjoint lifetime)
    TASSIGN(gammaTile, 0x20000);
    TASSIGN(dgammaTile, 0x21000);

    ShapeDyn blkShape(1, 1, 1, block, dv);
    StrideDyn blkStride(block * dv, block * dv, block * dv, dv, 1);
    ShapeDyn colShape(1, 1, 1, block, 1);
    StrideDyn colStride(block, block, block, 1, 1);

    for (int k = 0; k < K; ++k) {
        int row0 = k * block;
        int elt0 = row0 * dv;

        GlobalF gOutG(g_out + elt0, blkShape, blkStride);
        GlobalF dSG(dS + elt0, blkShape, blkStride);

        // --- form d[p] = g_out[p] (+ message from rank p+1) into sTile ---
        if (is_source) {
            TLOAD(sTile, gOutG);           // d = g_out (no incoming message)
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        } else {
            // Wait for the higher neighbour to land message block k. The g_out
            // load is issued *after* the barrier so it cannot race the previous
            // block's dS store (the terminal rank has no end-of-loop barrier).
            pto::comm::Signal sig(signal + k);
            pto::comm::TWAIT(sig, epoch, pto::comm::WaitCmp::GE);
            pipe_barrier(PIPE_ALL);

            GlobalF recvG(recv + elt0, blkShape, blkStride);
            TLOAD(recvTile, recvG);
            TLOAD(sTile, gOutG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TADD(sTile, sTile, recvTile);  // sTile = d[p]
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }

        // Publish d[p] to the dS output.
        TSTORE(dSG, sTile);

        if (is_terminal) {
            continue;  // rank 0: gamma[0] unused, dgamma[0] left at 0, nothing to send.
        }

        // Send the message first (short critical path, mirroring the forward
        // kernel: compute the outgoing block, store it remotely, flush, notify —
        // with no reduction work in between). dgamma is a purely local reduction
        // done afterwards. sTile (d) is preserved for both: the message reads it,
        // and the dgamma product reads it before tmpTile reuses its region.
        GlobalF outPrevG(out_prev + elt0, blkShape, blkStride);
        GlobalF gammaG(gamma + row0, colShape, colStride);
        GlobalF dgammaG(dgamma + row0, colShape, colStride);

        // --- message m = gamma[p] (*) d[p] -> recvTile, forwarded to rank p-1 ---
        TLOAD(gammaTile, gammaG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TROWEXPANDMUL(recvTile, sTile, gammaTile);  // gamma (*) d, per-row broadcast
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);

        int peer = my_rank - 1;
        __gm__ float *remote_recv = CommRemotePtr(commCtx, recv + elt0, peer);
        GlobalF remoteG(remote_recv, blkShape, blkStride);
        TSTORE(remoteG, recvTile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
        dcci((__gm__ int32_t *)remote_recv, ENTIRE_DATA_CACHE, CACHELINE_OUT);
#if defined(__CPU_SIM)
        dsb(0);
#else
        dsb(DSB_DDR);
#endif
        pipe_barrier(PIPE_ALL);
#endif
        __gm__ int32_t *peer_signal = CommRemotePtr(commCtx, signal + k, peer);
        pto::comm::Signal nsig(peer_signal);
        pto::comm::TNOTIFY(nsig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);

        // --- dgamma[p] = rowsum_dv( d[p] (*) out_prev ) ---
        // recvTile is free once the remote store has read it; reload out_prev.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        TLOAD(recvTile, outPrevG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TMUL(recvTile, sTile, recvTile);            // recvTile = d (*) out_prev; sTile dead after
        pipe_barrier(PIPE_V);                        // product ready before rowsum; tmp aliases dead sTile
        TROWSUM(dgammaTile, recvTile, tmpTile);     // per-row sum over dv -> [block, 1]
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(dgammaG, dgammaTile);

        // Iteration boundary (reached by source + middle ranks): flush all pipes
        // before the next block reloads sTile/recvTile/tmp (which share regions).
        // The source rank has no per-block TWAIT barrier to cover this otherwise.
        pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
}
