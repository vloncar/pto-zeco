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
 * AllScan kernel — direct PTO-runtime ("simpler") port of pto-allscan/allscan.py.
 *
 * Sequential scan over P ranks arranged in a ring:
 *
 *     out[0]   = S_local[0]
 *     out[p]   = S_local[p] + gamma[p] (*) out[p-1]      (p = 1 .. P-1)
 *
 * where (*) is a per-row broadcast multiply: gamma is [dk, 1] and broadcasts
 * across the dv columns of the [dk, dv] state.  Work is pipelined over K
 * blocks of BLOCK = dk/K rows so that rank p+1 can start block k as soon as
 * rank p has forwarded block k (matches the K-deep pipeline in allscan.py).
 *
 * One uniform kernel runs on every rank; behaviour is selected from rankId:
 *   rank 0          : source — emit S_local, push to rank 1, no wait.
 *   rank 1..P-2     : receive from prev, fuse, push to next.
 *   rank P-1        : receive from prev, fuse, no push (chain terminates).
 *
 * Synchronisation uses the symmetric HCCL window (CommContext): each rank
 * forwards its computed block straight into the *next* rank's recv slot with a
 * remote TSTORE, then bumps that rank's per-block signal with TNOTIFY. The
 * receiver TWAITs on its own signal before reading. Signals are AtomicAdd'd by
 * exactly 1 per run, and the window is zeroed at domain-allocation time, so the
 * caller passes the 1-based run index as `epoch` and we wait for `signal >=
 * epoch`. This keeps the kernel correct across repeated runs against a
 * persistent domain (benchmark reuse) without re-zeroing the window each call.
 *
 * args layout (see allscan_orch.cpp):
 *   tensor(0) S_local  [dk, dv]            INPUT
 *   tensor(1) gamma    [dk, 1]             INPUT
 *   tensor(2) output   [dk, dv]            OUTPUT_EXISTING
 *   tensor(3) scratch  recv[dk*dv] + sig[K] INOUT  (HCCL window)
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

// Max tile capacity. dk and dv are <= 128 in every benchmark config; a block
// is at most [dk, dv] (K == 1).  The valid extent is set per-block at runtime.
static constexpr int kMaxRows = 128;
static constexpr int kMaxCols = 128;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[pe] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *s_local_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *gamma_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    int dk = static_cast<int>(args[4]);
    int dv = static_cast<int>(args[5]);
    int K = static_cast<int>(args[6]);
    int nranks = static_cast<int>(args[7]);
    int32_t epoch = static_cast<int32_t>(args[8]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[9]);

    __gm__ float *s_local =
        reinterpret_cast<__gm__ float *>(s_local_tensor->buffer.addr) + s_local_tensor->start_offset;
    __gm__ float *gamma = reinterpret_cast<__gm__ float *>(gamma_tensor->buffer.addr) + gamma_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    // scratch = recv region (dk*dv floats) followed by K int32 signal slots.
    __gm__ float *recv = reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;
    __gm__ int32_t *signal = reinterpret_cast<__gm__ int32_t *>(recv + dk * dv);

    int my_rank = static_cast<int>(commCtx->rankId);
    int last_rank = nranks - 1;
    int block = dk / K;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GlobalF = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileF = pto::Tile<pto::TileType::Vec, float, kMaxRows, kMaxCols, pto::BLayout::RowMajor, -1, -1>;
    // Tile cols must be 32-byte aligned (multiple of 8 floats); gamma is a
    // single logical column, so use an 8-wide capacity with valid cols = 1.
    using GammaTile = pto::Tile<pto::TileType::Vec, float, kMaxRows, 8, pto::BLayout::RowMajor, -1, -1>;

    TileF recvTile(block, dv);
    TileF sTile(block, dv);
    GammaTile gammaTile(block, 1);
    TASSIGN(recvTile, 0x0);
    TASSIGN(sTile, 0x10000);
    TASSIGN(gammaTile, 0x20000);

    // [block, dv] view; first three (size-1) dims carry the block stride.
    ShapeDyn blkShape(1, 1, 1, block, dv);
    StrideDyn blkStride(block * dv, block * dv, block * dv, dv, 1);
    // [block, 1] view for the per-row gamma column.
    ShapeDyn gamShape(1, 1, 1, block, 1);
    StrideDyn gamStride(block, block, block, 1, 1);

    for (int k = 0; k < K; ++k) {
        int row0 = k * block;
        int elt0 = row0 * dv;

        GlobalF sLocalG(s_local + elt0, blkShape, blkStride);
        GlobalF outputG(output + elt0, blkShape, blkStride);

        if (my_rank == 0) {
            // Source rank: out = S_local, nothing to receive.
            TLOAD(sTile, sLocalG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        } else {
            // Wait for the predecessor to land block k in our recv slot.
            pto::comm::Signal sig(signal + k);
            pto::comm::TWAIT(sig, epoch, pto::comm::WaitCmp::GE);
            pipe_barrier(PIPE_ALL);

            GlobalF recvG(recv + elt0, blkShape, blkStride);
            GlobalF gammaG(gamma + row0, gamShape, gamStride);
            TLOAD(recvTile, recvG);
            TLOAD(gammaTile, gammaG);
            TLOAD(sTile, sLocalG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // recv := gamma (*) recv  (per-row broadcast), then s := s_local + recv.
            TROWEXPANDMUL(recvTile, recvTile, gammaTile);
            TADD(sTile, sTile, recvTile);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }

        // Publish this rank's result block to the local output.
        TSTORE(outputG, sTile);

        if (my_rank != last_rank) {
            // Forward the block straight into the next rank's recv slot, then
            // signal it. The store must be globally visible before the notify,
            // hence the cache flush + barrier (hardware paths only; on sim the
            // window is shared host memory and the store is already visible).
            int peer = my_rank + 1;
            __gm__ float *remote_recv = CommRemotePtr(commCtx, recv + elt0, peer);
            GlobalF remoteG(remote_recv, blkShape, blkStride);
            TSTORE(remoteG, sTile);
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
        }
    }
    pipe_barrier(PIPE_ALL);
}
