/*
 * GLA gate_cumsum (simpler runtime, Cube core).
 *
 * Within-chunk inclusive prefix sum of the per-dim log-gates, computed as a
 * single triangular matmul per chunk:
 *
 *     g_cs = tril @ g          tril[t,s] = 1 for s <= t (lower-tri ones [C,C])
 *
 * so g_cs[t,d] = sum_{s<=t} g[s,d].  One [C,C] @ [C,D] matmul; the orchestration
 * (gate_cumsum_orch.cpp) submits one such task per chunk. D == C == TILE (128).
 *
 * The cube-matmul body (TLOAD -> TMOV to L0A/L0B -> TMATMUL -> TSTORE, with pipe
 * sync) is the pattern from examples/.../benchmark_bgemm/kernels/aic/
 * kernel_gemm_tile.cpp.
 *
 * Args (Tensor*): [0]=tril [C,C] IN, [1]=g_chunk [C,D] IN, [2]=g_cs_chunk [C,D] OUT.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T a, T b) {
    return (b == 0) ? 0 : (a + b - 1) / b * b;
}

template <int TILE>
static __aicore__ void trilmatmul_tile(__gm__ float *a, __gm__ float *b, __gm__ float *c) {
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(float);
    constexpr int M = CeilAlign<int>(TILE, 16);
    constexpr int K = CeilAlign<int>(TILE, blockAlign);
    constexpr int N = CeilAlign<int>(TILE, blockAlign);

    using GlobalData =
        GlobalTensor<float, Shape<1, 1, 1, TILE, TILE>, Stride<TILE * TILE, TILE * TILE, TILE * TILE, TILE, 1>>;
    GlobalData aG(a), bG(b), cG(c);

    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, M, K, TILE, TILE>;
    using RightTile = TileRight<float, K, N, TILE, TILE>;
    using AccTile = TileAcc<float, M, N, TILE, TILE>;

    TileMatA aMat;
    TileMatB bMat;
    TASSIGN(aMat, 0x0);
    TASSIGN(bMat, 0x20000);
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMat, aG);
    TLOAD(bMat, bG);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMat);
    TMOV(bTile, bMat);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(cG, cTile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a = reinterpret_cast<__gm__ Tensor *>(args[0]);  // tril [C,C]
    __gm__ Tensor *b = reinterpret_cast<__gm__ Tensor *>(args[1]);  // g_chunk [C,D]
    __gm__ Tensor *c = reinterpret_cast<__gm__ Tensor *>(args[2]);  // g_cs_chunk [C,D]
    int S = static_cast<int>(args[3]);                              // tile size (square: C==D)

    __gm__ float *ap = reinterpret_cast<__gm__ float *>(a->buffer.addr) + a->start_offset;
    __gm__ float *bp = reinterpret_cast<__gm__ float *>(b->buffer.addr) + b->start_offset;
    __gm__ float *cp = reinterpret_cast<__gm__ float *>(c->buffer.addr) + c->start_offset;

    // Runtime dispatch to a compile-time tile size (the benchmark_bgemm pattern):
    // the whole GLA pipeline is square when C==D, so one size drives every tile.
    switch (S) {
    case 16:  trilmatmul_tile<16>(ap, bp, cp);  break;
    case 32:  trilmatmul_tile<32>(ap, bp, cp);  break;
    case 64:  trilmatmul_tile<64>(ap, bp, cp);  break;
    default:  trilmatmul_tile<128>(ap, bp, cp); break;
    }
}
