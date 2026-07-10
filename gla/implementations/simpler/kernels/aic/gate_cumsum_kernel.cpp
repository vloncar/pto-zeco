/*
 * GLA gate_cumsum (simpler runtime, Cube core).
 *
 * Within-chunk inclusive prefix sum of the per-dim log-gates, computed as a
 * single triangular matmul per chunk:
 *
 *     g_cs = tril @ g          tril[t,s] = 1 for s <= t (lower-tri ones [C,C])
 *
 * so g_cs[t,d] = sum_{s<=t} g[s,d].  One [C,C] @ [C,D] matmul (NN, M=C, N=D,
 * K=C); the orchestration (gate_cumsum_orch.cpp) submits one such task per chunk.
 * C and D are runtime scalars, each dispatched to a compile-time template over
 * {16,32,64,128} (C == D reduces to the square case).
 *
 * The cube-matmul body (TLOAD -> TMOV to L0A/L0B -> TMATMUL -> TSTORE, with pipe
 * sync) is the pattern from examples/.../benchmark_bgemm/kernels/aic/
 * kernel_gemm_tile.cpp.
 *
 * Args (Tensor*): [0]=tril [C,C] IN, [1]=g_chunk [C,D] IN, [2]=g_cs_chunk [C,D] OUT;
 *                 scalar[0]=C, scalar[1]=D.
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

// out[rM,rN] = tril[rM,rK] @ g[rK,rN]  (NN).  For gate_cumsum rM == rK == C, rN == D.
template <int rM, int rN, int rK>
static __aicore__ void trilmatmul_tile(__gm__ float *a, __gm__ float *b, __gm__ float *c) {
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(float);
    constexpr int M = CeilAlign<int>(rM, 16);
    constexpr int K = CeilAlign<int>(rK, blockAlign);
    constexpr int N = CeilAlign<int>(rN, blockAlign);

    using GlobalA = GlobalTensor<float, Shape<1, 1, 1, rM, rK>, Stride<rM * rK, rM * rK, rM * rK, rK, 1>>;
    using GlobalB = GlobalTensor<float, Shape<1, 1, 1, rK, rN>, Stride<rK * rN, rK * rN, rK * rN, rN, 1>>;
    using GlobalC = GlobalTensor<float, Shape<1, 1, 1, rM, rN>, Stride<rM * rN, rM * rN, rM * rN, rN, 1>>;
    GlobalA aG(a);
    GlobalB bG(b);
    GlobalC cG(c);

    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, rM, rK, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, rK, rN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, M, K, rM, rK>;
    using RightTile = TileRight<float, K, N, rK, rN>;
    using AccTile = TileAcc<float, M, N, rM, rN>;

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

// Dispatch N (=D) for a compile-time C (drives both M and K, since tril is [C,C]).
template <int C>
static __aicore__ void trilmatmul_by_d(int d, __gm__ float *a, __gm__ float *b, __gm__ float *c) {
    switch (d) {
    case 16:  trilmatmul_tile<C, 16, C>(a, b, c);  break;
    case 32:  trilmatmul_tile<C, 32, C>(a, b, c);  break;
    case 64:  trilmatmul_tile<C, 64, C>(a, b, c);  break;
    default:  trilmatmul_tile<C, 128, C>(a, b, c); break;
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a = reinterpret_cast<__gm__ Tensor *>(args[0]);  // tril [C,C]
    __gm__ Tensor *b = reinterpret_cast<__gm__ Tensor *>(args[1]);  // g_chunk [C,D]
    __gm__ Tensor *c = reinterpret_cast<__gm__ Tensor *>(args[2]);  // g_cs_chunk [C,D]
    int C = static_cast<int>(args[3]);
    int D = static_cast<int>(args[4]);

    __gm__ float *ap = reinterpret_cast<__gm__ float *>(a->buffer.addr) + a->start_offset;
    __gm__ float *bp = reinterpret_cast<__gm__ float *>(b->buffer.addr) + b->start_offset;
    __gm__ float *cp = reinterpret_cast<__gm__ float *>(c->buffer.addr) + c->start_offset;

    // Runtime dispatch of C (M,K) and D (N) to compile-time tile sizes
    // (the benchmark_bgemm pattern, extended to a rectangular tril @ g).
    switch (C) {
    case 16:  trilmatmul_by_d<16>(D, ap, bp, cp);  break;
    case 32:  trilmatmul_by_d<32>(D, ap, bp, cp);  break;
    case 64:  trilmatmul_by_d<64>(D, ap, bp, cp);  break;
    default:  trilmatmul_by_d<128>(D, ap, bp, cp); break;
    }
}
