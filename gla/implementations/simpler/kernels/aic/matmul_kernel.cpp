/*
 * GLA generalised (rectangular) matmul (simpler runtime, Cube core).
 *
 * out = op(A) @ op(B), out is [M,N], contraction dim Kc, fp32.  M, N, Kc are
 * runtime scalars each dispatched to a compile-time template over {16,32,64,128}
 * (all dims are C or D in the GLA pipeline, both in that set).  A ``mode`` scalar
 * selects the transpose variant so a single kernel covers every GLA matmul:
 *
 *   mode 0 (NN): out[M,N] = A[M,Kc] @ B[Kc,N]          (q_eff@S, Aqk@v)
 *   mode 1 (TN): out[M,N] = A[Kc,M]^T @ B[Kc,N]        (k_rest^T@v  in chunk_h)
 *   mode 2 (NT): out[M,N] = A[M,Kc] @ B[N,Kc]^T        (q_eff@k_eff^T in chunk_o)
 *
 * The GLA matmul shapes (M,N,Kc) are: KV=(D,D,C) TN, inter=(C,D,D) NN,
 * Aqk=(C,C,D) NT, intra=(C,D,C) NN.  When C==D they are all square SxSxS.
 *
 * Every dim is <= 128 (one L0 tile), so no blocking is needed.  Tiles > 128 (head
 * dim 256) are a follow-up (F3 Phase 3) blocked on fp32 cube K-accumulation being
 * unsupported on a2a3 — see allscan/issues/fp32-cube-k-accumulation/.
 *
 * Transpose is done by loading the operand row-major into an L1 "Mat" tile
 * (ColMajor block / RowMajor sub) and TRESHAPE-ing it to the transposed ZN
 * layout (RowMajor block / ColMajor sub) before TEXTRACT into L0A/L0B — the
 * recipe proven in the square kernel and in chunk_o_kda.cpp::gemm_oneshot.
 *
 * Args (Tensor*): [0]=A IN, [1]=B IN, [2]=C OUT;
 *                 scalar[0]=mode, scalar[1]=M, scalar[2]=N, scalar[3]=Kc.
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

namespace {

template <int R, int Cc>
using GmData = GlobalTensor<float, Shape<1, 1, 1, R, Cc>, Stride<R * Cc, R * Cc, R * Cc, Cc, 1>>;

// L1 "Mat" tile: ColMajor block layout, RowMajor sub-layout (loaded from a
// row-major GM buffer).  ZN is the transposed reinterpretation.
template <int R, int Cc>
using MatL1 = Tile<TileType::Mat, float, R, Cc, BLayout::ColMajor, R, Cc, SLayout::RowMajor, 512, PadValue::Zero>;
template <int R, int Cc>
using MatL1ZN = Tile<TileType::Mat, float, R, Cc, BLayout::RowMajor, R, Cc, SLayout::ColMajor, 512, PadValue::Zero>;
template <int R, int Cc>
using L0A = Tile<TileType::Left, float, R, Cc, BLayout::RowMajor, R, Cc, SLayout::RowMajor, 512, PadValue::Zero>;
template <int R, int Cc>
using L0B = Tile<TileType::Right, float, R, Cc, BLayout::RowMajor, R, Cc, SLayout::ColMajor, 512, PadValue::Zero>;

// mode: 0=NN, 1=TN (transpose A), 2=NT (transpose B).  out[M,N], contraction Kc.
template <int M, int N, int Kc, int MODE>
static __aicore__ void mm_impl(__gm__ float *a, __gm__ float *b, __gm__ float *c) {
    // Physical GM shape of each operand (row-major): A is [Kc,M] for TN else
    // [M,Kc]; B is [N,Kc] for NT else [Kc,N]; C is [M,N].
    constexpr int AR = (MODE == 1) ? Kc : M;
    constexpr int AC = (MODE == 1) ? M : Kc;
    constexpr int BR = (MODE == 2) ? N : Kc;
    constexpr int BC = (MODE == 2) ? Kc : N;

    GmData<AR, AC> aG(a);
    GmData<BR, BC> bG(b);
    GmData<M, N> cG(c);

    MatL1<AR, AC> aMat;
    MatL1<BR, BC> bMat;
    TASSIGN(aMat, 0x0);
    TASSIGN(bMat, 0x20000);

    L0A<M, Kc> l0a;
    L0B<Kc, N> l0b;
    TASSIGN(l0a, 0x0);
    TASSIGN(l0b, 0x0);
    TileAcc<float, M, N, M, N> cTile;
    TASSIGN(cTile, 0x0);

    TLOAD(aMat, aG);
    TLOAD(bMat, bG);
    auto ev = (event_t)(((int)EVENT_ID0 + 1) % 8);
    set_flag(PIPE_MTE2, PIPE_MTE1, ev);
    wait_flag(PIPE_MTE2, PIPE_MTE1, ev);
    set_flag(PIPE_M, PIPE_MTE1, ev);
    wait_flag(PIPE_M, PIPE_MTE1, ev);

    if constexpr (MODE == 1) {
        // transpose A: reinterpret [Kc,M] -> [M,Kc] into L0A.
        MatL1ZN<M, Kc> aT;
        TRESHAPE(aT, aMat);
        TEXTRACT(l0a, aT, 0, 0);
    } else {
        TEXTRACT(l0a, aMat, 0, 0);
    }
    if constexpr (MODE == 2) {
        // transpose B: reinterpret [N,Kc] -> [Kc,N] into L0B.
        MatL1ZN<Kc, N> bT;
        TRESHAPE(bT, bMat);
        TEXTRACT(l0b, bT, 0, 0);
    } else {
        TEXTRACT(l0b, bMat, 0, 0);
    }

    set_flag(PIPE_MTE1, PIPE_M, ev);
    wait_flag(PIPE_MTE1, PIPE_M, ev);
    TMATMUL(cTile, l0a, l0b);
    set_flag(PIPE_M, PIPE_FIX, ev);
    wait_flag(PIPE_M, PIPE_FIX, ev);
    TSTORE(cG, cTile);
    pipe_sync();
}

// Dispatch the (runtime) transpose mode for compile-time M,N,Kc.
template <int M, int N, int Kc>
static __aicore__ void mm_by_mode(int mode, __gm__ float *a, __gm__ float *b, __gm__ float *c) {
    if (mode == 1) {
        mm_impl<M, N, Kc, 1>(a, b, c);
    } else if (mode == 2) {
        mm_impl<M, N, Kc, 2>(a, b, c);
    } else {
        mm_impl<M, N, Kc, 0>(a, b, c);
    }
}

// Runtime -> compile-time size dispatch, one level per dim, over {16,32,64,128}.
template <int M, int N>
static __aicore__ void mm_by_kc(int kc, int mode, __gm__ float *a, __gm__ float *b, __gm__ float *c) {
    switch (kc) {
    case 16:  mm_by_mode<M, N, 16>(mode, a, b, c);  break;
    case 32:  mm_by_mode<M, N, 32>(mode, a, b, c);  break;
    case 64:  mm_by_mode<M, N, 64>(mode, a, b, c);  break;
    default:  mm_by_mode<M, N, 128>(mode, a, b, c); break;
    }
}
template <int M>
static __aicore__ void mm_by_n(int n, int kc, int mode, __gm__ float *a, __gm__ float *b, __gm__ float *c) {
    switch (n) {
    case 16:  mm_by_kc<M, 16>(kc, mode, a, b, c);  break;
    case 32:  mm_by_kc<M, 32>(kc, mode, a, b, c);  break;
    case 64:  mm_by_kc<M, 64>(kc, mode, a, b, c);  break;
    default:  mm_by_kc<M, 128>(kc, mode, a, b, c); break;
    }
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int mode = static_cast<int>(args[3]);
    int M = static_cast<int>(args[4]);
    int N = static_cast<int>(args[5]);
    int Kc = static_cast<int>(args[6]);

    __gm__ float *ap = reinterpret_cast<__gm__ float *>(a->buffer.addr) + a->start_offset;
    __gm__ float *bp = reinterpret_cast<__gm__ float *>(b->buffer.addr) + b->start_offset;
    __gm__ float *cp = reinterpret_cast<__gm__ float *>(c->buffer.addr) + c->start_offset;

    // Runtime dispatch of each dim (M,N,Kc) to a compile-time tile size, then the
    // transpose mode; covers every rectangular GLA matmul with M,N,Kc in {16..128}.
    switch (M) {
    case 16:  mm_by_n<16>(N, Kc, mode, ap, bp, cp);  break;
    case 32:  mm_by_n<32>(N, Kc, mode, ap, bp, cp);  break;
    case 64:  mm_by_n<64>(N, Kc, mode, ap, bp, cp);  break;
    default:  mm_by_n<128>(N, Kc, mode, ap, bp, cp); break;
    }
}
