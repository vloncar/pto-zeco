/*
 * GLA generalised square SxSxS matmul (simpler runtime, Cube core).
 *
 * out = op(A) @ op(B), all dims == TILE == S, fp32; the tile size S is a runtime
 * scalar dispatched to a compile-time template over {16,32,64,128} (square: the
 * whole GLA pipeline is SxSxS when chunk C == head D).  A ``mode`` scalar selects
 * the transpose variant so a single kernel covers every matmul in the GLA
 * pipeline:
 *
 *   mode 0 (NN): out[M,N] = A[M,Kc] @ B[Kc,N]          (q_eff@S, Aqk@v)
 *   mode 1 (TN): out[M,N] = A[Kc,M]^T @ B[Kc,N]        (k_rest^T@v  in chunk_h)
 *   mode 2 (NT): out[M,N] = A[M,Kc] @ B[N,Kc]^T        (q_eff@k_eff^T in chunk_o)
 *
 * Transpose is the single-shot L1->L0 pattern proven in the vendored
 * chunk_o_kda.cpp::gemm_oneshot (TRESHAPE the L1 tile to a ZN layout, then
 * TEXTRACT into L0A/L0B).  Every GLA matmul has inner dim == TILE <= 128 == one
 * L0 tile, so no K-slicing is needed.
 *
 * Args (Tensor*): [0]=A [TILE,TILE] IN, [1]=B [TILE,TILE] IN, [2]=C [TILE,TILE] OUT;
 *                 scalar[0]=mode.
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

// mode: 0=NN, 1=TN (transpose A), 2=NT (transpose B).  M==N==Kc==TILE.
template <int TILE, int MODE>
static __aicore__ void mm_impl(__gm__ float *a, __gm__ float *b, __gm__ float *c) {
    constexpr int M = TILE, N = TILE, Kc = TILE;

    GmData<TILE, TILE> aG(a), bG(b), cG(c);

    // A is physically [Kc,M] for TN, else [M,Kc]; B is [N,Kc] for NT, else [Kc,N].
    // Since every dim is TILE the physical GM shape is [TILE,TILE] either way.
    MatL1<TILE, TILE> aMat;
    MatL1<TILE, TILE> bMat;
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
        MatL1ZN<TILE, TILE> aT;
        TRESHAPE(aT, aMat);
        TEXTRACT(l0a, aT, 0, 0);
    } else {
        TEXTRACT(l0a, aMat, 0, 0);
    }
    if constexpr (MODE == 2) {
        // transpose B: reinterpret [N,Kc] -> [Kc,N] into L0B.
        MatL1ZN<TILE, TILE> bT;
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

// Dispatch the (runtime) transpose mode for a compile-time tile size.
template <int TILE>
static __aicore__ void mm_dispatch_mode(int mode, __gm__ float *a, __gm__ float *b, __gm__ float *c) {
    if (mode == 1) {
        mm_impl<TILE, 1>(a, b, c);
    } else if (mode == 2) {
        mm_impl<TILE, 2>(a, b, c);
    } else {
        mm_impl<TILE, 0>(a, b, c);
    }
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int mode = static_cast<int>(args[3]);
    int S = static_cast<int>(args[4]);  // tile size (square: C==D)

    __gm__ float *ap = reinterpret_cast<__gm__ float *>(a->buffer.addr) + a->start_offset;
    __gm__ float *bp = reinterpret_cast<__gm__ float *>(b->buffer.addr) + b->start_offset;
    __gm__ float *cp = reinterpret_cast<__gm__ float *>(c->buffer.addr) + c->start_offset;

    // Every GLA matmul is square (M==N==Kc==S) when C==D, so one size drives all
    // three transpose variants; runtime dispatch to a compile-time tile size.
    switch (S) {
    case 16:  mm_dispatch_mode<16>(mode, ap, bp, cp);  break;
    case 32:  mm_dispatch_mode<32>(mode, ap, bp, cp);  break;
    case 64:  mm_dispatch_mode<64>(mode, ap, bp, cp);  break;
    default:  mm_dispatch_mode<128>(mode, ap, bp, cp); break;
    }
}
