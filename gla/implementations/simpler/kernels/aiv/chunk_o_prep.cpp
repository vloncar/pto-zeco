/*
 * GLA chunk_o prep (simpler runtime, Vector core).
 *
 * Pre-scale queries/keys by the within-chunk cumulative gate:
 *   q_eff[c,k] = q[c,k] * exp(g_cs[c,k])      (exp(g_cs) <= 1, bounded)
 *   k_eff[c,k] = k[c,k] * exp(-g_cs[c,k])     (can be large; kept fp32)
 *
 * Args (Tensor*): [0]=q [C,K] IN, [1]=k [C,K] IN, [2]=g_cs [C,K] IN,
 *                 [3]=q_eff [C,K] OUT, [4]=k_eff [C,K] OUT;
 *                 scalar[0]=C, scalar[1]=D(=K).  C and D each dispatch over
 *                 {16,32,64,128} (C == D reduces to the square case).
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

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
using Gm2D = GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, Stride<1, 1, 1, DYNAMIC, 1>>;
template <int R, int Cc>
using Ub = Tile<TileType::Vec, float, R, Cc, BLayout::RowMajor, R, Cc, SLayout::NoneBox, 512, PadValue::Null>;
}  // namespace

template <int C, int K>
static __aicore__ void prep_impl(__gm__ float *q, __gm__ float *k, __gm__ float *gcs,
                                 __gm__ float *qeff, __gm__ float *keff) {
    Gm2D qG(q, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D kG(k, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D gcsG(gcs, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D qeffG(qeff, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D keffG(keff, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));

    Ub<C, K> gT;    TASSIGN(gT, 0x0);              // g_cs, then exp(-g_cs)
    Ub<C, K> eT;    TASSIGN(eT, C * K * 4);        // exp(g_cs)
    Ub<C, K> xT;    TASSIGN(xT, 2 * C * K * 4);    // q, then k

    TLOAD(gT, gcsG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TEXP(eT, gT);                    // eT = exp(g_cs)
    pipe_barrier(PIPE_V);

    // q_eff = q * exp(g_cs)
    TLOAD(xT, qG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(xT, xT, eT);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(qeffG, xT);

    // exp(-g_cs) into gT
    TMULS(gT, gT, -1.0f);
    pipe_barrier(PIPE_V);
    TEXP(gT, gT);
    pipe_barrier(PIPE_V);

    // k_eff = k * exp(-g_cs)  (reuse xT slot for k)
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(xT, kG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(xT, xT, gT);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(keffG, xT);
    pipe_sync();
}

template <int C>
static __aicore__ void prep_by_d(int d, __gm__ float *q, __gm__ float *k, __gm__ float *gcs,
                                 __gm__ float *qeff, __gm__ float *keff) {
    switch (d) {
    case 16:  prep_impl<C, 16>(q, k, gcs, qeff, keff);   break;
    case 32:  prep_impl<C, 32>(q, k, gcs, qeff, keff);   break;
    case 64:  prep_impl<C, 64>(q, k, gcs, qeff, keff);   break;
    default:  prep_impl<C, 128>(q, k, gcs, qeff, keff);  break;
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *q = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *k = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *gcs = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *qeff = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *keff = reinterpret_cast<__gm__ Tensor *>(args[4]);
    int C = static_cast<int>(args[5]);  // chunk size (rows)
    int D = static_cast<int>(args[6]);  // head dim K (cols)

    __gm__ float *qp = reinterpret_cast<__gm__ float *>(q->buffer.addr) + q->start_offset;
    __gm__ float *kp = reinterpret_cast<__gm__ float *>(k->buffer.addr) + k->start_offset;
    __gm__ float *gcsp = reinterpret_cast<__gm__ float *>(gcs->buffer.addr) + gcs->start_offset;
    __gm__ float *qeffp = reinterpret_cast<__gm__ float *>(qeff->buffer.addr) + qeff->start_offset;
    __gm__ float *keffp = reinterpret_cast<__gm__ float *>(keff->buffer.addr) + keff->start_offset;

    switch (C) {
    case 16:  prep_by_d<16>(D, qp, kp, gcsp, qeffp, keffp);   break;
    case 32:  prep_by_d<32>(D, qp, kp, gcsp, qeffp, keffp);   break;
    case 64:  prep_by_d<64>(D, qp, kp, gcsp, qeffp, keffp);   break;
    default:  prep_by_d<128>(D, qp, kp, gcsp, qeffp, keffp);  break;
    }
}
