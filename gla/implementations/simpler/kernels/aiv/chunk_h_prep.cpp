/*
 * GLA chunk_h prep (simpler runtime, Vector core).
 *
 * Per chunk, from the within-chunk cumulative gate g_cs and keys k, produce the
 * decayed keys and the state-decay column:
 *
 *   coeff[c,k] = exp(g_total[k] - g_cs[c,k])      g_total = g_cs[C-1, :]
 *   k_rest[c,k] = k[c,k] * coeff[c,k]                              -> [C,K]
 *   decay[k]    = exp(g_total[k])                                  -> [K,1]
 *
 * k_rest feeds the Cube matmul KV = k_rest^T @ v; decay row-scales the carried
 * state in chunk_h_update.  All fp32; K == D (head dim), C == chunk size, each a
 * runtime scalar dispatched to a compile-time template over {16,32,64,128}.
 *
 * Args (Tensor*): [0]=g_cs [C,K] IN, [1]=g_total [1,K] IN, [2]=k [C,K] IN,
 *                 [3]=k_rest [C,K] OUT, [4]=decay [K,1] OUT;
 *                 scalar[0]=C, scalar[1]=D(=K).
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
static __aicore__ void prep_impl(__gm__ float *gcs, __gm__ float *gtot, __gm__ float *k,
                                 __gm__ float *krest, __gm__ float *decay) {
    Gm2D gcsG(gcs, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D gtotG(gtot, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(1, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D kG(k, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D krestG(krest, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(C, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    Gm2D decayG(decay, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(K, 1), Stride<1, 1, 1, DYNAMIC, 1>(1));

    Ub<C, K> gcsT;      TASSIGN(gcsT, 0x0);
    Ub<C, K> coeffT;    TASSIGN(coeffT, C * K * 4);
    Ub<1, K> gtotT;     TASSIGN(gtotT, 2 * C * K * 4);
    Ub<K, 8> decayT;    TASSIGN(decayT, 2 * C * K * 4 + K * 4);  // [K,1] valid, 8-wide capacity

    TLOAD(gcsT, gcsG);
    TLOAD(gtotT, gtotG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // coeff = exp(g_total_broadcast - g_cs)
    TCOLEXPAND(coeffT, gtotT);          // broadcast g_total[1,K] down C rows
    pipe_barrier(PIPE_ALL);             // TCOLEXPAND lowers to a ubuf copy (not a pure
                                        // PIPE_V op); a PIPE_V-only barrier can let TSUB
                                        // read coeffT before the copy lands at S<128.
    TSUB(coeffT, coeffT, gcsT);
    pipe_barrier(PIPE_V);
    TEXP(coeffT, coeffT);
    pipe_barrier(PIPE_V);

    // k_rest = k * coeff  (reuse gcsT slot to load k)
    TLOAD(gcsT, kG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(coeffT, coeffT, gcsT);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(krestG, coeffT);

    // decay = exp(g_total)  (K values, stored as [K,1] column)
    Ub<1, K> decayRow;  TASSIGN(decayRow, 2 * C * K * 4 + K * 4);
    TEXP(decayRow, gtotT);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // K contiguous floats: same memory whether viewed [1,K] or [K,1].
    Gm2D decayRowG(decay, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(1, K), Stride<1, 1, 1, DYNAMIC, 1>(K));
    TSTORE(decayRowG, decayRow);
    pipe_sync();
}

template <int C>
static __aicore__ void prep_by_d(int d, __gm__ float *gcs, __gm__ float *gtot, __gm__ float *k,
                                 __gm__ float *krest, __gm__ float *decay) {
    switch (d) {
    case 16:  prep_impl<C, 16>(gcs, gtot, k, krest, decay);   break;
    case 32:  prep_impl<C, 32>(gcs, gtot, k, krest, decay);   break;
    case 64:  prep_impl<C, 64>(gcs, gtot, k, krest, decay);   break;
    default:  prep_impl<C, 128>(gcs, gtot, k, krest, decay);  break;
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *gcs = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *gtot = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *k = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *krest = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *decay = reinterpret_cast<__gm__ Tensor *>(args[4]);
    int C = static_cast<int>(args[5]);  // chunk size (rows)
    int D = static_cast<int>(args[6]);  // head dim K (cols)

    __gm__ float *gcsp = reinterpret_cast<__gm__ float *>(gcs->buffer.addr) + gcs->start_offset;
    __gm__ float *gtotp = reinterpret_cast<__gm__ float *>(gtot->buffer.addr) + gtot->start_offset;
    __gm__ float *kp = reinterpret_cast<__gm__ float *>(k->buffer.addr) + k->start_offset;
    __gm__ float *krestp = reinterpret_cast<__gm__ float *>(krest->buffer.addr) + krest->start_offset;
    __gm__ float *decayp = reinterpret_cast<__gm__ float *>(decay->buffer.addr) + decay->start_offset;

    switch (C) {
    case 16:  prep_by_d<16>(D, gcsp, gtotp, kp, krestp, decayp);   break;
    case 32:  prep_by_d<32>(D, gcsp, gtotp, kp, krestp, decayp);   break;
    case 64:  prep_by_d<64>(D, gcsp, gtotp, kp, krestp, decayp);   break;
    default:  prep_by_d<128>(D, gcsp, gtotp, kp, krestp, decayp);  break;
    }
}
