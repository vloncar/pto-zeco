/*
 * GLA chunk_o elementwise (simpler runtime, Vector core).
 *
 * out = A (op) B, elementwise over [R,Cc].  mode 0 = multiply (Aqk * tril mask,
 * shape [C,C]), mode 1 = add (o = inter + intra, shape [C,D]).  R and Cc are
 * runtime scalars each dispatched to a compile-time template over {16,32,64,128}.
 *
 * Args (Tensor*): [0]=A IN, [1]=B IN, [2]=out OUT;  scalar[0]=mode,
 *                 scalar[1]=R, scalar[2]=Cc.
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

template <int R, int Cc>
static __aicore__ void elt_impl(__gm__ float *a, __gm__ float *b, __gm__ float *o, int mode) {
    Gm2D aG(a, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(R, Cc), Stride<1, 1, 1, DYNAMIC, 1>(Cc));
    Gm2D bG(b, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(R, Cc), Stride<1, 1, 1, DYNAMIC, 1>(Cc));
    Gm2D oG(o, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(R, Cc), Stride<1, 1, 1, DYNAMIC, 1>(Cc));

    Ub<R, Cc> aT;  TASSIGN(aT, 0x0);
    Ub<R, Cc> bT;  TASSIGN(bT, R * Cc * 4);

    TLOAD(aT, aG);
    TLOAD(bT, bG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (mode == 1) {
        TADD(aT, aT, bT);
    } else {
        TMUL(aT, aT, bT);
    }
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(oG, aT);
    pipe_sync();
}

template <int R>
static __aicore__ void elt_by_cc(int cc, __gm__ float *a, __gm__ float *b, __gm__ float *o, int mode) {
    switch (cc) {
    case 16:  elt_impl<R, 16>(a, b, o, mode);   break;
    case 32:  elt_impl<R, 32>(a, b, o, mode);   break;
    case 64:  elt_impl<R, 64>(a, b, o, mode);   break;
    default:  elt_impl<R, 128>(a, b, o, mode);  break;
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *o = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int mode = static_cast<int>(args[3]);
    int R = static_cast<int>(args[4]);   // rows
    int Cc = static_cast<int>(args[5]);  // cols

    __gm__ float *ap = reinterpret_cast<__gm__ float *>(a->buffer.addr) + a->start_offset;
    __gm__ float *bp = reinterpret_cast<__gm__ float *>(b->buffer.addr) + b->start_offset;
    __gm__ float *op = reinterpret_cast<__gm__ float *>(o->buffer.addr) + o->start_offset;

    switch (R) {
    case 16:  elt_by_cc<16>(Cc, ap, bp, op, mode);   break;
    case 32:  elt_by_cc<32>(Cc, ap, bp, op, mode);   break;
    case 64:  elt_by_cc<64>(Cc, ap, bp, op, mode);   break;
    default:  elt_by_cc<128>(Cc, ap, bp, op, mode);  break;
    }
}
