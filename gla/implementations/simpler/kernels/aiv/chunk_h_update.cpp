/*
 * GLA chunk_h state update (simpler runtime, Vector core).
 *
 * Carries the [K,V] recurrent state S across the chunk loop.  Per chunk:
 *   s_snap = S                      (state ENTERING this chunk)
 *   S      = decay[k] * S + KV      decay = exp(g_total), KV = k_rest^T @ v
 *
 * ``is_first`` (chunk 0): S entering is 0, so s_snap = 0 and S = KV — the
 * incoming S buffer is not read (may be uninitialised).  S is threaded as an
 * INOUT tensor so the runtime serialises the recurrence across chunks.
 *
 * Args (Tensor*): [0]=KV [K,V] IN, [1]=decay [K,1] IN, [2]=S [K,V] INOUT,
 *                 [3]=s_snap [K,V] OUT;  scalar[0]=is_first.
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
using Gm2Ddn = GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, Stride<1, 1, 1, DYNAMIC, 1>, Layout::DN>;
template <int R, int Cc>
using UbND = Tile<TileType::Vec, float, R, Cc, BLayout::RowMajor, R, Cc, SLayout::NoneBox, 512, PadValue::Null>;
template <int R, int Cc>
using UbDN = Tile<TileType::Vec, float, R, Cc, BLayout::ColMajor, R, Cc, SLayout::NoneBox, 512, PadValue::Null>;
}  // namespace

template <int K, int V>
static __aicore__ void update_impl(__gm__ float *kv, __gm__ float *decay, __gm__ float *s,
                                   __gm__ float *ssnap, uint64_t is_first) {
    Gm2D kvG(kv, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(K, V), Stride<1, 1, 1, DYNAMIC, 1>(V));
    Gm2D sG(s, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(K, V), Stride<1, 1, 1, DYNAMIC, 1>(V));
    Gm2D snapG(ssnap, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(K, V), Stride<1, 1, 1, DYNAMIC, 1>(V));
    Gm2Ddn decayG(decay, Shape<1, 1, 1, DYNAMIC, DYNAMIC>(K, 1), Stride<1, 1, 1, DYNAMIC, 1>(1));

    UbND<K, V> sT;      TASSIGN(sT, 0x0);
    UbND<K, V> kvT;     TASSIGN(kvT, K * V * 4);
    UbDN<K, 1> decayT;  TASSIGN(decayT, 2 * K * V * 4);

    TLOAD(kvT, kvG);

    if (is_first) {
        // s_snap = 0 ; S = KV
        TEXPANDS(sT, 0.0f);
        pipe_barrier(PIPE_ALL);         // kv load (MTE2) + zero (V) done before stores
        TSTORE(snapG, sT);
        TSTORE(sG, kvT);
        pipe_sync();
        return;
    }

    // s_snap = S (state entering this chunk) — snapshot BEFORE modifying sT.
    TLOAD(sT, sG);
    TLOAD(decayT, decayG);
    pipe_barrier(PIPE_ALL);             // all loads (MTE2) complete
    TSTORE(snapG, sT);
    pipe_barrier(PIPE_ALL);             // snapshot store (MTE3 reads sT) done before overwrite

    // S = decay * S + KV.  TROWEXPANDMUL broadcasts the [K,1] decay column
    // across the V columns and multiplies in one op (no [K,V] scratch buffer,
    // keeping UB within budget) — same pattern as the AllScan gamma multiply.
    TROWEXPANDMUL(sT, sT, decayT);
    pipe_barrier(PIPE_V);
    TADD(sT, sT, kvT);
    pipe_barrier(PIPE_ALL);             // update (V) done before final store (MTE3)
    TSTORE(sG, sT);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *kv = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *decay = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *s = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *ssnap = reinterpret_cast<__gm__ Tensor *>(args[3]);
    uint64_t is_first = static_cast<uint64_t>(args[4]);

    update_impl<128, 128>(
        reinterpret_cast<__gm__ float *>(kv->buffer.addr) + kv->start_offset,
        reinterpret_cast<__gm__ float *>(decay->buffer.addr) + decay->start_offset,
        reinterpret_cast<__gm__ float *>(s->buffer.addr) + s->start_offset,
        reinterpret_cast<__gm__ float *>(ssnap->buffer.addr) + ssnap->start_offset,
        is_first);
}
