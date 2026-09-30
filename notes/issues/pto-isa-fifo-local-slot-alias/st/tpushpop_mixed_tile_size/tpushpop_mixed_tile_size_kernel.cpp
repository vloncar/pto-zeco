/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * Two DIFFERENTLY-SIZED tiles live in one cross-core ring at the same time.
 *
 * A TPipe's SLOT_SIZE is fixed at the pipe, but TPOP is templated per call site, so one ring
 * may carry tiles of different shapes. The consumer must still place slot i at
 * i * SLOT_SIZE in its local buffer. Striding instead by the popped tile's OWN size makes
 * slot 1 of a smaller tile land INSIDE slot 0 of a larger one, and the two tiles -- which
 * the FIFO considers to be in different slots -- share local memory.
 *
 * Here the vector core pushes both operands of a matmul; the cube pops both, then moves them
 * to Left/Right and multiplies. Holding two tiles at once uses 2 of the ring's 8 slots, which
 * is what a FIFO of depth > 1 is for. For [M,K] @ [K,N] the operands occupy M*K and K*N
 * elements at local slots 0 and 1, so a stride of "the tile's own size" overlaps them iff
 * K*N < M*K, i.e. N < M -- K cancels.
 *
 * case1/case3 (and case4/case6 over a DIR_BOTH pipe) are exactly that, and fail without the
 * SLOT_SIZE stride. case2/case5 push equal-sized tiles, where the two strides coincide, and
 * pass either way -- they are the control that isolates tile-size heterogeneity as the
 * trigger rather than cross-core transport, TMATMUL or TSTORE.
 */

#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

// Slot must hold the larger of the two operands; both cross the same ring.
template <typename T, int validM, int validK, int validN>
AICORE constexpr inline uint32_t SlotSizeFor()
{
    constexpr uint32_t aElems = validM * validK;
    constexpr uint32_t bElems = validK * validN;
    return (aElems > bElems ? aElems : bElems) * sizeof(T);
}

// Vector side: load both operands from GM and push them through the ring back to back.
template <typename MatPipe, typename VecTileA, typename VecTileB, typename GlobalA, typename GlobalB, typename T>
AICORE inline void PushBothOperands(MatPipe& mPipe, __gm__ T* srcA, __gm__ T* srcB)
{
    VecTileA vecA;
    VecTileB vecB;
    TASSIGN(vecA, 0x0);
    TASSIGN(vecB, 0x20000);

    GlobalA globalA(srcA);
    GlobalB globalB(srcB);

    // Flag ledger, per (pipe, pipe, event) triple. Each of the two pushes consumes the two
    // primed flags and re-arms them; the trailing pair of waits drains what the second push
    // re-armed. Net zero on every triple.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // ---- operand A ----
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(vecA, globalA);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<MatPipe, VecTileA, TileSplitAxis::TILE_NO_SPLIT>(mPipe, vecA);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    // ---- operand B (the smaller one when validN < validM) ----
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(vecB, globalB);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<MatPipe, VecTileB, TileSplitAxis::TILE_NO_SPLIT>(mPipe, vecB);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

// Cube side: pop both operands while both are still live in the ring, then multiply.
template <
    typename MatPipe, typename PopTileA, typename PopTileB, typename LeftTile, typename RightTile, typename AccTile,
    typename GlobalOut, typename T>
AICORE inline void PopBothAndMatmul(MatPipe& mPipe, __gm__ T* out)
{
    PopTileA aMatTile;
    PopTileB bMatTile;

    LeftTile aTile;
    RightTile bTile;
    AccTile accTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(accTile, 0x0);

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    // Both tiles are popped before either is consumed, so they must occupy distinct local
    // slots. No MTE1 -> MTE2 flag is needed between them for the same reason: with disjoint
    // slots the second TPOP (MTE2) and the first TMOV (MTE1) touch different memory.
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TPOP<MatPipe, PopTileA, TileSplitAxis::TILE_NO_SPLIT>(mPipe, aMatTile);
    TPOP<MatPipe, PopTileB, TileSplitAxis::TILE_NO_SPLIT>(mPipe, bMatTile);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(mPipe);
    TFREE<MatPipe, TileSplitAxis::TILE_NO_SPLIT>(mPipe);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

    TMATMUL(accTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    GlobalOut globalOut(out);
    TSTORE<AccTile, GlobalOut>(globalOut, accTile);

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
}

template <typename T, int validM, int validK, int validN, uint8_t DIR = Direction::DIR_V2C>
__global__ AICORE void runTPushPopMixedTileSize(
    __gm__ uint64_t* ffts_addr, __gm__ T* out, __gm__ T* srcA, __gm__ T* srcB, __gm__ T* fifoMem)
{
    set_ffts_base_addr((uint64_t)ffts_addr);

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint32_t SLOT_SIZE = SlotSizeFor<T, validM, validK, validN>();
    constexpr uint32_t SLOT_NUM = 8;
    constexpr uint32_t LOCAL_SLOT_NUM = 8;
    constexpr uint32_t localFiFoBase = 0x0;

    // DIR is the ONLY difference between the V2C and DIR_BOTH cases: same data flow (vector
    // produces both operands, cube consumes them), same slot geometry, same tiles. A DIR_BOTH
    // pipe driven in one direction only is well-defined here -- the destructor drains the
    // producer side alone, and the unused C2V half is simply never signalled -- but it does
    // need TWO rings of GM, which the host allocates (see main.cpp).
    using MatPipe = TPipe<FLAG_ID, DIR, SLOT_SIZE, SLOT_NUM, LOCAL_SLOT_NUM, true>;
    MatPipe mPipe((__gm__ void*)fifoMem, 0x0, localFiFoBase);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(validM, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(validK, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(validN, blockAlign);

    using GlobalA = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalB = GlobalTensor<
        T, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GlobalOut = GlobalTensor<
        T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    using VecTileA = Tile<TileType::Vec, T, validM, validK, BLayout::RowMajor, validM, validK>;
    using VecTileB = Tile<TileType::Vec, T, validK, validN, BLayout::RowMajor, validK, validN>;

    using PopTileA =
        Tile<TileType::Mat, T, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using PopTileB =
        Tile<TileType::Mat, T, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<T, ALIGNED_M, ALIGNED_K, validM, validK>;
    using RightTile = TileRight<T, ALIGNED_K, ALIGNED_N, validK, validN>;
    using AccTile = TileAcc<T, validM, validN, validM, validN>;

    if constexpr (DAV_VEC) {
        PushBothOperands<MatPipe, VecTileA, VecTileB, GlobalA, GlobalB, T>(mPipe, srcA, srcB);
        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_CUBE) {
        PopBothAndMatmul<MatPipe, PopTileA, PopTileB, LeftTile, RightTile, AccTile, GlobalOut, T>(mPipe, out);
        pipe_barrier(PIPE_ALL);
    }
}

template <int32_t tilingKey>
void LaunchTPushPopMixedTileSize(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream)
{
    if constexpr (tilingKey == 1) {
        // Unequal tiles: 64x64 @ 64x32 -> 64x32, operands via the V2C pipe.
        runTPushPopMixedTileSize<float, 64, 64, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 2) {
        // Equal-sized tiles on the identical path. Control.
        runTPushPopMixedTileSize<float, 64, 64, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 3) {
        // Unequal tiles at a second shape.
        runTPushPopMixedTileSize<float, 32, 32, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 4) {
        // Same as case1 but over a DIR_BOTH pipe instead of DIR_V2C.
        runTPushPopMixedTileSize<float, 64, 64, 32, Direction::DIR_BOTH><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 5) {
        // DIR_BOTH equal-sized control.
        runTPushPopMixedTileSize<float, 64, 64, 64, Direction::DIR_BOTH><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    } else if constexpr (tilingKey == 6) {
        // Same as case3 but over a DIR_BOTH pipe.
        runTPushPopMixedTileSize<float, 32, 32, 16, Direction::DIR_BOTH><<<1, nullptr, stream>>>(
            reinterpret_cast<uint64_t*>(ffts), reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcA),
            reinterpret_cast<float*>(srcB), reinterpret_cast<float*>(fifoMem));
    }
}

template void LaunchTPushPopMixedTileSize<1>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMixedTileSize<2>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMixedTileSize<3>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMixedTileSize<4>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMixedTileSize<5>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
template void LaunchTPushPopMixedTileSize<6>(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);
