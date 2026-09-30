/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopMixedTileSize(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* fifoMem, void* stream);

#define ASSERT_ACL_OK(expr)                                             \
    do {                                                                \
        const auto ret = (expr);                                        \
        ASSERT_EQ(ret, ACL_SUCCESS) << #expr << " failed, ret=" << ret; \
    } while (0)

#define ASSERT_RT_OK(expr)                                                \
    do {                                                                  \
        const auto ret = (expr);                                          \
        ASSERT_EQ(ret, RT_ERROR_NONE) << #expr << " failed, ret=" << ret; \
    } while (0)

class TPushPopMixedTileSizeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

struct MixedTileSizeBuffers {
    uint8_t* dstHost = nullptr;
    uint8_t* src0Host = nullptr;
    uint8_t* src1Host = nullptr;
    uint8_t* dstDevice = nullptr;
    uint8_t* src0Device = nullptr;
    uint8_t* src1Device = nullptr;
    uint8_t* fifoDevice = nullptr;
};

void FreeBuffers(MixedTileSizeBuffers& buf)
{
    aclrtFree(buf.dstDevice);
    aclrtFree(buf.src0Device);
    aclrtFree(buf.src1Device);
    aclrtFree(buf.fifoDevice);
    aclrtFreeHost(buf.dstHost);
    aclrtFreeHost(buf.src0Host);
    aclrtFreeHost(buf.src1Host);
}

// Allocates host and device memory and loads the two operands from the generated input data.
// Any failure leaves `ok` false, so the caller tears down instead of running the kernel on
// buffers that were never populated -- a matmul over stale memory would compare as a
// meaningless pass or fail rather than reporting the real cause.
void PrepareBuffers(MixedTileSizeBuffers& buf, size_t aSize, size_t bSize, size_t cSize, size_t fifoSize, bool& ok)
{
    ok = false;
    ASSERT_ACL_OK(aclrtMallocHost((void**)(&buf.dstHost), cSize));
    ASSERT_ACL_OK(aclrtMallocHost((void**)(&buf.src0Host), aSize));
    ASSERT_ACL_OK(aclrtMallocHost((void**)(&buf.src1Host), bSize));
    ASSERT_ACL_OK(aclrtMalloc((void**)&buf.dstDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ASSERT_ACL_OK(aclrtMalloc((void**)&buf.src0Device, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ASSERT_ACL_OK(aclrtMalloc((void**)&buf.src1Device, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ASSERT_ACL_OK(aclrtMalloc((void**)&buf.fifoDevice, fifoSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/x1_gm.bin", aSize, buf.src0Host, aSize)) << "missing x1_gm.bin";
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/x2_gm.bin", bSize, buf.src1Host, bSize)) << "missing x2_gm.bin";

    ASSERT_ACL_OK(aclrtMemcpy(buf.src0Device, aSize, buf.src0Host, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ASSERT_ACL_OK(aclrtMemcpy(buf.src1Device, bSize, buf.src1Host, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ok = true;
}

// Runs the kernel and brings the result back. Kept separate from the caller so that a failed
// check returns from here and the caller still reaches its teardown -- an ASSERT_ in the
// middle of the allocating function would skip the frees and leak the device buffers.
template <int32_t key>
void RunKernel(MixedTileSizeBuffers& buf, size_t cSize, aclrtStream stream)
{
    uint64_t ffts{0};
    uint32_t fftsLen{0};
    // A zero FFTS base would silently misdirect every cross-core sync in the kernel.
    ASSERT_RT_OK(rtGetC2cCtrlAddr(&ffts, &fftsLen));

    LaunchTPushPopMixedTileSize<key>(
        (uint8_t*)ffts, buf.dstDevice, buf.src0Device, buf.src1Device, buf.fifoDevice, stream);

    ASSERT_ACL_OK(aclrtSynchronizeStream(stream));
    ASSERT_ACL_OK(aclrtMemcpy(buf.dstHost, cSize, buf.dstDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    ASSERT_TRUE(WriteFile(GetGoldenDir() + "/output_z.bin", buf.dstHost, cSize));
}

template <typename T>
void CompareWithGolden(size_t elemCount, size_t cSize, uint32_t M, uint32_t K, uint32_t N)
{
    std::vector<T> golden(elemCount);
    std::vector<T> devFinal(elemCount);
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/golden.bin", cSize, golden.data(), cSize));
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/output_z.bin", cSize, devFinal.data(), cSize));

    bool ret = ResultCmp(golden, devFinal, 0.0001f);
    EXPECT_TRUE(ret) << "M=" << M << " K=" << K << " N=" << N << (N != M ? "  (operand tiles differ in size)" : "");
}

// rings = 1 for a DIR_V2C pipe, 2 for DIR_BOTH (which lays the V2C ring after the C2V one,
// so the GM buffer must cover both even when only one direction is driven).
template <typename T, int32_t key, uint32_t rings = 1>
void TPushPopMixedTileSizeTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(T);
    size_t bFileSize = K * N * sizeof(T);
    size_t cFileSize = M * N * sizeof(T);
    // SLOT_NUM (8) slots, each big enough for the larger operand -- must match the kernel.
    size_t fifoSize = rings * 8 * std::max(aFileSize, bFileSize);

    ASSERT_ACL_OK(aclInit(nullptr));
    ASSERT_ACL_OK(aclrtSetDevice(0));
    aclrtStream stream;
    ASSERT_ACL_OK(aclrtCreateStream(&stream));

    MixedTileSizeBuffers buf;
    bool ready = false;
    PrepareBuffers(buf, aFileSize, bFileSize, cFileSize, fifoSize, ready);
    if (ready) {
        RunKernel<key>(buf, cFileSize, stream);
    }

    FreeBuffers(buf);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    ASSERT_TRUE(ready);
    CompareWithGolden<T>(M * N, cFileSize, M, K, N);
}

// A[64,64] and B[64,32] differ in size, so they must not share a local slot.
TEST_F(TPushPopMixedTileSizeTest, case1_unequal_64x64x32) { TPushPopMixedTileSizeTestFunc<float, 1>(64, 64, 32); }

// Equal-sized operands on the identical path. Control.
TEST_F(TPushPopMixedTileSizeTest, case2_equal_64x64x64) { TPushPopMixedTileSizeTestFunc<float, 2>(64, 64, 64); }

// A second pair of unequal tiles.
TEST_F(TPushPopMixedTileSizeTest, case3_unequal_32x32x16) { TPushPopMixedTileSizeTestFunc<float, 3>(32, 32, 16); }

// The same three cases over a DIR_BOTH pipe instead of DIR_V2C. Everything else -- data flow,
// slot geometry, tile types, flag ledger -- is identical, so the outcome tracks the local slot
// stride rather than the pipe direction.
TEST_F(TPushPopMixedTileSizeTest, case4_unequal_64x64x32_dir_both)
{
    TPushPopMixedTileSizeTestFunc<float, 4, 2>(64, 64, 32);
}

TEST_F(TPushPopMixedTileSizeTest, case5_equal_64x64x64_dir_both)
{
    TPushPopMixedTileSizeTestFunc<float, 5, 2>(64, 64, 64);
}

TEST_F(TPushPopMixedTileSizeTest, case6_unequal_32x32x16_dir_both)
{
    TPushPopMixedTileSizeTestFunc<float, 6, 2>(32, 32, 16);
}
