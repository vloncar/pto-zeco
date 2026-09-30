#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os

import numpy as np

np.random.seed(19)


class MixedTileSizeParams:
    def __init__(self, m, k, n):
        self.m = m
        self.k = k
        self.n = n


def gen_golden_data(params):
    x1_gm = np.random.uniform(-1, 1, [params.m, params.k]).astype(np.float32)
    x2_gm = np.random.uniform(-1, 1, [params.k, params.n]).astype(np.float32)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    # Names must match the TEST_F(TPushPopMixedTileSizeTest, <name>) definitions in main.cpp.
    case_name_list = [
        "TPushPopMixedTileSizeTest.case1_unequal_64x64x32",
        "TPushPopMixedTileSizeTest.case2_equal_64x64x64",
        "TPushPopMixedTileSizeTest.case3_unequal_32x32x16",
        "TPushPopMixedTileSizeTest.case4_unequal_64x64x32_dir_both",
        "TPushPopMixedTileSizeTest.case5_equal_64x64x64_dir_both",
        "TPushPopMixedTileSizeTest.case6_unequal_32x32x16_dir_both",
    ]

    case_params_list = [
        MixedTileSizeParams(64, 64, 32),
        MixedTileSizeParams(64, 64, 64),
        MixedTileSizeParams(32, 32, 16),
        MixedTileSizeParams(64, 64, 32),
        MixedTileSizeParams(64, 64, 64),
        MixedTileSizeParams(32, 32, 16),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)
