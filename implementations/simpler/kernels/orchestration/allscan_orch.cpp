/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * AllScan orchestration shim — one AIV task per chip.
 *
 *   tensor(0) S_local  INPUT
 *   tensor(1) gamma    INPUT
 *   tensor(2) output   OUTPUT_EXISTING
 *   tensor(3) scratch  INOUT (HCCL window: recv region + per-block signals)
 *   scalar(0) dk
 *   scalar(1) dv
 *   scalar(2) K
 *   scalar(3) nranks
 *   scalar(4) epoch
 *   scalar(5) CommContext device pointer
 */

#include <stdint.h>

#include "pto_orchestration_api.h"

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
allscan_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 10,  // 4 tensors + 6 scalars
    };
}

__attribute__((visibility("default"))) void allscan_orchestration(const ChipStorageTaskArgs &orch_args) {
    Tensor s_local = from_tensor_arg(orch_args.tensor(0));
    Tensor gamma = from_tensor_arg(orch_args.tensor(1));
    Tensor output = from_tensor_arg(orch_args.tensor(2));
    Tensor scratch = from_tensor_arg(orch_args.tensor(3));

    Arg params;
    params.add_input(s_local);
    params.add_input(gamma);
    params.add_output(output);
    params.add_inout(scratch);
    params.add_scalar(orch_args.scalar(0));  // dk
    params.add_scalar(orch_args.scalar(1));  // dv
    params.add_scalar(orch_args.scalar(2));  // K
    params.add_scalar(orch_args.scalar(3));  // nranks
    params.add_scalar(orch_args.scalar(4));  // epoch
    params.add_scalar(orch_args.scalar(5));  // CommContext
    rt_submit_aiv_task(0, params);
}

}  // extern "C"
