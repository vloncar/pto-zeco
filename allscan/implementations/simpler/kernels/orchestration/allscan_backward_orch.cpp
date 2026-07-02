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
 * AllScan backward orchestration shim — one AIV task per chip.
 *
 *   tensor(0) g_out    INPUT
 *   tensor(1) gamma    INPUT
 *   tensor(2) out_prev INPUT
 *   tensor(3) dS       OUTPUT_EXISTING
 *   tensor(4) dgamma   OUTPUT_EXISTING
 *   tensor(5) scratch  INOUT (HCCL window: recv region + per-block signals)
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
allscan_backward_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 12,  // 6 tensors + 6 scalars
    };
}

__attribute__((visibility("default"))) void allscan_backward_orchestration(const L2TaskArgs &orch_args) {
    const Tensor &g_out = orch_args.tensor(0).ref();
    const Tensor &gamma = orch_args.tensor(1).ref();
    const Tensor &out_prev = orch_args.tensor(2).ref();
    const Tensor &dS = orch_args.tensor(3).ref();
    const Tensor &dgamma = orch_args.tensor(4).ref();
    const Tensor &scratch = orch_args.tensor(5).ref();

    L0TaskArgs params;
    params.add_input(g_out);
    params.add_input(gamma);
    params.add_input(out_prev);
    params.add_output(dS);
    params.add_output(dgamma);
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
