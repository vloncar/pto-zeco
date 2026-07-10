/*
 * Standalone test orchestration for matmul_kernel.cpp — submits a single
 * S x S x S matmul in the mode given by config[0] (0=NN,1=TN,2=NT), tile size
 * config[1] (defaults to 128 if absent/zero).
 *
 * Args: [0]=A IN, [1]=B IN, [2]=C OUT, [3]=config int64[>=1]={mode[,S]} IN.
 */
#include <stddef.h>
#include <stdint.h>
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MM 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &a = orch_args.tensor(0).ref();
    const Tensor &b = orch_args.tensor(1).ref();
    const Tensor &c = orch_args.tensor(2).ref();
    int64_t *cfg = orch_args.tensor(3).ref().data_as<int64_t>();
    uint64_t mode = static_cast<uint64_t>(cfg[0]);
    uint64_t S = static_cast<uint64_t>(cfg[1]);

    PTO2_SCOPE_GUARD();
    L0TaskArgs params;
    params.add_input(a);
    params.add_input(b);
    params.add_output(c);
    params.add_scalar(mode);
    params.add_scalar(S);
    rt_submit_aic_task(FUNC_MM, params);
}

}  // extern "C"
