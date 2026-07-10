/*
 * GLA gate_cumsum orchestration (simpler runtime).
 *
 * Submits one Cube tril-matmul task per chunk: g_cs[n] = tril @ g[n], for
 * n = 0 .. N-1. Reuses the tensormap_and_ringbuffer orchestration API pattern
 * (see examples/.../benchmark_bgemm/kernels/orchestration/bgemm_orch.cpp).
 *
 * Orchestration args: [0]=tril [C,C] IN, [1]=g [N*C,D] IN, [2]=g_cs [N*C,D] OUT,
 *                     [3]=config int64[3] = [C, D, N] IN.
 * The Cube incore (gate_cumsum_kernel.cpp) takes 3 args: [tril, g_chunk, g_cs_chunk].
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_CUMSUM 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &tril = orch_args.tensor(0).ref();
    const Tensor &g = orch_args.tensor(1).ref();
    const Tensor &g_cs = orch_args.tensor(2).ref();

    int64_t *cfg = orch_args.tensor(3).ref().data_as<int64_t>();
    int C = static_cast<int>(cfg[0]);
    int Dd = static_cast<int>(cfg[1]);
    int N = static_cast<int>(cfg[2]);

    uint64_t chunk_elems = static_cast<uint64_t>(C) * Dd;
    uint32_t chunk_shape[1] = {static_cast<uint32_t>(chunk_elems)};

    for (int n = 0; n < N; n++) {
        PTO2_SCOPE_GUARD();
        uint32_t off[1] = {static_cast<uint32_t>(static_cast<uint64_t>(n) * chunk_elems)};
        Tensor g_view = g.view(chunk_shape, off);
        Tensor gcs_view = g_cs.view(chunk_shape, off);

        L0TaskArgs params;
        params.add_input(tril);
        params.add_input(g_view);
        params.add_output(gcs_view);
        params.add_scalar(static_cast<uint64_t>(C));   // M == K (tril is [C,C])
        params.add_scalar(static_cast<uint64_t>(Dd));  // N (g is [C,D])
        rt_submit_aic_task(FUNC_CUMSUM, params);
    }
}

}  // extern "C"
