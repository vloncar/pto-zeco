/*
 * GLA chunk_o orchestration (simpler runtime).
 *
 * Per chunk (chunks are independent — each reads its own snapshot S_n):
 *   q_eff, k_eff = prep(q,k,g_cs)                     (aiv)
 *   inter = q_eff @ S_n                               (aic, NN)
 *   Aqk   = q_eff @ k_eff^T                           (aic, NT)
 *   Aqk_m = Aqk * tril                                (aiv, mul)
 *   intra = Aqk_m @ v                                 (aic, NN)
 *   o_n   = inter + intra                             (aiv, add)
 *
 * S_n = s_snap[n] is the (S_recv-folded) snapshot state entering chunk n; the
 * cross-device history was folded on host (linearity), so v_corr == v here.
 *
 * Args: [0]=q [N*C,K] IN, [1]=k [N*C,K] IN, [2]=v [N*C,V] IN, [3]=g_cs [N*C,K] IN,
 *       [4]=s_snap [N,K,V] IN, [5]=tril [C,C] IN, [6]=o [N*C,V] OUT,
 *       [7]=config int64[3]={C,D,N} IN.
 */

#include <stddef.h>
#include <stdint.h>
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MM 0
#define FUNC_PREP 1
#define FUNC_ELT 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 8};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &q = orch_args.tensor(0).ref();
    const Tensor &k = orch_args.tensor(1).ref();
    const Tensor &v = orch_args.tensor(2).ref();
    const Tensor &gcs = orch_args.tensor(3).ref();
    const Tensor &ssnap = orch_args.tensor(4).ref();
    const Tensor &tril = orch_args.tensor(5).ref();
    const Tensor &o = orch_args.tensor(6).ref();

    int64_t *cfg = orch_args.tensor(7).ref().data_as<int64_t>();
    int C = static_cast<int>(cfg[0]);
    int D = static_cast<int>(cfg[1]);
    int N = static_cast<int>(cfg[2]);

    uint32_t ck_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(D)};
    uint32_t kv_flat[1] = {static_cast<uint32_t>(D) * static_cast<uint32_t>(D)};

    uint32_t ck_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(D)};
    uint32_t cc_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(C)};

    TensorCreateInfo qeff_ci(ck_shape, 2, DataType::FLOAT32);
    TensorCreateInfo keff_ci(ck_shape, 2, DataType::FLOAT32);
    TensorCreateInfo inter_ci(ck_shape, 2, DataType::FLOAT32);
    TensorCreateInfo aqk_ci(cc_shape, 2, DataType::FLOAT32);
    TensorCreateInfo aqkm_ci(cc_shape, 2, DataType::FLOAT32);
    TensorCreateInfo intra_ci(ck_shape, 2, DataType::FLOAT32);

    for (int n = 0; n < N; n++) {
        PTO2_SCOPE_GUARD();
        uint32_t nu = static_cast<uint32_t>(n);
        uint32_t ck = static_cast<uint32_t>(C) * D;
        uint32_t kv = static_cast<uint32_t>(D) * D;
        uint32_t ck_off[1] = {nu * ck};
        uint32_t kv_off[1] = {nu * kv};

        Tensor q_n = q.view(ck_flat, ck_off);
        Tensor k_n = k.view(ck_flat, ck_off);
        Tensor v_n = v.view(ck_flat, ck_off);
        Tensor gcs_n = gcs.view(ck_flat, ck_off);
        Tensor s_n = ssnap.view(kv_flat, kv_off);
        Tensor o_n = o.view(ck_flat, ck_off);

        // prep -> q_eff, k_eff
        L0TaskArgs p_prep;
        p_prep.add_input(q_n);
        p_prep.add_input(k_n);
        p_prep.add_input(gcs_n);
        p_prep.add_output(qeff_ci);
        p_prep.add_output(keff_ci);
        p_prep.add_scalar(static_cast<uint64_t>(C));  // tile size S (square: C==D)
        TaskOutputTensors prep_outs = rt_submit_aiv_task(FUNC_PREP, p_prep);
        const Tensor &qeff = prep_outs.get_ref(0);
        const Tensor &keff = prep_outs.get_ref(1);

        // inter = q_eff @ S_n  (NN)
        L0TaskArgs p_inter;
        p_inter.add_input(qeff);
        p_inter.add_input(s_n);
        p_inter.add_output(inter_ci);
        p_inter.add_scalar(static_cast<uint64_t>(0));  // mode NN
        p_inter.add_scalar(static_cast<uint64_t>(C));  // tile size S (square: C==D)
        TaskOutputTensors inter_outs = rt_submit_aic_task(FUNC_MM, p_inter);
        const Tensor &inter = inter_outs.get_ref(0);

        // Aqk = q_eff @ k_eff^T  (NT)
        L0TaskArgs p_aqk;
        p_aqk.add_input(qeff);
        p_aqk.add_input(keff);
        p_aqk.add_output(aqk_ci);
        p_aqk.add_scalar(static_cast<uint64_t>(2));  // mode NT
        p_aqk.add_scalar(static_cast<uint64_t>(C));  // tile size S (square: C==D)
        TaskOutputTensors aqk_outs = rt_submit_aic_task(FUNC_MM, p_aqk);
        const Tensor &aqk = aqk_outs.get_ref(0);

        // Aqk_m = Aqk * tril  (mul)
        L0TaskArgs p_mask;
        p_mask.add_input(aqk);
        p_mask.add_input(tril);
        p_mask.add_output(aqkm_ci);
        p_mask.add_scalar(static_cast<uint64_t>(0));  // mode mul
        p_mask.add_scalar(static_cast<uint64_t>(C));  // tile size S (square: C==D)
        TaskOutputTensors mask_outs = rt_submit_aiv_task(FUNC_ELT, p_mask);
        const Tensor &aqkm = mask_outs.get_ref(0);

        // intra = Aqk_m @ v  (NN)
        L0TaskArgs p_intra;
        p_intra.add_input(aqkm);
        p_intra.add_input(v_n);
        p_intra.add_output(intra_ci);
        p_intra.add_scalar(static_cast<uint64_t>(0));  // mode NN
        p_intra.add_scalar(static_cast<uint64_t>(C));  // tile size S (square: C==D)
        TaskOutputTensors intra_outs = rt_submit_aic_task(FUNC_MM, p_intra);
        const Tensor &intra = intra_outs.get_ref(0);

        // o_n = inter + intra  (add)
        L0TaskArgs p_out;
        p_out.add_input(inter);
        p_out.add_input(intra);
        p_out.add_inout(o_n);
        p_out.add_scalar(static_cast<uint64_t>(1));  // mode add
        p_out.add_scalar(static_cast<uint64_t>(C));  // tile size S (square: C==D)
        rt_submit_aiv_task(FUNC_ELT, p_out);
    }
}

}  // extern "C"
