/*
 * GLA grad_h orchestration (simpler runtime) — the backward of chunk_h.
 *
 * The state-stage adjoint of the ZeCO backward.  The cross-chunk reverse
 * recurrence for the state adjoint dSloc / decay adjoint dcvec is the cheap
 * linear glue and runs on host (mirroring the forward's _S_total / _shift_snaps);
 * each chunk's dSloc[n] is then handed to this kernel, so chunks are INDEPENDENT
 * here (like chunk_o), no on-device recurrence.  Per chunk, recompute
 * Kstate=k*exp(g_total-g_cs) (= k_rest, reusing chunk_h_prep) and run two matmuls:
 *
 *   Kstate,_ = prep(g_cs, g_total, k)                (aiv; decay output ignored)
 *   dKstate  = v @ dSloc^T        [C,dk]             (aic NT)
 *   dVs      = Kstate @ dSloc     [C,dv]  (state dV) (aic NN)
 *
 * dKstate / dVs are the RAW adjoints; the host finishes them (dk_state =
 * dKstate*exp(g_total-g_cs), dg_cs_state = -dk_state*k + the g_total row
 * correction, dV += dVs) — the linear glue.  Reuses matmul / chunk_h_prep, so
 * grad_h adds NO new device kernel.
 *
 * Args: [0]=k [N*C,dk] IN, [1]=v [N*C,dv] IN, [2]=g_cs [N*C,dk] IN,
 *       [3]=dSloc [N,dk,dv] IN, [4]=dKstate [N*C,dk] OUT, [5]=dVs [N*C,dv] OUT,
 *       [6]=config int64[4]={C,dk,dv,N} IN.
 */

#include <stddef.h>
#include <stdint.h>
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MM 0
#define FUNC_PREP 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 7};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &k = orch_args.tensor(0).ref();
    const Tensor &v = orch_args.tensor(1).ref();
    const Tensor &gcs = orch_args.tensor(2).ref();
    const Tensor &dSloc = orch_args.tensor(3).ref();
    const Tensor &dKstate = orch_args.tensor(4).ref();
    const Tensor &dVs = orch_args.tensor(5).ref();

    int64_t *cfg = orch_args.tensor(6).ref().data_as<int64_t>();
    int C = static_cast<int>(cfg[0]);
    int dk = static_cast<int>(cfg[1]);
    int dv = static_cast<int>(cfg[2]);
    int N = static_cast<int>(cfg[3]);

    // k/g_cs/dKstate are [C,dk]; v/dVs are [C,dv]; dSloc/Kstate-state is [dk,dv].
    uint32_t cdk_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(dk)};
    uint32_t cdv_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(dv)};
    uint32_t kv_flat[1] = {static_cast<uint32_t>(dk) * static_cast<uint32_t>(dv)};
    uint32_t row_flat[1] = {static_cast<uint32_t>(dk)};

    uint32_t cdk_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(dk)};
    uint32_t dec_shape[2] = {static_cast<uint32_t>(dk), 1};

    TensorCreateInfo krest_ci(cdk_shape, 2, DataType::FLOAT32);
    TensorCreateInfo decay_ci(dec_shape, 2, DataType::FLOAT32);

    for (int n = 0; n < N; n++) {
        PTO2_SCOPE_GUARD();
        uint32_t nu = static_cast<uint32_t>(n);
        uint32_t cdk = static_cast<uint32_t>(C) * dk;
        uint32_t cdv = static_cast<uint32_t>(C) * dv;
        uint32_t kv = static_cast<uint32_t>(dk) * dv;
        uint32_t dk_off[1] = {nu * cdk};
        uint32_t dv_off[1] = {nu * cdv};
        uint32_t kv_off[1] = {nu * kv};
        uint32_t gtot_off[1] = {nu * cdk + (static_cast<uint32_t>(C) - 1) * dk};

        Tensor k_n = k.view(cdk_flat, dk_off);
        Tensor v_n = v.view(cdv_flat, dv_off);
        Tensor gcs_n = gcs.view(cdk_flat, dk_off);
        Tensor gtot_n = gcs.view(row_flat, gtot_off);
        Tensor dSloc_n = dSloc.view(kv_flat, kv_off);
        Tensor dKstate_n = dKstate.view(cdk_flat, dk_off);
        Tensor dVs_n = dVs.view(cdv_flat, dv_off);

        // prep -> k_rest (Kstate) [C,dk], decay [dk,1] (decay ignored here)
        L0TaskArgs p_prep;
        p_prep.add_input(gcs_n);
        p_prep.add_input(gtot_n);
        p_prep.add_input(k_n);
        p_prep.add_output(krest_ci);
        p_prep.add_output(decay_ci);
        p_prep.add_scalar(static_cast<uint64_t>(C));
        p_prep.add_scalar(static_cast<uint64_t>(dk));
        TaskOutputTensors prep_outs = rt_submit_aiv_task(FUNC_PREP, p_prep);
        const Tensor &Kstate = prep_outs.get_ref(0);

        // dKstate = v @ dSloc^T  (NT): [C,dv]@[dv,dk] = [C,dk], M=C,N=dk,Kc=dv
        L0TaskArgs p_dks;
        p_dks.add_input(v_n);
        p_dks.add_input(dSloc_n);
        p_dks.add_inout(dKstate_n);
        p_dks.add_scalar(static_cast<uint64_t>(2));   // NT
        p_dks.add_scalar(static_cast<uint64_t>(C));
        p_dks.add_scalar(static_cast<uint64_t>(dk));
        p_dks.add_scalar(static_cast<uint64_t>(dv));
        rt_submit_aic_task(FUNC_MM, p_dks);

        // dVs = Kstate @ dSloc  (NN): [C,dk]@[dk,dv] = [C,dv], M=C,N=dv,Kc=dk
        L0TaskArgs p_dvs;
        p_dvs.add_input(Kstate);
        p_dvs.add_input(dSloc_n);
        p_dvs.add_inout(dVs_n);
        p_dvs.add_scalar(static_cast<uint64_t>(0));   // NN
        p_dvs.add_scalar(static_cast<uint64_t>(C));
        p_dvs.add_scalar(static_cast<uint64_t>(dv));
        p_dvs.add_scalar(static_cast<uint64_t>(dk));
        rt_submit_aic_task(FUNC_MM, p_dvs);
    }
}

}  // extern "C"
