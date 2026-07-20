/*
 * GLA grad_o orchestration (simpler runtime) — the backward of chunk_o.
 *
 * The output-stage adjoint of the ZeCO backward (see gla.common.gla_chunk_backward
 * / the kernel-form reference).  Per chunk (chunks independent — each reads its own
 * folded snapshot H_n and dO_n), recompute the forward's q_eff/k_eff/scores, then
 * run the seven backward matmuls.  All map onto the general (M,N,Kc,mode) matmul
 * kernel, so grad_o adds NO new device kernel — it reuses matmul / chunk_o_prep /
 * chunk_o_elt.  Let Qt=q_eff=q*exp(g_cs), Kin=k_eff=k*exp(-g_cs),
 * scores=(Qt@Kin^T)*tril, H_n=snapshot state (folded on host):
 *
 *   Qt,Kin      = prep(q,k,g_cs)                              (aiv)
 *   Aqk         = Qt @ Kin^T          -> scores = Aqk*tril    (aic NT, aiv mul)
 *   dH_n        = Qt^T @ dO           [dk,dv]                 (aic TN)
 *   dsc         = (dO @ v^T) * tril   [C,C]                   (aic NT, aiv mul)
 *   dQt         = dO @ H_n^T + dsc @ Kin   [C,dk]            (aic NT, aic NN, aiv add)
 *   dKin        = dsc^T @ Qt          [C,dk]                  (aic TN)
 *   dVi         = scores^T @ dO       [C,dv]  (intra dV)      (aic TN)
 *
 * The cheap gate elementwise that finishes the chunk (dq=dQt*exp(g_cs),
 * dk_intra=dKin*exp(-g_cs), dg_cs_o=dq*q-dk_intra*k) is done on host — the linear
 * glue, mirroring the forward's _S_total/_shift_snaps.  dH_n feeds the host reverse
 * recurrence that produces dSloc for grad_h.
 *
 * Args: [0]=q [N*C,dk] IN, [1]=k [N*C,dk] IN, [2]=v [N*C,dv] IN, [3]=g_cs [N*C,dk] IN,
 *       [4]=snap [N,dk,dv] IN (folded H), [5]=dO [N*C,dv] IN, [6]=tril [C,C] IN,
 *       [7]=dQt [N*C,dk] OUT, [8]=dKin [N*C,dk] OUT, [9]=dVi [N*C,dv] OUT,
 *       [10]=dH [N,dk,dv] OUT, [11]=config int64[4]={C,dk,dv,N} IN.
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
    return PTO2OrchestrationConfig{.expected_arg_count = 12};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &q = orch_args.tensor(0).ref();
    const Tensor &k = orch_args.tensor(1).ref();
    const Tensor &v = orch_args.tensor(2).ref();
    const Tensor &gcs = orch_args.tensor(3).ref();
    const Tensor &snap = orch_args.tensor(4).ref();
    const Tensor &dO = orch_args.tensor(5).ref();
    const Tensor &tril = orch_args.tensor(6).ref();
    const Tensor &dQt = orch_args.tensor(7).ref();
    const Tensor &dKin = orch_args.tensor(8).ref();
    const Tensor &dVi = orch_args.tensor(9).ref();
    const Tensor &dH = orch_args.tensor(10).ref();

    int64_t *cfg = orch_args.tensor(11).ref().data_as<int64_t>();
    int C = static_cast<int>(cfg[0]);
    int dk = static_cast<int>(cfg[1]);
    int dv = static_cast<int>(cfg[2]);
    int N = static_cast<int>(cfg[3]);

    // q/k/g_cs/dQt/dKin are [C,dk]; v/dO/dVi are [C,dv]; Aqk/scores/dsc are [C,C];
    // the snapshot/dH state is [dk,dv] — the same three widths as chunk_o.
    uint32_t cdk_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(dk)};
    uint32_t cdv_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(dv)};
    uint32_t kv_flat[1] = {static_cast<uint32_t>(dk) * static_cast<uint32_t>(dv)};

    uint32_t cdk_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(dk)};
    uint32_t cdv_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(dv)};
    uint32_t cc_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(C)};
    uint32_t kv_shape[2] = {static_cast<uint32_t>(dk), static_cast<uint32_t>(dv)};

    TensorCreateInfo qeff_ci(cdk_shape, 2, DataType::FLOAT32);
    TensorCreateInfo keff_ci(cdk_shape, 2, DataType::FLOAT32);
    TensorCreateInfo aqk_ci(cc_shape, 2, DataType::FLOAT32);
    TensorCreateInfo scores_ci(cc_shape, 2, DataType::FLOAT32);
    TensorCreateInfo dscraw_ci(cc_shape, 2, DataType::FLOAT32);
    TensorCreateInfo dsc_ci(cc_shape, 2, DataType::FLOAT32);
    TensorCreateInfo dqtr_ci(cdk_shape, 2, DataType::FLOAT32);
    TensorCreateInfo dqti_ci(cdk_shape, 2, DataType::FLOAT32);
    TensorCreateInfo dH_ci(kv_shape, 2, DataType::FLOAT32);

    for (int n = 0; n < N; n++) {
        PTO2_SCOPE_GUARD();
        uint32_t nu = static_cast<uint32_t>(n);
        uint32_t cdk = static_cast<uint32_t>(C) * dk;
        uint32_t cdv = static_cast<uint32_t>(C) * dv;
        uint32_t kv = static_cast<uint32_t>(dk) * dv;
        uint32_t dk_off[1] = {nu * cdk};
        uint32_t dv_off[1] = {nu * cdv};
        uint32_t kv_off[1] = {nu * kv};

        Tensor q_n = q.view(cdk_flat, dk_off);
        Tensor k_n = k.view(cdk_flat, dk_off);
        Tensor v_n = v.view(cdv_flat, dv_off);
        Tensor gcs_n = gcs.view(cdk_flat, dk_off);
        Tensor H_n = snap.view(kv_flat, kv_off);
        Tensor dO_n = dO.view(cdv_flat, dv_off);
        Tensor dQt_n = dQt.view(cdk_flat, dk_off);
        Tensor dKin_n = dKin.view(cdk_flat, dk_off);
        Tensor dVi_n = dVi.view(cdv_flat, dv_off);
        Tensor dH_n = dH.view(kv_flat, kv_off);

        // prep -> q_eff (Qt) [C,dk], k_eff (Kin) [C,dk]
        L0TaskArgs p_prep;
        p_prep.add_input(q_n);
        p_prep.add_input(k_n);
        p_prep.add_input(gcs_n);
        p_prep.add_output(qeff_ci);
        p_prep.add_output(keff_ci);
        p_prep.add_scalar(static_cast<uint64_t>(C));
        p_prep.add_scalar(static_cast<uint64_t>(dk));
        TaskOutputTensors prep_outs = rt_submit_aiv_task(FUNC_PREP, p_prep);
        const Tensor &Qt = prep_outs.get_ref(0);
        const Tensor &Kin = prep_outs.get_ref(1);

        // Aqk = Qt @ Kin^T  (NT): [C,dk]@[dk,C] = [C,C], M=C,N=C,Kc=dk
        L0TaskArgs p_aqk;
        p_aqk.add_input(Qt);
        p_aqk.add_input(Kin);
        p_aqk.add_output(aqk_ci);
        p_aqk.add_scalar(static_cast<uint64_t>(2));   // NT
        p_aqk.add_scalar(static_cast<uint64_t>(C));
        p_aqk.add_scalar(static_cast<uint64_t>(C));
        p_aqk.add_scalar(static_cast<uint64_t>(dk));
        TaskOutputTensors aqk_outs = rt_submit_aic_task(FUNC_MM, p_aqk);
        const Tensor &aqk = aqk_outs.get_ref(0);

        // scores = Aqk * tril  (mul): [C,C]
        L0TaskArgs p_sc;
        p_sc.add_input(aqk);
        p_sc.add_input(tril);
        p_sc.add_output(scores_ci);
        p_sc.add_scalar(static_cast<uint64_t>(0));   // mul
        p_sc.add_scalar(static_cast<uint64_t>(C));
        p_sc.add_scalar(static_cast<uint64_t>(C));
        TaskOutputTensors sc_outs = rt_submit_aiv_task(FUNC_ELT, p_sc);
        const Tensor &scores = sc_outs.get_ref(0);

        // dH_n = Qt^T @ dO  (TN): [dk,C]@[C,dv] = [dk,dv], M=dk,N=dv,Kc=C
        L0TaskArgs p_dH;
        p_dH.add_input(Qt);
        p_dH.add_input(dO_n);
        p_dH.add_inout(dH_n);
        p_dH.add_scalar(static_cast<uint64_t>(1));   // TN
        p_dH.add_scalar(static_cast<uint64_t>(dk));
        p_dH.add_scalar(static_cast<uint64_t>(dv));
        p_dH.add_scalar(static_cast<uint64_t>(C));
        rt_submit_aic_task(FUNC_MM, p_dH);

        // dsc_raw = dO @ v^T  (NT): [C,dv]@[dv,C] = [C,C], M=C,N=C,Kc=dv
        L0TaskArgs p_dscr;
        p_dscr.add_input(dO_n);
        p_dscr.add_input(v_n);
        p_dscr.add_output(dscraw_ci);
        p_dscr.add_scalar(static_cast<uint64_t>(2));   // NT
        p_dscr.add_scalar(static_cast<uint64_t>(C));
        p_dscr.add_scalar(static_cast<uint64_t>(C));
        p_dscr.add_scalar(static_cast<uint64_t>(dv));
        TaskOutputTensors dscr_outs = rt_submit_aic_task(FUNC_MM, p_dscr);
        const Tensor &dsc_raw = dscr_outs.get_ref(0);

        // dsc = dsc_raw * tril  (mul): [C,C]
        L0TaskArgs p_dsc;
        p_dsc.add_input(dsc_raw);
        p_dsc.add_input(tril);
        p_dsc.add_output(dsc_ci);
        p_dsc.add_scalar(static_cast<uint64_t>(0));   // mul
        p_dsc.add_scalar(static_cast<uint64_t>(C));
        p_dsc.add_scalar(static_cast<uint64_t>(C));
        TaskOutputTensors dsc_outs = rt_submit_aiv_task(FUNC_ELT, p_dsc);
        const Tensor &dsc = dsc_outs.get_ref(0);

        // dQt_recon = dO @ H_n^T  (NT): [C,dv]@[dv,dk] = [C,dk], M=C,N=dk,Kc=dv
        L0TaskArgs p_dqr;
        p_dqr.add_input(dO_n);
        p_dqr.add_input(H_n);
        p_dqr.add_output(dqtr_ci);
        p_dqr.add_scalar(static_cast<uint64_t>(2));   // NT
        p_dqr.add_scalar(static_cast<uint64_t>(C));
        p_dqr.add_scalar(static_cast<uint64_t>(dk));
        p_dqr.add_scalar(static_cast<uint64_t>(dv));
        TaskOutputTensors dqr_outs = rt_submit_aic_task(FUNC_MM, p_dqr);
        const Tensor &dqt_recon = dqr_outs.get_ref(0);

        // dQt_intra = dsc @ Kin  (NN): [C,C]@[C,dk] = [C,dk], M=C,N=dk,Kc=C
        L0TaskArgs p_dqi;
        p_dqi.add_input(dsc);
        p_dqi.add_input(Kin);
        p_dqi.add_output(dqti_ci);
        p_dqi.add_scalar(static_cast<uint64_t>(0));   // NN
        p_dqi.add_scalar(static_cast<uint64_t>(C));
        p_dqi.add_scalar(static_cast<uint64_t>(dk));
        p_dqi.add_scalar(static_cast<uint64_t>(C));
        TaskOutputTensors dqi_outs = rt_submit_aic_task(FUNC_MM, p_dqi);
        const Tensor &dqt_intra = dqi_outs.get_ref(0);

        // dQt = dQt_recon + dQt_intra  (add): [C,dk]
        L0TaskArgs p_dqt;
        p_dqt.add_input(dqt_recon);
        p_dqt.add_input(dqt_intra);
        p_dqt.add_inout(dQt_n);
        p_dqt.add_scalar(static_cast<uint64_t>(1));   // add
        p_dqt.add_scalar(static_cast<uint64_t>(C));
        p_dqt.add_scalar(static_cast<uint64_t>(dk));
        rt_submit_aiv_task(FUNC_ELT, p_dqt);

        // dKin = dsc^T @ Qt  (TN): [C,C]@[C,dk] = [C,dk], M=C,N=dk,Kc=C
        L0TaskArgs p_dkin;
        p_dkin.add_input(dsc);
        p_dkin.add_input(Qt);
        p_dkin.add_inout(dKin_n);
        p_dkin.add_scalar(static_cast<uint64_t>(1));   // TN
        p_dkin.add_scalar(static_cast<uint64_t>(C));
        p_dkin.add_scalar(static_cast<uint64_t>(dk));
        p_dkin.add_scalar(static_cast<uint64_t>(C));
        rt_submit_aic_task(FUNC_MM, p_dkin);

        // dVi = scores^T @ dO  (TN): [C,C]@[C,dv] = [C,dv], M=C,N=dv,Kc=C
        L0TaskArgs p_dvi;
        p_dvi.add_input(scores);
        p_dvi.add_input(dO_n);
        p_dvi.add_inout(dVi_n);
        p_dvi.add_scalar(static_cast<uint64_t>(1));   // TN
        p_dvi.add_scalar(static_cast<uint64_t>(C));
        p_dvi.add_scalar(static_cast<uint64_t>(dv));
        p_dvi.add_scalar(static_cast<uint64_t>(C));
        rt_submit_aic_task(FUNC_MM, p_dvi);
    }
}

}  // extern "C"
