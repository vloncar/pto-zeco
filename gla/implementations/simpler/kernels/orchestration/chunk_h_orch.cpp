/*
 * GLA chunk_h orchestration (simpler runtime).
 *
 * Runs the recurrent hidden-state scan over N chunks, carrying the [K,V] state
 * S from S = 0 and emitting the per-chunk snapshot (state entering each chunk):
 *
 *   for n in 0..N-1:
 *     k_rest, decay = prep(g_cs[n], g_total[n], k[n])     (aiv)
 *     KV            = k_rest^T @ v[n]                      (aic, mode TN)
 *     s_snap[n], S  = update(KV, decay, S, is_first=n==0)  (aiv)
 *
 * S is allocated once and threaded INOUT so the runtime serialises the
 * recurrence; s_snap[n] is written into the (external) OUT tensor.
 *
 * Args: [0]=k [N*C,dk] IN, [1]=v [N*C,dv] IN, [2]=g_cs [N*C,dk] IN,
 *       [3]=s_snap [N,dk,dv] OUT, [4]=config int64[4]={C,dk,dv,N} IN.
 */

#include <stddef.h>
#include <stdint.h>
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MM 0
#define FUNC_PREP 1
#define FUNC_UPDATE 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &k = orch_args.tensor(0).ref();
    const Tensor &v = orch_args.tensor(1).ref();
    const Tensor &gcs = orch_args.tensor(2).ref();
    const Tensor &ssnap = orch_args.tensor(3).ref();

    int64_t *cfg = orch_args.tensor(4).ref().data_as<int64_t>();
    int C = static_cast<int>(cfg[0]);
    int dk = static_cast<int>(cfg[1]);
    int dv = static_cast<int>(cfg[2]);
    int N = static_cast<int>(cfg[3]);

    // Flat (1D) views: the kernels impose their own 2D shape on the buffer, so a
    // view only needs the correct flat element offset (the proven M-A pattern; 2D
    // offsets are not applied on a 1D base).  k/g_cs are [C,dk], v is [C,dv], and
    // the state (S/KV/snap) is [dk,dv] — three widths when dk != dv.
    uint32_t cdk_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(dk)};
    uint32_t cdv_flat[1] = {static_cast<uint32_t>(C) * static_cast<uint32_t>(dv)};
    uint32_t kv_flat[1] = {static_cast<uint32_t>(dk) * static_cast<uint32_t>(dv)};
    uint32_t row_flat[1] = {static_cast<uint32_t>(dk)};

    uint32_t cdk_shape[2] = {static_cast<uint32_t>(C), static_cast<uint32_t>(dk)};
    uint32_t kv_shape[2] = {static_cast<uint32_t>(dk), static_cast<uint32_t>(dv)};
    uint32_t dec_shape[2] = {static_cast<uint32_t>(dk), 1};

    TensorCreateInfo krest_ci(cdk_shape, 2, DataType::FLOAT32);
    TensorCreateInfo decay_ci(dec_shape, 2, DataType::FLOAT32);
    TensorCreateInfo kv_ci(kv_shape, 2, DataType::FLOAT32);

    // Persistent state S [dk,dv], carried across chunks.  It must be allocated
    // in a scope that ENCLOSES the whole chunk loop (each iteration opens its
    // own inner scope for the per-chunk temporaries) — same lifetime pattern as
    // the carried accumulator in the paged_attention example.
    PTO2_SCOPE() {
    TaskOutputTensors s_alloc = alloc_tensors(kv_ci);
    const Tensor &S = s_alloc.get_ref(0);

    for (int n = 0; n < N; n++) {
        PTO2_SCOPE_GUARD();
        uint32_t nu = static_cast<uint32_t>(n);
        uint32_t cdk = static_cast<uint32_t>(C) * dk;
        uint32_t cdv = static_cast<uint32_t>(C) * dv;
        uint32_t kv = static_cast<uint32_t>(dk) * dv;
        uint32_t dk_off[1] = {nu * cdk};
        uint32_t dv_off[1] = {nu * cdv};
        uint32_t gtot_off[1] = {nu * cdk + (static_cast<uint32_t>(C) - 1) * dk};
        uint32_t snap_off[1] = {nu * kv};

        Tensor k_n = k.view(cdk_flat, dk_off);
        Tensor v_n = v.view(cdv_flat, dv_off);
        Tensor gcs_n = gcs.view(cdk_flat, dk_off);
        Tensor gtot_n = gcs.view(row_flat, gtot_off);
        Tensor snap_n = ssnap.view(kv_flat, snap_off);

        // prep -> k_rest [C,dk], decay [dk,1]
        L0TaskArgs p_prep;
        p_prep.add_input(gcs_n);
        p_prep.add_input(gtot_n);
        p_prep.add_input(k_n);
        p_prep.add_output(krest_ci);
        p_prep.add_output(decay_ci);
        p_prep.add_scalar(static_cast<uint64_t>(C));   // rows (chunk size)
        p_prep.add_scalar(static_cast<uint64_t>(dk));  // cols (key dim)
        TaskOutputTensors prep_outs = rt_submit_aiv_task(FUNC_PREP, p_prep);
        const Tensor &krest = prep_outs.get_ref(0);
        const Tensor &decay = prep_outs.get_ref(1);

        // KV = k_rest^T @ v  (mode 1 = TN): [dk,C]@[C,dv] = [dk,dv], M=dk, N=dv, Kc=C
        L0TaskArgs p_mm;
        p_mm.add_input(krest);
        p_mm.add_input(v_n);
        p_mm.add_output(kv_ci);
        p_mm.add_scalar(static_cast<uint64_t>(1));     // mode TN
        p_mm.add_scalar(static_cast<uint64_t>(dk));    // M
        p_mm.add_scalar(static_cast<uint64_t>(dv));    // N
        p_mm.add_scalar(static_cast<uint64_t>(C));     // Kc
        TaskOutputTensors mm_outs = rt_submit_aic_task(FUNC_MM, p_mm);
        const Tensor &KV = mm_outs.get_ref(0);

        // update: s_snap[n] = S ; S = decay*S + KV   (state [dk,dv])
        L0TaskArgs p_up;
        p_up.add_input(KV);
        p_up.add_input(decay);
        p_up.add_inout(S);
        p_up.add_inout(snap_n);
        p_up.add_scalar(static_cast<uint64_t>(n == 0 ? 1 : 0));  // is_first
        p_up.add_scalar(static_cast<uint64_t>(dk));              // state rows (key dim)
        p_up.add_scalar(static_cast<uint64_t>(dv));              // state cols (value dim)
        rt_submit_aiv_task(FUNC_UPDATE, p_up);
    }
    }  // PTO2_SCOPE
}

}  // extern "C"
