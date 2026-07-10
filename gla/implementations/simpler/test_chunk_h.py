#!/usr/bin/env python3
"""Milestone B: validate the GLA chunk_h simpler kernels (recurrent state scan).

Runs prep (aiv) -> KV = k_rest^T@v (aic, TN) -> update (aiv) over N chunks,
carrying the [K,V] state S from 0, and checks the per-chunk snapshots
``s_snap[n]`` (state entering chunk n) against a torch golden.

Run (base env — simpler runtime, NOT torch-npu)::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd gla/implementations/simpler
    python test_chunk_h.py --platform a2a3 --device 6
"""

import torch
from simpler.task_interface import ArgDirection as Dr

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


def _gcs(g, N, C, D):
    """Within-chunk inclusive cumsum of log-gates -> [N*C, D]."""
    out = torch.zeros_like(g)
    for n in range(N):
        out[n * C:(n + 1) * C] = torch.cumsum(g[n * C:(n + 1) * C], dim=0)
    return out


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestChunkH(SceneTestCase):
    RTOL = 2e-3
    ATOL = 2e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/chunk_h_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.OUT, Dr.IN],
        },
        "incores": [
            {"func_id": 0, "name": "MM", "source": "kernels/aic/matmul_kernel.cpp",
             "core_type": "aic", "signature": [Dr.IN, Dr.IN, Dr.OUT]},
            {"func_id": 1, "name": "PREP", "source": "kernels/aiv/chunk_h_prep.cpp",
             "core_type": "aiv", "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.OUT, Dr.OUT]},
            {"func_id": 2, "name": "UPDATE", "source": "kernels/aiv/chunk_h_update.cpp",
             "core_type": "aiv", "signature": [Dr.IN, Dr.IN, Dr.INOUT, Dr.INOUT]},
        ],
    }

    CASES = [
        {"name": "C128_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 128, "D": 128, "N": 2}},
        {"name": "C128_N4", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 128, "D": 128, "N": 4}},
        {"name": "C32_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 32, "N": 2}},
        {"name": "C32_N4", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 32, "N": 4}},
    ]

    def generate_args(self, params):
        C, D, N = params["C"], params["D"], params["N"]
        torch.manual_seed(0)
        k = torch.randn(N * C, D, dtype=torch.float32) * 0.1
        v = torch.randn(N * C, D, dtype=torch.float32) * 0.1
        g = -torch.rand(N * C, D, dtype=torch.float32) * 0.05   # negative log-gates
        g_cs = _gcs(g, N, C, D)
        s_snap = torch.zeros(N, D, D, dtype=torch.float32)
        config = torch.tensor([C, D, N], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("k", k.flatten()), Tensor("v", v.flatten()),
            Tensor("g_cs", g_cs.flatten()), Tensor("s_snap", s_snap.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        C, D, N = params["C"], params["D"], params["N"]
        k = args.k.reshape(N * C, D)
        v = args.v.reshape(N * C, D)
        g_cs = args.g_cs.reshape(N * C, D)
        s_snap = args.s_snap.reshape(N, D, D)
        S = torch.zeros(D, D, dtype=torch.float32)
        for n in range(N):
            s_snap[n] = S
            gcs_n = g_cs[n * C:(n + 1) * C]
            g_total = gcs_n[-1]
            coeff = torch.exp(g_total.unsqueeze(0) - gcs_n)
            k_rest = k[n * C:(n + 1) * C] * coeff
            KV = k_rest.t() @ v[n * C:(n + 1) * C]
            S = torch.exp(g_total).unsqueeze(1) * S + KV


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
