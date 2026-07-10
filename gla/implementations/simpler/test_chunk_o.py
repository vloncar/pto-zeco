#!/usr/bin/env python3
"""Milestone C: validate the GLA chunk_o simpler kernels (output stage).

Per chunk: prep (q_eff,k_eff) -> inter=q_eff@S (NN) -> Aqk=q_eff@k_eff^T (NT)
-> mask Aqk*tril -> intra=Aqk_m@v (NN) -> o=inter+intra.  Checked against a
torch golden with a given (already S_recv-folded) snapshot S_n.

Run (base env — simpler runtime)::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd gla/implementations/simpler
    python test_chunk_o.py --platform a2a3 --device 6
"""

import torch
from simpler.task_interface import ArgDirection as Dr

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


def _gcs(g, N, C, D):
    out = torch.zeros_like(g)
    for n in range(N):
        out[n * C:(n + 1) * C] = torch.cumsum(g[n * C:(n + 1) * C], dim=0)
    return out


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestChunkO(SceneTestCase):
    RTOL = 2e-3
    ATOL = 2e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/chunk_o_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.OUT, Dr.IN],
        },
        "incores": [
            {"func_id": 0, "name": "MM", "source": "kernels/aic/matmul_kernel.cpp",
             "core_type": "aic", "signature": [Dr.IN, Dr.IN, Dr.OUT]},
            {"func_id": 1, "name": "PREP", "source": "kernels/aiv/chunk_o_prep.cpp",
             "core_type": "aiv", "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.OUT, Dr.OUT]},
            {"func_id": 2, "name": "ELT", "source": "kernels/aiv/chunk_o_elt.cpp",
             "core_type": "aiv", "signature": [Dr.IN, Dr.IN, Dr.OUT]},
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
        {"name": "C32_N1", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 32, "N": 1}},
        {"name": "C64_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 64, "D": 64, "N": 2}},
        # rectangular C != D (Phase 2)
        {"name": "C32_D64_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 64, "N": 2}},
        {"name": "C32_D64_N4", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 64, "N": 4}},
    ]

    def generate_args(self, params):
        C, D, N = params["C"], params["D"], params["N"]
        torch.manual_seed(1)
        q = torch.randn(N * C, D, dtype=torch.float32) * 0.1
        k = torch.randn(N * C, D, dtype=torch.float32) * 0.1
        v = torch.randn(N * C, D, dtype=torch.float32) * 0.1
        g = -torch.rand(N * C, D, dtype=torch.float32) * 0.05
        g_cs = _gcs(g, N, C, D)
        s_snap = torch.randn(N, D, D, dtype=torch.float32) * 0.1
        tril = torch.tril(torch.ones(C, C, dtype=torch.float32))  # inclusive lower
        o = torch.zeros(N * C, D, dtype=torch.float32)
        config = torch.tensor([C, D, N], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("q", q.flatten()), Tensor("k", k.flatten()), Tensor("v", v.flatten()),
            Tensor("g_cs", g_cs.flatten()), Tensor("s_snap", s_snap.flatten()),
            Tensor("tril", tril.flatten()), Tensor("o", o.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        C, D, N = params["C"], params["D"], params["N"]
        q = args.q.reshape(N * C, D)
        k = args.k.reshape(N * C, D)
        v = args.v.reshape(N * C, D)
        g_cs = args.g_cs.reshape(N * C, D)
        s_snap = args.s_snap.reshape(N, D, D)
        tril = args.tril.reshape(C, C)
        o = args.o.reshape(N * C, D)
        for n in range(N):
            sl = slice(n * C, (n + 1) * C)
            gcs_n = g_cs[sl]
            q_eff = q[sl] * torch.exp(gcs_n)
            k_eff = k[sl] * torch.exp(-gcs_n)
            inter = q_eff @ s_snap[n]
            aqk = (q_eff @ k_eff.t()) * tril
            intra = aqk @ v[sl]
            o[sl] = inter + intra


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
