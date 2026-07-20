#!/usr/bin/env python3
"""Milestone B3.2: validate the GLA grad_h simpler kernels (state-stage backward).

grad_h is the backward of chunk_h.  The cross-chunk reverse recurrence that
produces dSloc[n] is host glue; this kernel takes dSloc[n] per chunk and runs the
two state matmuls (recomputing Kstate=k*exp(g_total-g_cs) via chunk_h_prep):

    dKstate = v @ dSloc^T      [C,dk]
    dVs     = Kstate @ dSloc   [C,dv]     (state dV)

Checked against a torch golden.  The host finishes them (dk_state =
dKstate*exp(g_total-g_cs), dg_cs_state, dV += dVs).

Run (base env — simpler runtime)::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd gla/implementations/simpler
    python test_grad_h.py --platform a2a3 --device 6
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
class TestGradH(SceneTestCase):
    RTOL = 2e-3
    ATOL = 2e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/grad_h_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.OUT, Dr.OUT, Dr.IN],
        },
        "incores": [
            {"func_id": 0, "name": "MM", "source": "kernels/aic/matmul_kernel.cpp",
             "core_type": "aic", "signature": [Dr.IN, Dr.IN, Dr.OUT]},
            {"func_id": 1, "name": "PREP", "source": "kernels/aiv/chunk_h_prep.cpp",
             "core_type": "aiv", "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.OUT, Dr.OUT]},
        ],
    }

    CASES = [
        {"name": "C128_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 128, "D": 128, "N": 2}},
        {"name": "C32_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 32, "N": 2}},
        {"name": "C32_N4", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 32, "N": 4}},
        {"name": "C64_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 64, "D": 64, "N": 2}},
        # rectangular C != D
        {"name": "C32_D64_N2", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "D": 64, "N": 2}},
        # dk != dv
        {"name": "dk32_dv64_N4", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 32, "dk": 32, "dv": 64, "N": 4}},
        {"name": "dk64_dv32_N2", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"C": 64, "dk": 64, "dv": 32, "N": 2}},
    ]

    def generate_args(self, params):
        C, N = params["C"], params["N"]
        dk = params.get("dk", params.get("D"))
        dv = params.get("dv", params.get("D"))
        torch.manual_seed(5)
        k = torch.randn(N * C, dk, dtype=torch.float32) * 0.1
        v = torch.randn(N * C, dv, dtype=torch.float32) * 0.1
        g = -torch.rand(N * C, dk, dtype=torch.float32) * 0.05
        g_cs = _gcs(g, N, C, dk)
        dSloc = torch.randn(N, dk, dv, dtype=torch.float32) * 0.1
        dKstate = torch.zeros(N * C, dk, dtype=torch.float32)
        dVs = torch.zeros(N * C, dv, dtype=torch.float32)
        config = torch.tensor([C, dk, dv, N], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("k", k.flatten()), Tensor("v", v.flatten()),
            Tensor("g_cs", g_cs.flatten()), Tensor("dSloc", dSloc.flatten()),
            Tensor("dKstate", dKstate.flatten()), Tensor("dVs", dVs.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        C, N = params["C"], params["N"]
        dk = params.get("dk", params.get("D"))
        dv = params.get("dv", params.get("D"))
        k = args.k.reshape(N * C, dk)
        v = args.v.reshape(N * C, dv)
        g_cs = args.g_cs.reshape(N * C, dk)
        dSloc = args.dSloc.reshape(N, dk, dv)
        dKstate = args.dKstate.reshape(N * C, dk)
        dVs = args.dVs.reshape(N * C, dv)
        for n in range(N):
            sl = slice(n * C, (n + 1) * C)
            gcs_n = g_cs[sl]
            g_total = gcs_n[-1]
            Kstate = k[sl] * torch.exp(g_total - gcs_n)
            dSloc_n = dSloc[n]
            dKstate[sl] = v[sl] @ dSloc_n.t()
            dVs[sl] = Kstate @ dSloc_n


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
