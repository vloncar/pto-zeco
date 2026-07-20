#!/usr/bin/env python3
"""Milestone A: validate the GLA gate_cumsum simpler kernel (Cube tril-matmul).

Establishes the GLA-in-simpler build/launch pipeline: a Cube incore kernel +
orchestration compiled and dispatched by the SceneTestCase harness, checked
against a torch golden (within-chunk cumulative sum via ``tril @ g``).

Run (base env — simpler runtime, NOT torch-npu)::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd gla/implementations/simpler
    python test_gate_cumsum.py --platform a2a3 --device 6
"""

import torch
from simpler.task_interface import ArgDirection as Dr

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestGateCumsum(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/gate_cumsum_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [Dr.IN, Dr.IN, Dr.OUT, Dr.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "CUMSUM",
                "source": "kernels/aic/gate_cumsum_kernel.cpp",
                "core_type": "aic",
                "signature": [Dr.IN, Dr.IN, Dr.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "C128_N2",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"C": 128, "D": 128, "N": 2},
        },
        {
            "name": "C32_N2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"C": 32, "D": 32, "N": 2},
        },
        {   # rectangular C != D (Phase 2): tril[C,C] @ g[C,dk]
            "name": "C32_D64_N2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"C": 32, "D": 64, "N": 2},
        },
        {   # dk != dv (F7): gate_cumsum uses only dk, but the config carries dv (and
            # N at cfg[3]) — this guards that plumbing.
            "name": "dk32_dv64_N2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"C": 32, "dk": 32, "dv": 64, "N": 2},
        },
    ]

    def generate_args(self, params):
        C, N = params["C"], params["N"]
        dk = params.get("dk", params.get("D"))
        dv = params.get("dv", params.get("D"))
        tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        g = torch.randn(N * C, dk, dtype=torch.float32) * 0.05   # log-gate-ish magnitude
        g_cs = torch.zeros(N * C, dk, dtype=torch.float32)
        config = torch.tensor([C, dk, dv, N], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("tril", tril.flatten()),
            Tensor("g", g.flatten()),
            Tensor("g_cs", g_cs.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        C, N = params["C"], params["N"]
        dk = params.get("dk", params.get("D"))
        tril = args.tril.reshape(C, C)
        g = args.g.reshape(N * C, dk)
        g_cs = args.g_cs.reshape(N * C, dk)
        for n in range(N):
            g_cs[n * C:(n + 1) * C] = tril @ g[n * C:(n + 1) * C]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
