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
    ]

    def generate_args(self, params):
        C, Dd, N = params["C"], params["D"], params["N"]
        tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        g = torch.randn(N * C, Dd, dtype=torch.float32) * 0.05   # log-gate-ish magnitude
        g_cs = torch.zeros(N * C, Dd, dtype=torch.float32)
        config = torch.tensor([C, Dd, N], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("tril", tril.flatten()),
            Tensor("g", g.flatten()),
            Tensor("g_cs", g_cs.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        C, Dd, N = params["C"], params["D"], params["N"]
        tril = args.tril.reshape(C, C)
        g = args.g.reshape(N * C, Dd)
        g_cs = args.g_cs.reshape(N * C, Dd)
        for n in range(N):
            g_cs[n * C:(n + 1) * C] = tril @ g[n * C:(n + 1) * C]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
