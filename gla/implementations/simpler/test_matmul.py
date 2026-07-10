#!/usr/bin/env python3
"""De-risk the generalised GLA matmul kernel (NN / TN / NT modes) on HW.

Each case runs a single square SxSxS fp32 matmul in one transpose mode (S from
the ``T`` param, dispatched over {16,32,64,128}) and compares against torch.
Establishes the transposed-matmul path (k_rest^T@v, q_eff@k_eff^T) shared by
chunk_h and chunk_o, and guards the runtime-size dispatch at S=32 and S=128.

Run (base env)::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd gla/implementations/simpler
    python test_matmul.py --platform a2a3 --device 6
"""

import torch
from simpler.task_interface import ArgDirection as Dr

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMatmul(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/matmul_test_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [Dr.IN, Dr.IN, Dr.OUT, Dr.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MM",
                "source": "kernels/aic/matmul_kernel.cpp",
                "core_type": "aic",
                "signature": [Dr.IN, Dr.IN, Dr.OUT],
            },
        ],
    }

    CASES = [
        {"name": "NN", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"mode": 0, "T": 128}},
        {"name": "TN", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"mode": 1, "T": 128}},
        {"name": "NT", "platforms": ["a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"mode": 2, "T": 128}},
        {"name": "NN32", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"mode": 0, "T": 32}},
        {"name": "TN32", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"mode": 1, "T": 32}},
        {"name": "NT32", "platforms": ["a2a3sim", "a2a3"],
         "config": {"aicpu_thread_num": 4, "block_dim": 24}, "params": {"mode": 2, "T": 32}},
    ]

    def generate_args(self, params):
        T = params["T"]
        torch.manual_seed(0)
        A = torch.randn(T, T, dtype=torch.float32) * 0.1
        B = torch.randn(T, T, dtype=torch.float32) * 0.1
        C = torch.zeros(T, T, dtype=torch.float32)
        config = torch.tensor([params["mode"], T], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()), Tensor("B", B.flatten()),
            Tensor("C", C.flatten()), Tensor("config", config),
        )

    def compute_golden(self, args, params):
        T = params["T"]
        A = args.A.reshape(T, T)
        B = args.B.reshape(T, T)
        C = args.C.reshape(T, T)
        mode = params["mode"]
        if mode == 1:      # TN: A^T @ B
            C[:] = A.t() @ B
        elif mode == 2:    # NT: A @ B^T
            C[:] = A @ B.t()
        else:              # NN
            C[:] = A @ B


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
