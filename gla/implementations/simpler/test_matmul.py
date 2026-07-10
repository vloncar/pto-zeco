#!/usr/bin/env python3
"""De-risk the generalised GLA matmul kernel (NN / TN / NT modes) on HW.

Each case runs a single rectangular ``M x N x Kc`` fp32 matmul in one transpose
mode and compares against torch.  Dims (M, N, Kc) are runtime scalars, each
dispatched to a compile-time template over {16,32,64,128}.  Covers the square
path (regression from Phase 1) and the four rectangular GLA triples that appear
when C != D (here C=32, D=64):

    KV    (TN): (M,N,Kc) = (D,D,C) = (64,64,32)
    inter (NN): (M,N,Kc) = (C,D,D) = (32,64,64)
    Aqk   (NT): (M,N,Kc) = (C,C,D) = (32,32,64)
    intra (NN): (M,N,Kc) = (C,D,C) = (32,64,32)

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

    _CFG = {"aicpu_thread_num": 4, "block_dim": 24}
    _HW = ["a2a3"]
    _SIM_HW = ["a2a3sim", "a2a3"]

    CASES = [
        # square regression (Phase 1): (mode, M, N, Kc)
        {"name": "NN", "platforms": _HW, "config": _CFG, "params": {"mode": 0, "M": 128, "N": 128, "Kc": 128}},
        {"name": "TN", "platforms": _HW, "config": _CFG, "params": {"mode": 1, "M": 128, "N": 128, "Kc": 128}},
        {"name": "NT", "platforms": _HW, "config": _CFG, "params": {"mode": 2, "M": 128, "N": 128, "Kc": 128}},
        {"name": "NN32", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 0, "M": 32, "N": 32, "Kc": 32}},
        {"name": "TN32", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 1, "M": 32, "N": 32, "Kc": 32}},
        {"name": "NT32", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 2, "M": 32, "N": 32, "Kc": 32}},
        # rectangular GLA triples at C=32, D=64
        {"name": "KV_TN", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 1, "M": 64, "N": 64, "Kc": 32}},
        {"name": "inter_NN", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 0, "M": 32, "N": 64, "Kc": 64}},
        {"name": "Aqk_NT", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 2, "M": 32, "N": 32, "Kc": 64}},
        {"name": "intra_NN", "platforms": _SIM_HW, "config": _CFG, "params": {"mode": 0, "M": 32, "N": 64, "Kc": 32}},
    ]

    def generate_args(self, params):
        mode, M, N, Kc = params["mode"], params["M"], params["N"], params["Kc"]
        torch.manual_seed(0)
        # Physical (row-major) GM shapes: A is [Kc,M] for TN else [M,Kc];
        # B is [N,Kc] for NT else [Kc,N]; C is [M,N].
        a_shape = (Kc, M) if mode == 1 else (M, Kc)
        b_shape = (N, Kc) if mode == 2 else (Kc, N)
        A = torch.randn(*a_shape, dtype=torch.float32) * 0.1
        B = torch.randn(*b_shape, dtype=torch.float32) * 0.1
        C = torch.zeros(M, N, dtype=torch.float32)
        config = torch.tensor([mode, M, N, Kc], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()), Tensor("B", B.flatten()),
            Tensor("C", C.flatten()), Tensor("config", config),
        )

    def compute_golden(self, args, params):
        mode, M, N, Kc = params["mode"], params["M"], params["N"], params["Kc"]
        a_shape = (Kc, M) if mode == 1 else (M, Kc)
        b_shape = (N, Kc) if mode == 2 else (Kc, N)
        A = args.A.reshape(*a_shape)
        B = args.B.reshape(*b_shape)
        C = args.C.reshape(M, N)
        if mode == 1:      # TN: A^T @ B   (A is [Kc,M])
            C[:] = A.t() @ B
        elif mode == 2:    # NT: A @ B^T   (B is [N,Kc])
            C[:] = A @ B.t()
        else:              # NN: A @ B
            C[:] = A @ B


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
