#!/usr/bin/env python3
"""Milestone B3.1: validate the GLA grad_o simpler kernels (output-stage backward).

grad_o is the backward of chunk_o.  Per chunk it recomputes Qt=q*exp(g_cs),
Kin=k*exp(-g_cs), scores=(Qt@Kin^T)*tril, then runs the seven backward matmuls
producing the RAW adjoints (before the host gate scaling):

    dH   = Qt^T @ dO              [dk,dv]
    dQt  = dO @ H^T + dsc @ Kin   [C,dk]     dsc = (dO@v^T)*tril
    dKin = dsc^T @ Qt             [C,dk]
    dVi  = scores^T @ dO          [C,dv]     (intra dV)

Checked against a torch golden.  The gate elementwise (dq=dQt*exp(g_cs) etc.) is
done on host in SimplerZeCo.backward, not here.

Run (base env — simpler runtime)::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    cd gla/implementations/simpler
    python test_grad_o.py --platform a2a3 --device 6
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
class TestGradO(SceneTestCase):
    RTOL = 2e-3
    ATOL = 2e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/grad_o_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.IN, Dr.IN,
                          Dr.OUT, Dr.OUT, Dr.OUT, Dr.OUT, Dr.IN],
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
        torch.manual_seed(3)
        q = torch.randn(N * C, dk, dtype=torch.float32) * 0.1
        k = torch.randn(N * C, dk, dtype=torch.float32) * 0.1
        v = torch.randn(N * C, dv, dtype=torch.float32) * 0.1
        g = -torch.rand(N * C, dk, dtype=torch.float32) * 0.05
        g_cs = _gcs(g, N, C, dk)
        s_snap = torch.randn(N, dk, dv, dtype=torch.float32) * 0.1
        dO = torch.randn(N * C, dv, dtype=torch.float32) * 0.1
        tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        dQt = torch.zeros(N * C, dk, dtype=torch.float32)
        dKin = torch.zeros(N * C, dk, dtype=torch.float32)
        dVi = torch.zeros(N * C, dv, dtype=torch.float32)
        dH = torch.zeros(N, dk, dv, dtype=torch.float32)
        config = torch.tensor([C, dk, dv, N], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("q", q.flatten()), Tensor("k", k.flatten()), Tensor("v", v.flatten()),
            Tensor("g_cs", g_cs.flatten()), Tensor("s_snap", s_snap.flatten()),
            Tensor("dO", dO.flatten()), Tensor("tril", tril.flatten()),
            Tensor("dQt", dQt.flatten()), Tensor("dKin", dKin.flatten()),
            Tensor("dVi", dVi.flatten()), Tensor("dH", dH.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        C, N = params["C"], params["N"]
        dk = params.get("dk", params.get("D"))
        dv = params.get("dv", params.get("D"))
        q = args.q.reshape(N * C, dk)
        k = args.k.reshape(N * C, dk)
        v = args.v.reshape(N * C, dv)
        g_cs = args.g_cs.reshape(N * C, dk)
        s_snap = args.s_snap.reshape(N, dk, dv)
        dO = args.dO.reshape(N * C, dv)
        tril = args.tril.reshape(C, C)
        dQt = args.dQt.reshape(N * C, dk)
        dKin = args.dKin.reshape(N * C, dk)
        dVi = args.dVi.reshape(N * C, dv)
        dH = args.dH.reshape(N, dk, dv)
        for n in range(N):
            sl = slice(n * C, (n + 1) * C)
            gcs_n = g_cs[sl]
            Qt = q[sl] * torch.exp(gcs_n)
            Kin = k[sl] * torch.exp(-gcs_n)
            scores = (Qt @ Kin.t()) * tril
            dO_n = dO[sl]
            H_n = s_snap[n]
            dH[n] = Qt.t() @ dO_n
            dsc = (dO_n @ v[sl].t()) * tril
            dQt[sl] = dO_n @ H_n.t() + dsc @ Kin
            dKin[sl] = dsc.t() @ Qt
            dVi[sl] = scores.t() @ dO_n


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
