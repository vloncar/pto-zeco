#!/usr/bin/env python3
"""Simpler-runtime ZeCO / GLA backend — hand-written PTO-ISA kernels + real AllScan.

The GLA compute runs entirely in the **simpler runtime** (base env: torch 2.6 +
pypto/simpler, NO torch-npu) via the hand-written orchestrated kernels in
``kernels/`` (validated by ``test_gate_cumsum/chunk_h/chunk_o.py``):

    stage1 = gate_cumsum -> chunk_h(w=0, u=v)   ->  s_snap, g_cs   (per rank, 1 device)
    S_total (local end state)                    ->  host (linearity, exact)
    AllScan  out[p] = S_total[p] + gamma[p] . out[p-1]  (REAL multi-device HCCL)
    S_recv[p] = out[p-1] ; fold into snapshots    ->  host (shift_snaps)
    stage2 = chunk_o(v_corr = v)                 ->  O           (per rank, 1 device)

The cross-device boundary is the existing **simpler AllScan** (``allscan/
implementations/simpler``) — a genuine device-to-device HCCL collective — so the
whole ZeCO is one runtime, no torch-npu/pypto coexistence. The compute runs as
hand-written PTO-ISA kernels in the simpler runtime (not torch-npu-launched
``.so`` kernels).

Requires ``dk == dv == D``, ``L % C == 0``, and — currently — ``C == D`` (square
tiles). The incore kernels dispatch a runtime tile size to a compile-time template
over ``{16, 32, 64, 128}`` (the ``benchmark_bgemm`` pattern), so the whole pipeline
runs at any of those sizes as long as chunk == head dim. When ``C == D`` every GLA
matmul is square (``M == N == Kc``), which is what lets a single size scalar drive
all of gate_cumsum / chunk_h / chunk_o. Non-square ``C != D`` (e.g. the bench's
``D == 64`` configs) needs the matmul kernel generalised to independent ``M, N, Kc``
— a follow-up (F3 Phase 2).
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gla.common import ZeCoImpl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = "tensormap_and_ringbuffer"
from simpler.task_interface import ArgDirection as _D  # noqa: E402

# ---------------------------------------------------------------------------
# CALLABLE specs (mirror the test_*.py SceneTestCase declarations)
# ---------------------------------------------------------------------------

def _spec(orch_src, orch_sig, incores):
    return {
        "orchestration": {"source": os.path.join(HERE, orch_src),
                          "function_name": "aicpu_orchestration_entry", "signature": orch_sig},
        "incores": [dict(func_id=fid, name=nm, source=os.path.join(HERE, src),
                         core_type=ct, signature=sig) for (fid, nm, src, ct, sig) in incores],
    }


GATE_CUMSUM_SPEC = _spec(
    "kernels/orchestration/gate_cumsum_orch.cpp",
    [_D.IN, _D.IN, _D.OUT, _D.IN],
    [(0, "CUMSUM", "kernels/aic/gate_cumsum_kernel.cpp", "aic", [_D.IN, _D.IN, _D.OUT])],
)

CHUNK_H_SPEC = _spec(
    "kernels/orchestration/chunk_h_orch.cpp",
    [_D.IN, _D.IN, _D.IN, _D.OUT, _D.IN],
    [(0, "MM", "kernels/aic/matmul_kernel.cpp", "aic", [_D.IN, _D.IN, _D.OUT]),
     (1, "PREP", "kernels/aiv/chunk_h_prep.cpp", "aiv", [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]),
     (2, "UPDATE", "kernels/aiv/chunk_h_update.cpp", "aiv", [_D.IN, _D.IN, _D.INOUT, _D.INOUT])],
)

CHUNK_O_SPEC = _spec(
    "kernels/orchestration/chunk_o_orch.cpp",
    [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.IN],
    [(0, "MM", "kernels/aic/matmul_kernel.cpp", "aic", [_D.IN, _D.IN, _D.OUT]),
     (1, "PREP", "kernels/aiv/chunk_o_prep.cpp", "aiv", [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]),
     (2, "ELT", "kernels/aiv/chunk_o_elt.cpp", "aiv", [_D.IN, _D.IN, _D.OUT])],
)


# ---------------------------------------------------------------------------
# Host-side pieces (pure torch; the ZeCO linearity glue — S_total advance,
# shift_snaps fold, gamma = A.prod — kept here so the whole backend stays in the
# base env with no torch-npu dependency)
# ---------------------------------------------------------------------------

def _S_total(s_snap, g_cs, k, v, L, C, D):
    """Advance the last chunk snapshot through the last chunk -> local end state [D,D]."""
    n_chunks = L // C
    off = (n_chunks - 1) * C
    g_cs_last = g_cs[off:off + C]
    g_total = g_cs_last[-1]
    k_last = k[off:off + C]
    v_last = v[off:off + C]
    k_rest = k_last * torch.exp(g_total.unsqueeze(0) - g_cs_last)
    return torch.exp(g_total).unsqueeze(1) * s_snap[-1] + k_rest.t() @ v_last


def _shift_snaps(s_snap, A_rank, S_recv, L, C, D):
    """Fold the received boundary state into the chunk snapshots (host, fp32)."""
    n_chunks = L // C
    A_ch = A_rank.reshape(n_chunks, C, D).prod(dim=1)          # [n_chunks, D]
    c = torch.ones(n_chunks, D)
    if n_chunks > 1:
        c[1:] = torch.cumprod(A_ch, dim=0)[:-1]
    return s_snap + c.unsqueeze(-1) * S_recv.unsqueeze(0)


# ---------------------------------------------------------------------------
# Single-device compute runner (reuses the SceneTestCase L2 harness internals)
# ---------------------------------------------------------------------------

_SPECS = {"gate_cumsum": GATE_CUMSUM_SPEC, "chunk_h": CHUNK_H_SPEC, "chunk_o": CHUNK_O_SPEC}


class _ComputeRunner:
    """Runs the GLA compute orchestrations on one device via the simpler L3 Worker.

    Device/worker constraints (established empirically on this runtime):
      * a device hosts one worker at a time (device-exclusive), and
      * an L3 chip child binds to the first callable it runs — a *different*
        callable on the same worker fails to stage.
    So each kernel invocation stands up a fresh single-callable L3 worker
    (``device_ids=[one]``, ``num_sub_workers=0``), submits one chip task via an
    ``orch_fn`` (the AllScan launch pattern), and tears it down. The kernel
    *compile* is session-cached, so only the (cheap) worker init/close repeats.
    Tensors are staged through shared memory (the chip child is a subprocess).
    """

    def __init__(self, device_id, platform):
        self.device_id = device_id
        self.platform = platform

    def run(self, name, sig, named_tensors):
        """Run orchestration ``name``; ``named_tensors`` = (label, torch tensor) in
        orchestration-signature order. OUT/INOUT tensors are written in place."""
        from simpler.task_interface import (ArgDirection, CallConfig, TaskArgs,
                                            TensorArgType)
        from simpler.worker import Worker
        from simpler_setup.scene_test import _compile_chip_callable_from_spec
        from simpler_setup.torch_interop import make_tensor_arg

        D = ArgDirection
        _tag = {D.IN: TensorArgType.INPUT, D.OUT: TensorArgType.OUTPUT_EXISTING,
                D.INOUT: TensorArgType.INOUT}

        staged = []
        for i, (lbl, t) in enumerate(named_tensors):
            dt = torch.int64 if t.dtype == torch.int64 else torch.float32
            shm = torch.zeros(t.numel(), dtype=dt).share_memory_()
            shm.copy_(t.reshape(-1).to(dt))
            staged.append((t, shm, sig[i]))

        key = f"{name}:{self.platform}:{self.device_id}"
        cc = _compile_chip_callable_from_spec(_SPECS[name], self.platform, RUNTIME, key)
        worker = Worker(level=3, device_ids=[self.device_id], num_sub_workers=0,
                        platform=self.platform, runtime=RUNTIME)
        cid = worker.register(cc)
        worker.init()
        try:
            def orch_fn(orch, _args, cfg):
                chip_args = TaskArgs()
                for (_t, shm, d) in staged:
                    chip_args.add_tensor(make_tensor_arg(shm), _tag[d])
                orch.submit_next_level(cid, chip_args, cfg, worker=0)

            worker.run(orch_fn, args=None, config=CallConfig())
        finally:
            worker.close()

        for (t, shm, d) in staged:
            if d in (D.OUT, D.INOUT):
                t.reshape(-1).copy_(shm.to(t.dtype))

    def close(self):
        pass


class SimplerZeCo(ZeCoImpl):
    """Simpler-runtime GLA: gate_cumsum -> chunk_h(w=0) -> chunk_o(v=v); real AllScan boundary."""

    name = "simpler"

    def build(self, P, L, C, dk, dv, device_ids, platform):
        assert dk == dv, f"simpler GLA kernels assume K == V (D); got dk={dk} dv={dv}"
        assert L % C == 0, f"L={L} not divisible by C={C}"
        # The incore kernels dispatch one tile size over {16,32,64,128}; that only
        # covers the GLA pipeline when every matmul is square, i.e. chunk == head dim.
        # Non-square (C != D) needs the matmul kernel generalised to M,N,Kc (F3 P2).
        assert C == dk, (
            f"simpler GLA kernels currently require C == D (square tiles); got C={C} D={dk}. "
            f"C != D (non-square) is not yet supported.")
        assert C in (16, 32, 64, 128), (
            f"simpler GLA tile size must be one of {{16,32,64,128}}; got C={C}")
        self.P, self.L, self.C, self.D = P, L, C, dk
        self.device_ids = list(device_ids[:P])
        self.platform = platform
        self.N = L // C
        self._tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        self._config = torch.tensor([C, dk, self.N], dtype=torch.int64)
        # Fully distributed: rank p's GLA shard is computed on its OWN device
        # (device_ids[p]) and the boundary state is exchanged over the real
        # multi-device AllScan collective. One compute runner per rank/device.
        self._runners = [_ComputeRunner(d, platform) for d in self.device_ids]

    def _stage1(self, p, Q, K, V, A):
        """Run gate_cumsum + chunk_h on rank p's device -> (s_snap, g_cs, S_total)."""
        L, C, D, N = self.L, self.C, self.D, self.N
        r = self._runners[p]
        g_log = torch.log(A).contiguous()

        g_cs = torch.zeros(L, D, dtype=torch.float32)
        r.run("gate_cumsum", GATE_CUMSUM_SPEC["orchestration"]["signature"],
              [("tril", self._tril), ("g", g_log), ("g_cs", g_cs), ("config", self._config)])

        s_snap = torch.zeros(N, D, D, dtype=torch.float32)
        r.run("chunk_h", CHUNK_H_SPEC["orchestration"]["signature"],
              [("k", K.contiguous()), ("v", V.contiguous()), ("g_cs", g_cs),
               ("s_snap", s_snap), ("config", self._config)])

        S_total = _S_total(s_snap, g_cs, K, V, L, C, D)
        return s_snap, g_cs, S_total

    def _stage2(self, p, Q, K, V, g_cs, s_shift):
        """Run chunk_o on rank p's device -> O [L,D]."""
        L, C, D = self.L, self.C, self.D
        o = torch.zeros(L, D, dtype=torch.float32)
        self._runners[p].run(
            "chunk_o", CHUNK_O_SPEC["orchestration"]["signature"],
            [("q", Q.contiguous()), ("k", K.contiguous()), ("v", V.contiguous()),
             ("g_cs", g_cs), ("s_snap", s_shift.contiguous()), ("tril", self._tril),
             ("o", o), ("config", self._config)])
        return o

    def _boundary(self, S_totals, A):
        """AllScan prefix: returns out [P,D,D] (out[p] = S_total[p] + gamma[p]*out[p-1]).

        The AllScan worker is built + run + closed here so it never holds the
        devices while the per-rank compute workers (also created/closed per call)
        need them — a device hosts one worker at a time.
        """
        from allscan.implementations.simpler.impl import SimplerAllscan
        P, D = self.P, self.D
        gammas = torch.stack([A[p].reshape(-1, D).prod(dim=0).reshape(D, 1) for p in range(P)])
        S_locals = torch.stack(S_totals)          # [P,D,D]
        outputs = torch.zeros(P, D, D, dtype=torch.float32)
        allscan = SimplerAllscan()
        allscan.build(D, D, 1, P, self.device_ids, self.platform)
        try:
            allscan.run(S_locals, gammas, outputs)
        finally:
            allscan.close()
        return outputs

    def forward(self, Q, K, V, A):
        P, L, C, D = self.P, self.L, self.C, self.D
        s_snaps, g_css, S_totals = [], [], []
        for p in range(P):
            s_snap, g_cs, S_total = self._stage1(p, Q[p], K[p], V[p], A[p])
            s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)

        if P == 1:
            S_recvs = [torch.zeros(D, D, dtype=torch.float32)]
        else:
            out = self._boundary(S_totals, A)                 # real multi-device AllScan
            S_recvs = [torch.zeros(D, D, dtype=torch.float32) if p == 0 else out[p - 1]
                       for p in range(P)]

        O = torch.zeros(P, L, D, dtype=torch.float32)
        for p in range(P):
            s_shift = _shift_snaps(s_snaps[p], A[p], S_recvs[p], L, C, D)
            O[p] = self._stage2(p, Q[p], K[p], V[p], g_css[p], s_shift)
        return O

    def close(self):
        for r in getattr(self, "_runners", []):
            r.close()
        self._runners = []
