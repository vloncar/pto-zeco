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

Requires ``L % C == 0``; ``C``, ``dk`` and ``dv`` may all differ and each must be
one of ``{16, 32, 64, 128}``.  The incore kernels dispatch the runtime tile dims
to compile-time templates (the ``benchmark_bgemm`` pattern): the matmul kernel
takes independent ``M, N, Kc`` (F3 Phase 2), so every GLA matmul —
``KV=[dk,dv]<-[C,·]`` (TN), ``inter=[C,dv]<-·[dk,dv]`` (NN), ``Aqk=[C,C]<-·`` (NT),
``intra=[C,dv]`` (NN) — and the rectangular vector stages (state ``[dk,dv]``,
gates ``[C,dk]``, values ``[C,dv]``) run at any ``C, dk, dv`` (F7).  Tiles above
128 (head dim 256) still need blocking (shared with pypto's ceiling).
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

# --- backward (B3): grad_o (output-stage) + grad_h (state-stage) ---
# Both reuse the general matmul + the forward prep/elt kernels, so they add no new
# device kernel; the cross-chunk grad recurrence + gate arithmetic run on host.
GRAD_O_SPEC = _spec(
    "kernels/orchestration/grad_o_orch.cpp",
    [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT, _D.OUT, _D.IN],
    [(0, "MM", "kernels/aic/matmul_kernel.cpp", "aic", [_D.IN, _D.IN, _D.OUT]),
     (1, "PREP", "kernels/aiv/chunk_o_prep.cpp", "aiv", [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]),
     (2, "ELT", "kernels/aiv/chunk_o_elt.cpp", "aiv", [_D.IN, _D.IN, _D.OUT])],
)

GRAD_H_SPEC = _spec(
    "kernels/orchestration/grad_h_orch.cpp",
    [_D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT, _D.IN],
    [(0, "MM", "kernels/aic/matmul_kernel.cpp", "aic", [_D.IN, _D.IN, _D.OUT]),
     (1, "PREP", "kernels/aiv/chunk_h_prep.cpp", "aiv", [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT])],
)


# ---------------------------------------------------------------------------
# Host-side pieces (pure torch; the ZeCO linearity glue — S_total advance,
# shift_snaps fold, gamma = A.prod — kept here so the whole backend stays in the
# base env with no torch-npu dependency)
# ---------------------------------------------------------------------------

def _S_total(s_snap, g_cs, k, v, L, C):
    """Advance the last chunk snapshot through the last chunk -> local end state [dk,dv].

    Shape-driven (g_cs/k are [.,dk], v is [.,dv]), so dk != dv is handled: the return
    is exp(g_total)[:,None]*s_snap[-1] + k_rest^T @ v_last = [dk,dv].
    """
    n_chunks = L // C
    off = (n_chunks - 1) * C
    g_cs_last = g_cs[off:off + C]
    g_total = g_cs_last[-1]
    k_last = k[off:off + C]
    v_last = v[off:off + C]
    k_rest = k_last * torch.exp(g_total.unsqueeze(0) - g_cs_last)
    return torch.exp(g_total).unsqueeze(1) * s_snap[-1] + k_rest.t() @ v_last


def _shift_snaps(s_snap, A_rank, S_recv, L, C, dk):
    """Fold the received boundary state into the chunk snapshots (host, fp32).

    Gates are per key dim, so the cumulative decay ``c`` is [n_chunks, dk] and
    broadcasts over the dv columns of the [dk,dv] state.
    """
    n_chunks = L // C
    A_ch = A_rank.reshape(n_chunks, C, dk).prod(dim=1)         # [n_chunks, dk]
    c = torch.ones(n_chunks, dk)
    if n_chunks > 1:
        c[1:] = torch.cumprod(A_ch, dim=0)[:-1]
    return s_snap + c.unsqueeze(-1) * S_recv.unsqueeze(0)


# ---------------------------------------------------------------------------
# Single-device compute runner (reuses the SceneTestCase L2 harness internals)
# ---------------------------------------------------------------------------

_SPECS = {"gate_cumsum": GATE_CUMSUM_SPEC, "chunk_h": CHUNK_H_SPEC, "chunk_o": CHUNK_O_SPEC,
          "grad_o": GRAD_O_SPEC, "grad_h": GRAD_H_SPEC}


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
        assert L % C == 0, f"L={L} not divisible by C={C}"
        # The incore kernels dispatch each runtime tile dim to a compile-time template
        # over {16,32,64,128}; the matmul kernel is rectangular (M,N,Kc), so C, dk and
        # dv may all differ, each must be a dispatchable size.  Tiles > 128 need blocking.
        for nm, val in (("C", C), ("dk", dk), ("dv", dv)):
            assert val in (16, 32, 64, 128), (
                f"simpler GLA {nm} must be one of {{16,32,64,128}}; got {nm}={val}")
        self.P, self.L, self.C, self.dk, self.dv = P, L, C, dk, dv
        self.device_ids = list(device_ids[:P])
        self.platform = platform
        self.N = L // C
        self._tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        # triu = tril^T: feeding it to the gate_cumsum kernel computes the REVERSE
        # cumulative sum (triu @ dg_cs), the gate backward's b->a chain (B3).
        self._triu = torch.triu(torch.ones(C, C, dtype=torch.float32))
        self._config = torch.tensor([C, dk, dv, self.N], dtype=torch.int64)
        # Fully distributed: rank p's GLA shard is computed on its OWN device
        # (device_ids[p]) and the boundary state is exchanged over the real
        # multi-device AllScan collective. One compute runner per rank/device.
        self._runners = [_ComputeRunner(d, platform) for d in self.device_ids]

    def _stage1(self, p, Q, K, V, A):
        """Run gate_cumsum + chunk_h on rank p's device -> (s_snap, g_cs, S_total)."""
        L, C, dk, dv, N = self.L, self.C, self.dk, self.dv, self.N
        r = self._runners[p]
        g_log = torch.log(A).contiguous()

        g_cs = torch.zeros(L, dk, dtype=torch.float32)
        r.run("gate_cumsum", GATE_CUMSUM_SPEC["orchestration"]["signature"],
              [("tril", self._tril), ("g", g_log), ("g_cs", g_cs), ("config", self._config)])

        s_snap = torch.zeros(N, dk, dv, dtype=torch.float32)
        r.run("chunk_h", CHUNK_H_SPEC["orchestration"]["signature"],
              [("k", K.contiguous()), ("v", V.contiguous()), ("g_cs", g_cs),
               ("s_snap", s_snap), ("config", self._config)])

        S_total = _S_total(s_snap, g_cs, K, V, L, C)
        return s_snap, g_cs, S_total

    def _stage2(self, p, Q, K, V, g_cs, s_shift):
        """Run chunk_o on rank p's device -> O [L,dv]."""
        L, dv = self.L, self.dv
        o = torch.zeros(L, dv, dtype=torch.float32)
        self._runners[p].run(
            "chunk_o", CHUNK_O_SPEC["orchestration"]["signature"],
            [("q", Q.contiguous()), ("k", K.contiguous()), ("v", V.contiguous()),
             ("g_cs", g_cs), ("s_snap", s_shift.contiguous()), ("tril", self._tril),
             ("o", o), ("config", self._config)])
        return o

    def _boundary(self, S_totals, A):
        """AllScan prefix: returns out [P,dk,dv] (out[p] = S_total[p] + gamma[p]*out[p-1]).

        The AllScan worker is built + run + closed here so it never holds the
        devices while the per-rank compute workers (also created/closed per call)
        need them — a device hosts one worker at a time.
        """
        from allscan.implementations.simpler.impl import SimplerAllscan
        P, dk, dv = self.P, self.dk, self.dv
        gammas = torch.stack([A[p].reshape(-1, dk).prod(dim=0).reshape(dk, 1) for p in range(P)])
        S_locals = torch.stack(S_totals)          # [P,dk,dv]
        outputs = torch.zeros(P, dk, dv, dtype=torch.float32)
        allscan = SimplerAllscan()
        allscan.build(dk, dv, 1, P, self.device_ids, self.platform)
        try:
            allscan.run(S_locals, gammas, outputs)
        finally:
            allscan.close()
        return outputs

    def forward(self, Q, K, V, A):
        P, L, C, dk, dv = self.P, self.L, self.C, self.dk, self.dv
        s_snaps, g_css, S_totals = [], [], []
        for p in range(P):
            s_snap, g_cs, S_total = self._stage1(p, Q[p], K[p], V[p], A[p])
            s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)

        if P == 1:
            S_recvs = [torch.zeros(dk, dv, dtype=torch.float32)]
        else:
            out = self._boundary(S_totals, A)                 # real multi-device AllScan
            S_recvs = [torch.zeros(dk, dv, dtype=torch.float32) if p == 0 else out[p - 1]
                       for p in range(P)]

        O = torch.zeros(P, L, dv, dtype=torch.float32)
        for p in range(P):
            s_shift = _shift_snaps(s_snaps[p], A[p], S_recvs[p], L, C, dk)
            O[p] = self._stage2(p, Q[p], K[p], V[p], g_css[p], s_shift)
        return O

    # ---------------------------------------------------------------------
    # Backward (B3): SP-decomposed GLA operator backward on the simpler kernels.
    # grad_o (output stage) + grad_h (state stage) run the per-chunk backward
    # matmuls on device; the AllScan-backward reverse ring carries the boundary
    # gradient across devices; the cross-chunk grad recurrence + the gate
    # arithmetic (dq/dk scaling, dg_cs assembly, reverse-cumsum -> dA) are the
    # cheap host linear glue.  Mirrors gla.common.gla_chunk_backward op-for-op.
    # ---------------------------------------------------------------------

    def _grad_o(self, p, Q, K, V, g_cs, H, dO):
        """Run grad_o on rank p's device -> (dQt, dKin, dVi, dH) raw adjoints."""
        L, dk, dv, N = self.L, self.dk, self.dv, self.N
        dQt = torch.zeros(L, dk, dtype=torch.float32)
        dKin = torch.zeros(L, dk, dtype=torch.float32)
        dVi = torch.zeros(L, dv, dtype=torch.float32)
        dH = torch.zeros(N, dk, dv, dtype=torch.float32)
        self._runners[p].run(
            "grad_o", GRAD_O_SPEC["orchestration"]["signature"],
            [("q", Q.contiguous()), ("k", K.contiguous()), ("v", V.contiguous()),
             ("g_cs", g_cs), ("snap", H.contiguous()), ("dO", dO.contiguous()),
             ("tril", self._tril), ("dQt", dQt), ("dKin", dKin), ("dVi", dVi),
             ("dH", dH), ("config", self._config)])
        return dQt, dKin, dVi, dH

    def _grad_h(self, p, K, V, g_cs, dSloc):
        """Run grad_h on rank p's device -> (dKstate, dVs) raw state adjoints."""
        L, dk, dv = self.L, self.dk, self.dv
        dKstate = torch.zeros(L, dk, dtype=torch.float32)
        dVs = torch.zeros(L, dv, dtype=torch.float32)
        self._runners[p].run(
            "grad_h", GRAD_H_SPEC["orchestration"]["signature"],
            [("k", K.contiguous()), ("v", V.contiguous()), ("g_cs", g_cs),
             ("dSloc", dSloc.contiguous()), ("dKstate", dKstate), ("dVs", dVs),
             ("config", self._config)])
        return dKstate, dVs

    def _reverse_cumsum(self, p, dg_cs):
        """Per-chunk reverse cumulative sum via the gate_cumsum kernel + triu."""
        L, dk = self.L, self.dk
        out = torch.zeros(L, dk, dtype=torch.float32)
        self._runners[p].run(
            "gate_cumsum", GATE_CUMSUM_SPEC["orchestration"]["signature"],
            [("triu", self._triu), ("dg_cs", dg_cs.contiguous()), ("out", out),
             ("config", self._config)])
        return out

    def _boundary_backward(self, g_outs, A, outs):
        """AllScan reverse ring: g_out[p] -> (dS_total[p], dgamma[p]) [dk,dv]/[dk,1]."""
        from allscan.implementations.simpler.impl import SimplerAllscan
        P, dk, dv = self.P, self.dk, self.dv
        gammas = torch.stack([A[p].reshape(-1, dk).prod(dim=0).reshape(dk, 1) for p in range(P)])
        dS = torch.zeros(P, dk, dv, dtype=torch.float32)
        dgamma = torch.zeros(P, dk, 1, dtype=torch.float32)
        allscan = SimplerAllscan()
        allscan.build(dk, dv, 1, P, self.device_ids, self.platform)
        try:
            allscan.run_backward(g_outs, gammas, outs, dS, dgamma)
        finally:
            allscan.close()
        return dS, dgamma

    def backward(self, Q, K, V, A, dO):
        """SP-decomposed ZeCO backward; args/return as in ZeCoImpl.backward."""
        P, L, C, dk, dv, N = self.P, self.L, self.C, self.dk, self.dv, self.N
        zkv = torch.zeros(dk, dv, dtype=torch.float32)

        # --- Phase A: forward stage1 per rank (g_cs, unfolded snaps, S_total) ---
        s_snaps, g_css, S_totals = [], [], []
        for p in range(P):
            s_snap, g_cs, S_total = self._stage1(p, Q[p], K[p], V[p], A[p])
            s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)

        # Per-rank host decay quantities: gamma_n [N,dk], cprev_n [N,dk].
        gammas_n, cprev_n = [], []
        for p in range(P):
            g_last = g_css[p].reshape(N, C, dk)[:, -1, :]      # [N,dk] = g_total per chunk
            gam = torch.exp(g_last)
            c = torch.ones(N, dk, dtype=torch.float32)
            if N > 1:
                c[1:] = torch.cumprod(gam, dim=0)[:-1]
            gammas_n.append(gam); cprev_n.append(c)

        # --- Phase B1: forward AllScan -> outs (the S_recv values) ---
        if P == 1:
            outs = None
            S_recvs = [zkv]
        else:
            outs = self._boundary(S_totals, A)
            S_recvs = [zkv if p == 0 else outs[p - 1] for p in range(P)]

        # --- Phase C: grad_o per rank + host gate_o -> dq, dk_o, dg_cs_o, dH, dS_recv ---
        dq = torch.zeros(P, L, dk, dtype=torch.float32)
        dk_o = torch.zeros(P, L, dk, dtype=torch.float32)
        dgcs = torch.zeros(P, L, dk, dtype=torch.float32)
        dv_out = torch.zeros(P, L, dv, dtype=torch.float32)
        dH_all, dcprev_all = [], []
        dS_recv = torch.zeros(P, dk, dv, dtype=torch.float32)
        for p in range(P):
            H = _shift_snaps(s_snaps[p], A[p], S_recvs[p], L, C, dk)
            dQt, dKin, dVi, dH = self._grad_o(p, Q[p], K[p], V[p], g_css[p], H, dO[p])
            e = torch.exp(g_css[p]); ei = torch.exp(-g_css[p])
            dqo = dQt * e
            dko = dKin * ei
            dq[p] = dqo
            dk_o[p] = dko
            dgcs[p] = dqo * Q[p] - dko * K[p]               # dg_cs output stage
            dv_out[p] = dVi
            dH_all.append(dH)
            # boundary-state grad (fed to the reverse ring) + dcprev for dcvec.
            dcp = torch.zeros(N, dk, dtype=torch.float32)
            acc = torch.zeros(dk, dv, dtype=torch.float32)
            for n in range(N):
                dcp[n] = (dH[n] * S_recvs[p]).sum(dim=1)
                acc += cprev_n[p][n].unsqueeze(1) * dH[n]
            dcprev_all.append(dcp)
            dS_recv[p] = acc

        # --- Phase B2: backward AllScan reverse ring -> dS_total, dgamma ---
        if P == 1:
            dS_totals = [zkv]
            dgammas = [torch.zeros(dk, dtype=torch.float32)]
        else:
            g_out = torch.zeros(P, dk, dv, dtype=torch.float32)
            g_out[:P - 1] = dS_recv[1:]
            dS_b, dgamma_b = self._boundary_backward(g_out, A, outs)
            dS_totals = [dS_b[p] for p in range(P)]
            dgammas = [dgamma_b[p].squeeze(1) for p in range(P)]

        # --- Phase D: reverse recurrence + grad_h + gate_h + reverse-cumsum ---
        dA = torch.zeros(P, L, dk, dtype=torch.float32)
        dk_full = torch.zeros(P, L, dk, dtype=torch.float32)
        for p in range(P):
            # reverse chunk recurrence -> dSloc[N,dk,dv], dcvec[N,dk] (host glue).
            dSloc = torch.zeros(N, dk, dv, dtype=torch.float32)
            dcvec = torch.zeros(N, dk, dtype=torch.float32)
            cur_S = dS_totals[p].clone(); cur_c = dgammas[p].clone()
            for m in reversed(range(N)):
                dSloc[m] = cur_S; dcvec[m] = cur_c
                if m > 0:
                    cur_S = gammas_n[p][m].unsqueeze(1) * cur_S + dH_all[p][m]
                    cur_c = gammas_n[p][m] * cur_c + dcprev_all[p][m]

            dKstate, dVs = self._grad_h(p, K[p], V[p], g_css[p], dSloc)

            # gate_h: dk_h, dg_cs_h + the g_total (row C-1) corrections, per chunk.
            dgcs_p = dgcs[p].clone()
            dk_h = torch.zeros(L, dk, dtype=torch.float32)
            g_cs_ch = g_css[p].reshape(N, C, dk)
            for n in range(N):
                lo, hi = n * C, (n + 1) * C
                gtot = g_cs_ch[n, -1, :]
                dkh = dKstate[lo:hi] * torch.exp(gtot.unsqueeze(0) - g_css[p][lo:hi])
                dk_h[lo:hi] = dkh
                dgcs_p[lo:hi] += -dkh * K[p][lo:hi]
                dgamma_state = (dSloc[n] * s_snaps[p][n]).sum(dim=1)     # [dk]
                dgamma_c = dcvec[n] * cprev_n[p][n]                      # [dk]
                dgcs_p[hi - 1] += (dgamma_state + dgamma_c) * gammas_n[p][n] + (dkh * K[p][lo:hi]).sum(dim=0)
            dv_out[p] += dVs
            dk_full[p] = dk_o[p] + dk_h
            # gate backward: dA = reverse_cumsum(dg_cs) / a  (triu matmul on device).
            dP = self._reverse_cumsum(p, dgcs_p)
            dA[p] = dP / A[p]

        return dq, dk_full, dv_out, dA

    def close(self):
        for r in getattr(self, "_runners", []):
            r.close()
        self._runners = []
