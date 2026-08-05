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

    Each kernel invocation stands up a fresh single-callable L3 worker
    (``device_ids=[one]``, ``num_sub_workers=0``), submits one chip task via an
    ``orch_fn`` (the AllScan launch pattern), and tears it down. The kernel *compile*
    is session-cached, so only the (cheap) worker init/close repeats. Tensors are
    staged through shared memory (the chip child is a subprocess).

    **This cycling is forced, not incidental** (re-measured 2026-07-22, F4.1). Two
    runtime constraints combine:
      * a device hosts one worker at a time (device-exclusive), and
      * **an L3 chip child binds to the first callable it runs.** A second dispatch of
        a *different* callable on the same worker does not raise — it silently returns
        wrong data. Registering every callable up front (all pre-``init()``, which is
        the supported ordering) does not help: the first dispatch is correct and the
        next one is silently garbage. Registering post-``init()`` at least fails loudly
        (``chip_process: run failed with code -1``).
    Measured: a persistent multi-callable worker computes dQ/dK/dV correctly (~4e-7)
    and dA at ``max_rel = 1.0`` — the reverse-cumsum is the 2nd ``gate_cumsum``
    dispatch on an already-used worker. See
    ``allscan/issues/simpler-second-callable-silent-corruption/``. Do not "optimise"
    this into a persistent worker without re-checking that issue.
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

        def orch_fn_for(cid):
            def orch_fn(orch, _args, cfg):
                chip_args = TaskArgs()
                for (_t, shm, d) in staged:
                    chip_args.add_tensor(make_tensor_arg(shm), _tag[d])
                orch.submit_next_level(cid, chip_args, cfg, worker=0)
            return orch_fn

        key = f"{name}:{self.platform}:{self.device_id}"
        cc = _compile_chip_callable_from_spec(_SPECS[name], self.platform, RUNTIME, key)
        worker = Worker(level=3, device_ids=[self.device_id], num_sub_workers=0,
                        platform=self.platform, runtime=RUNTIME)
        cid = worker.register(cc)
        worker.init()
        try:
            worker.run(orch_fn_for(cid), args=None, config=CallConfig())
        finally:
            worker.close()

        for (t, shm, d) in staged:
            if d in (D.OUT, D.INOUT):
                t.reshape(-1).copy_(shm.to(t.dtype))

    def close(self):
        pass


class _PersistentComputeRunner(_ComputeRunner):
    """Holds ONE multi-callable L3 worker across many dispatches (steady-state path).

    ``_ComputeRunner`` stands up and tears down a worker per kernel invocation, which
    is what dominates this backend's per-call wall-clock (~35-43 s at P=2/4; ROADMAP
    B5.3/F4). This variant pays that once and then only dispatches, so a benchmark can
    report steady-state operator latency comparable to the pypto backend's.

    Two runtime constraints shape the design, both re-verified on runtime ``9922afdb``
    by ``scratchpad/probe_persistent_worker.py`` (2026-08-05):

    * **Every ``share_memory_()`` buffer must exist before ``init()``.** ``init()``
      eagerly forks the chip child, and a buffer allocated afterwards is invisible to
      it (``run failed with code -1``). Hence the two-phase :meth:`prepare` (allocate +
      register, no device work) / :meth:`open` (``init``) split, and why every dispatch
      copies into the *same* buffers rather than allocating fresh ones.
    * **A device hosts one worker at a time.** The AllScan boundary worker therefore
      cannot coexist with these compute workers — callers must :meth:`close` around the
      boundary phase and :meth:`open` again after. Reopening is still far cheaper than
      the per-kernel cycling it replaces.

    Multi-callable reuse was previously a silent-corruption trap
    (``allscan/issues/simpler-second-callable-silent-corruption/``, runtime
    ``a756969c``). It is re-checked end-to-end against ``expected_gla`` before this path
    is used — see :meth:`SimplerZeCo._check_persistent`. Do not widen its use without
    re-running that check.
    """

    def __init__(self, device_id, platform):
        super().__init__(device_id, platform)
        self._staged: dict[str, list] = {}
        self._cids: dict[str, object] = {}
        self._callables: dict[str, object] = {}
        self._worker = None

    def prepare(self, dispatches):
        """Allocate every IO buffer and compile/hold every callable. No device work.

        Args:
            dispatches: ``(name, sig, named_tensors)`` triples — the same argument
                lists :meth:`run` will later be called with (values are irrelevant
                here; only shape/dtype/direction are captured).
        """
        from simpler_setup.scene_test import _compile_chip_callable_from_spec

        for name, sig, named_tensors in dispatches:
            staged = []
            for i, (_lbl, t) in enumerate(named_tensors):
                dt = torch.int64 if t.dtype == torch.int64 else torch.float32
                shm = torch.zeros(t.numel(), dtype=dt).share_memory_()
                staged.append((shm, sig[i]))
            self._staged[name] = staged
            key = f"{name}:{self.platform}:{self.device_id}"
            self._callables[name] = _compile_chip_callable_from_spec(
                _SPECS[name], self.platform, RUNTIME, key)

    def open(self):
        """Stand the worker up (registers all callables, then ``init``)."""
        if self._worker is not None:
            return
        from simpler.worker import Worker

        self._worker = Worker(level=3, device_ids=[self.device_id], num_sub_workers=0,
                              platform=self.platform, runtime=RUNTIME)
        for name, cc in self._callables.items():
            self._cids[name] = self._worker.register(cc)
        self._worker.init()

    def run(self, name, sig, named_tensors):
        """Dispatch ``name`` on the held worker, reusing its pre-allocated buffers."""
        from simpler.task_interface import ArgDirection, CallConfig, TaskArgs, TensorArgType
        from simpler_setup.torch_interop import make_tensor_arg

        assert name in self._staged, f"{name} was not prepare()d on device {self.device_id}"
        self.open()

        D = ArgDirection
        tag = {D.IN: TensorArgType.INPUT, D.OUT: TensorArgType.OUTPUT_EXISTING,
               D.INOUT: TensorArgType.INOUT}
        staged = self._staged[name]
        assert len(staged) == len(named_tensors), (
            f"{name}: prepared {len(staged)} args but dispatched {len(named_tensors)}")

        for (shm, d), (_lbl, t) in zip(staged, named_tensors):
            if d in (D.IN, D.INOUT):
                shm.copy_(t.reshape(-1).to(shm.dtype))
            else:
                shm.zero_()

        cid = self._cids[name]

        def orch_fn(orch, _args, cfg):
            chip_args = TaskArgs()
            for (shm, d) in staged:
                chip_args.add_tensor(make_tensor_arg(shm), tag[d])
            orch.submit_next_level(cid, chip_args, cfg, worker=0)

        self._worker.run(orch_fn, args=None, config=CallConfig())

        for (shm, d), (_lbl, t) in zip(staged, named_tensors):
            if d in (D.OUT, D.INOUT):
                t.reshape(-1).copy_(shm.to(t.dtype))

    def close(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None
            self._cids = {}


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

    # ------------------------------------------------------------------
    # Steady-state (amortized) path — see _PersistentComputeRunner.
    # ------------------------------------------------------------------

    def _persistent_dispatches(self):
        """``(name, sig, named_tensors)`` templates for every kernel the forward runs.

        Shapes/dtypes/directions only — the values are placeholders. Must list every
        dispatch the forward will make, because all buffers have to exist before the
        worker's ``init()`` forks the chip child.
        """
        L, dk, dv, N = self.L, self.dk, self.dv, self.N
        z = torch.zeros
        return [
            ("gate_cumsum", GATE_CUMSUM_SPEC["orchestration"]["signature"],
             [("tril", self._tril), ("g", z(L, dk)), ("g_cs", z(L, dk)),
              ("config", self._config)]),
            ("chunk_h", CHUNK_H_SPEC["orchestration"]["signature"],
             [("k", z(L, dk)), ("v", z(L, dv)), ("g_cs", z(L, dk)),
              ("s_snap", z(N, dk, dv)), ("config", self._config)]),
            ("chunk_o", CHUNK_O_SPEC["orchestration"]["signature"],
             [("q", z(L, dk)), ("k", z(L, dk)), ("v", z(L, dv)), ("g_cs", z(L, dk)),
              ("s_snap", z(N, dk, dv)), ("tril", self._tril), ("o", z(L, dv)),
              ("config", self._config)]),
        ]

    def _use_persistent_runners(self):
        """Swap in prepared (not yet opened) persistent runners."""
        runners = [_PersistentComputeRunner(d, self.platform) for d in self.device_ids]
        dispatches = self._persistent_dispatches()
        for r in runners:
            r.prepare(dispatches)
        self._runners = runners

    def _use_plain_runners(self):
        """Restore the per-kernel worker runners (the always-safe path)."""
        for r in getattr(self, "_runners", []):
            r.close()
        self._runners = [_ComputeRunner(d, self.platform) for d in self.device_ids]

    def _forward_persistent(self, Q, K, V, A):
        """:meth:`forward` on held workers.

        Identical to :meth:`forward` except the compute workers are closed around the
        AllScan boundary — a device hosts one worker at a time, so the boundary's
        distributed worker cannot coexist with them. Stage 2 reopens them.
        """
        P, L, C, dk, dv = self.P, self.L, self.C, self.dk, self.dv
        s_snaps, g_css, S_totals = [], [], []
        for p in range(P):
            s_snap, g_cs, S_total = self._stage1(p, Q[p], K[p], V[p], A[p])
            s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)

        if P == 1:
            S_recvs = [torch.zeros(dk, dv, dtype=torch.float32)]
        else:
            for r in self._runners:      # free the devices for the AllScan worker
                r.close()
            out = self._boundary(S_totals, A)
            S_recvs = [torch.zeros(dk, dv, dtype=torch.float32) if p == 0 else out[p - 1]
                       for p in range(P)]

        O = torch.zeros(P, L, dv, dtype=torch.float32)
        for p in range(P):
            s_shift = _shift_snaps(s_snaps[p], A[p], S_recvs[p], L, C, dk)
            O[p] = self._stage2(p, Q[p], K[p], V[p], g_css[p], s_shift)
        return O

    def measure(self, Q, K, V, A, n_iters):
        """Steady-state per-forward latency, with a correctness gate on the fast path.

        The persistent path reuses one multi-callable worker per device, which was a
        silent-corruption trap on an older runtime
        (``allscan/issues/simpler-second-callable-silent-corruption/``). So it is never
        trusted blind: one persistent forward is checked against the per-kernel-worker
        forward first, and any mismatch (or any error) falls back to the safe path with
        ``amortized_timing`` left False, so the benchmark reports what it actually
        measured.
        """
        import time
        import warnings

        self._use_plain_runners()
        baseline = self.forward(Q, K, V, A)

        try:
            self._use_persistent_runners()
            got = self._forward_persistent(Q, K, V, A)
            err = (got - baseline).abs().max().item()
        except Exception as exc:  # noqa: BLE001 — degrade to the safe path
            warnings.warn(f"simpler: persistent-worker path unusable ({type(exc).__name__}: "
                          f"{str(exc)[:120]}); timing the per-kernel path instead", stacklevel=2)
            self._use_plain_runners()
            return super().measure(Q, K, V, A, n_iters)

        tol = 1e-3 * max(1.0, float(baseline.abs().max()))
        if not (err < tol):
            warnings.warn(f"simpler: persistent-worker forward disagrees with the per-kernel "
                          f"forward (max_abs_err={err:.3e} > {tol:.1e}) — see "
                          f"allscan/issues/simpler-second-callable-silent-corruption/; "
                          f"timing the per-kernel path instead", stacklevel=2)
            self._use_plain_runners()
            return super().measure(Q, K, V, A, n_iters)

        lat_ms = []
        try:
            for _ in range(n_iters):
                t0 = time.perf_counter()
                self._forward_persistent(Q, K, V, A)
                lat_ms.append((time.perf_counter() - t0) * 1e3)
        finally:
            for r in self._runners:
                r.close()

        self.amortized_timing = True
        return lat_ms

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

    # --- boundary AllScan (factored so ONE built worker can serve many heads) ---
    # A device hosts one worker at a time, so the AllScan worker must be built +
    # closed AROUND the compute (never held while the per-rank compute workers run).
    # Within a boundary phase, though, one built worker can do many run()s — that is
    # the multi-head amortization (F7.4): build once, run per head, close once,
    # instead of a fresh HCCL distributed-worker build per head.

    def _make_allscan(self, multi_h=None):
        """Build (and return) a SimplerAllscan worker; caller must close() it.

        ``multi_h``: number of heads if the worker will drive ``run_multi`` /
        ``run_multi_backward`` — the per-head share_memory buffers must be
        allocated before the worker's eager chip-child fork (simpler #1397).
        """
        from allscan.implementations.simpler.impl import SimplerAllscan
        allscan = SimplerAllscan()
        allscan.build(self.dk, self.dv, 1, self.P, self.device_ids, self.platform, multi_h=multi_h)
        return allscan

    def _gammas(self, A):
        """Per-rank total decay gamma [P,dk,1] = prod of A over the rank's tokens."""
        dk = self.dk
        return torch.stack([A[p].reshape(-1, dk).prod(dim=0).reshape(dk, 1) for p in range(self.P)])

    def _boundary_on(self, allscan, S_totals, A):
        """One forward AllScan run on an already-built worker -> out [P,dk,dv]."""
        P, dk, dv = self.P, self.dk, self.dv
        outputs = torch.zeros(P, dk, dv, dtype=torch.float32)
        allscan.run(torch.stack(S_totals), self._gammas(A), outputs)
        return outputs

    def _boundary_backward_on(self, allscan, g_outs, A, outs):
        """One reverse-ring AllScan run on an already-built worker -> (dS, dgamma)."""
        P, dk, dv = self.P, self.dk, self.dv
        dS = torch.zeros(P, dk, dv, dtype=torch.float32)
        dgamma = torch.zeros(P, dk, 1, dtype=torch.float32)
        allscan.run_backward(g_outs, self._gammas(A), outs, dS, dgamma)
        return dS, dgamma

    def _boundary(self, S_totals, A):
        """Single-head forward boundary: build + one run + close."""
        allscan = self._make_allscan()
        try:
            return self._boundary_on(allscan, S_totals, A)
        finally:
            allscan.close()

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
        """Single-head reverse-ring boundary: build + one run_backward + close."""
        allscan = self._make_allscan()
        try:
            return self._boundary_backward_on(allscan, g_outs, A, outs)
        finally:
            allscan.close()

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

    # ---------------------------------------------------------------------
    # Multi-head (F7.4): heads are independent GLA operators, so multi-head is H
    # single-head passes — but the boundary AllScan is BATCHED across heads. All
    # heads' stage-1 compute runs first (releasing the devices), then ONE AllScan
    # worker is built and run once per head, then all heads' stage-2 compute — so
    # the expensive HCCL distributed-worker build is paid ONCE per boundary phase
    # instead of once per head (the dominant P>1 per-head cost). Compute still
    # cycles per-kernel workers (runtime-forced). Same numerics as H single-head
    # passes; `MultiHeadZeCo` dispatches here when present.
    # ---------------------------------------------------------------------

    def forward_multihead(self, Q, K, V, A):
        """Multi-head forward with batched boundary; ``[P,H,L,dk/dv]`` in/out."""
        P, L, C, dk, dv = self.P, self.L, self.C, self.dk, self.dv
        H = Q.shape[1]
        zkv = torch.zeros(dk, dv, dtype=torch.float32)

        # Phase 1: stage-1 compute for every head (per-rank, per-kernel workers).
        heads = []                                    # per head: (s_snaps, g_css, S_totals)
        for h in range(H):
            s_snaps, g_css, S_totals = [], [], []
            for p in range(P):
                s_snap, g_cs, S_total = self._stage1(p, Q[p, h], K[p, h], V[p, h], A[p, h])
                s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)
            heads.append((s_snaps, g_css, S_totals))

        # Phase 2: ONE AllScan worker + ONE comm domain, all heads batched into
        # disjoint slots (run_multi) — amortizes both the HCCL worker build (F7.4)
        # and the per-head comm-domain alloc/free (F7.5a).
        if P == 1:
            S_recvs_h = [[zkv] for _ in range(H)]
        else:
            S_locals_list = [torch.stack(heads[h][2]) for h in range(H)]
            gammas_list = [self._gammas(A[:, h]) for h in range(H)]
            outs_list = [torch.zeros(P, dk, dv, dtype=torch.float32) for _ in range(H)]
            allscan = self._make_allscan(H)
            try:
                allscan.run_multi(S_locals_list, gammas_list, outs_list)
            finally:
                allscan.close()
            S_recvs_h = [[zkv if p == 0 else outs_list[h][p - 1] for p in range(P)]
                         for h in range(H)]

        # Phase 3: stage-2 compute for every head.
        O = torch.zeros(P, H, L, dv, dtype=torch.float32)
        for h in range(H):
            s_snaps, g_css, _ = heads[h]
            for p in range(P):
                s_shift = _shift_snaps(s_snaps[p], A[p, h], S_recvs_h[h][p], L, C, dk)
                O[p, h] = self._stage2(p, Q[p, h], K[p, h], V[p, h], g_css[p], s_shift)
        return O

    def backward_multihead(self, Q, K, V, A, dO):
        """Multi-head backward with batched fwd/bwd boundaries; ``[P,H,L,.]`` in/out."""
        P, L, C, dk, dv, N = self.P, self.L, self.C, self.dk, self.dv, self.N
        H = Q.shape[1]
        zkv = torch.zeros(dk, dv, dtype=torch.float32)

        # ---- Phase A: stage-1 + host decay quantities, all heads ----
        HA = []   # per head: (s_snaps, g_css, S_totals, gammas_n, cprev_n)
        for h in range(H):
            s_snaps, g_css, S_totals, gammas_n, cprev_n = [], [], [], [], []
            for p in range(P):
                s_snap, g_cs, S_total = self._stage1(p, Q[p, h], K[p, h], V[p, h], A[p, h])
                s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)
                g_last = g_cs.reshape(N, C, dk)[:, -1, :]
                gam = torch.exp(g_last)
                c = torch.ones(N, dk, dtype=torch.float32)
                if N > 1:
                    c[1:] = torch.cumprod(gam, dim=0)[:-1]
                gammas_n.append(gam); cprev_n.append(c)
            HA.append((s_snaps, g_css, S_totals, gammas_n, cprev_n))

        # ---- Phase B1: batched forward AllScan (one worker + one comm domain) ----
        outs_h = [None] * H
        S_recvs_h = []
        if P == 1:
            S_recvs_h = [[zkv] for _ in range(H)]
        else:
            S_locals_list = [torch.stack(HA[h][2]) for h in range(H)]
            gammas_list = [self._gammas(A[:, h]) for h in range(H)]
            outs_list = [torch.zeros(P, dk, dv, dtype=torch.float32) for _ in range(H)]
            allscan = self._make_allscan(H)
            try:
                allscan.run_multi(S_locals_list, gammas_list, outs_list)
            finally:
                allscan.close()
            for h in range(H):
                outs_h[h] = outs_list[h]
                S_recvs_h.append([zkv if p == 0 else outs_list[h][p - 1] for p in range(P)])

        # ---- Phase C: grad_o + host gate_o, all heads ----
        dq = torch.zeros(P, H, L, dk, dtype=torch.float32)
        dk_o = torch.zeros(P, H, L, dk, dtype=torch.float32)
        dgcs = torch.zeros(P, H, L, dk, dtype=torch.float32)
        dv_out = torch.zeros(P, H, L, dv, dtype=torch.float32)
        HC = []   # per head: (dH_all, dcprev_all, dS_recv)
        for h in range(H):
            s_snaps, g_css, _, _, cprev_n = HA[h]
            dH_all, dcprev_all = [], []
            dS_recv = torch.zeros(P, dk, dv, dtype=torch.float32)
            for p in range(P):
                Hf = _shift_snaps(s_snaps[p], A[p, h], S_recvs_h[h][p], L, C, dk)
                dQt, dKin, dVi, dH = self._grad_o(p, Q[p, h], K[p, h], V[p, h], g_css[p], Hf, dO[p, h])
                e = torch.exp(g_css[p]); ei = torch.exp(-g_css[p])
                dqo = dQt * e; dko = dKin * ei
                dq[p, h] = dqo; dk_o[p, h] = dko
                dgcs[p, h] = dqo * Q[p, h] - dko * K[p, h]
                dv_out[p, h] = dVi
                dcp = torch.zeros(N, dk, dtype=torch.float32)
                acc = torch.zeros(dk, dv, dtype=torch.float32)
                for n in range(N):
                    dcp[n] = (dH[n] * S_recvs_h[h][p]).sum(dim=1)
                    acc += cprev_n[p][n].unsqueeze(1) * dH[n]
                dH_all.append(dH); dcprev_all.append(dcp); dS_recv[p] = acc
            HC.append((dH_all, dcprev_all, dS_recv))

        # ---- Phase B2: batched backward AllScan (one worker + one comm domain) ----
        dS_totals_h, dgammas_h = [], []
        if P == 1:
            dS_totals_h = [[zkv] for _ in range(H)]
            dgammas_h = [[torch.zeros(dk, dtype=torch.float32)] for _ in range(H)]
        else:
            g_out_list = []
            for h in range(H):
                g_out = torch.zeros(P, dk, dv, dtype=torch.float32)
                g_out[:P - 1] = HC[h][2][1:]
                g_out_list.append(g_out)
            gammas_list = [self._gammas(A[:, h]) for h in range(H)]
            dS_list = [torch.zeros(P, dk, dv, dtype=torch.float32) for _ in range(H)]
            dgamma_list = [torch.zeros(P, dk, 1, dtype=torch.float32) for _ in range(H)]
            allscan = self._make_allscan(H)
            try:
                allscan.run_multi_backward(g_out_list, gammas_list, outs_h, dS_list, dgamma_list)
            finally:
                allscan.close()
            for h in range(H):
                dS_totals_h.append([dS_list[h][p] for p in range(P)])
                dgammas_h.append([dgamma_list[h][p].squeeze(1) for p in range(P)])

        # ---- Phase D: reverse recurrence + grad_h + gate_h + reverse-cumsum ----
        dA = torch.zeros(P, H, L, dk, dtype=torch.float32)
        dk_full = torch.zeros(P, H, L, dk, dtype=torch.float32)
        for h in range(H):
            s_snaps, g_css, _, gammas_n, cprev_n = HA[h]
            dH_all, dcprev_all, _ = HC[h]
            for p in range(P):
                dSloc = torch.zeros(N, dk, dv, dtype=torch.float32)
                dcvec = torch.zeros(N, dk, dtype=torch.float32)
                cur_S = dS_totals_h[h][p].clone(); cur_c = dgammas_h[h][p].clone()
                for m in reversed(range(N)):
                    dSloc[m] = cur_S; dcvec[m] = cur_c
                    if m > 0:
                        cur_S = gammas_n[p][m].unsqueeze(1) * cur_S + dH_all[p][m]
                        cur_c = gammas_n[p][m] * cur_c + dcprev_all[p][m]

                dKstate, dVs = self._grad_h(p, K[p, h], V[p, h], g_css[p], dSloc)
                dgcs_p = dgcs[p, h].clone()
                dk_h = torch.zeros(L, dk, dtype=torch.float32)
                g_cs_ch = g_css[p].reshape(N, C, dk)
                for n in range(N):
                    lo, hi = n * C, (n + 1) * C
                    gtot = g_cs_ch[n, -1, :]
                    dkh = dKstate[lo:hi] * torch.exp(gtot.unsqueeze(0) - g_css[p][lo:hi])
                    dk_h[lo:hi] = dkh
                    dgcs_p[lo:hi] += -dkh * K[p, h][lo:hi]
                    dgamma_state = (dSloc[n] * s_snaps[p][n]).sum(dim=1)
                    dgamma_c = dcvec[n] * cprev_n[p][n]
                    dgcs_p[hi - 1] += (dgamma_state + dgamma_c) * gammas_n[p][n] + (dkh * K[p, h][lo:hi]).sum(dim=0)
                dv_out[p, h] += dVs
                dk_full[p, h] = dk_o[p, h] + dk_h
                dP = self._reverse_cumsum(p, dgcs_p)
                dA[p, h] = dP / A[p, h]
        return dq, dk_full, dv_out, dA

    def close(self):
        for r in getattr(self, "_runners", []):
            r.close()
        self._runners = []
