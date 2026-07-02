"""PyPTO DSL AllScan — benchmark/test adapter.

Wraps :func:`build_allscan_program` (the ``pypto.language`` DSL program in
``program.py``) behind the :class:`AllscanImpl` interface.

The program is compiled once per config via ``pypto.ir.compile`` and then a
reusable ``DistributedWorker`` is stood up via ``compiled.prepare()`` — the
"prepare once, dispatch many" path. This mirrors the ``simpler`` backend's
persistent ``Worker`` (build-once / run-many) so the two are a fair comparison,
and it avoids the one-shot ``compiled(*args)`` path which forks + tears down a
fresh L3 Worker on *every* call (≈7 s/call, dominated by setup).

The ``DistributedWorker`` requires every IO buffer to be a shared-memory host
tensor allocated **before** ``prepare()`` and reused in place, so ``build``
allocates per-rank-stacked shared tensors and ``run`` copies in/out of them.
"""

from __future__ import annotations

import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import AllscanImpl  # noqa: E402
from implementations.pypto.batched_program import make_batched_builder  # noqa: E402


class PytoAllscan(AllscanImpl):
    """PyPTO DSL-compiled AllScan, dispatched on a reusable DistributedWorker.

    A single dispatch of one AllScan pays a full comm-domain alloc/free + drain
    round-trip — the dominant per-call cost. To compare the marginal kernel+comm
    cost against the simpler backend on equal footing, this backend runs a single
    *batched* program: ``_B`` independent AllScans in ONE dispatch under one comm
    domain (each ring on its OWN window buffers + a disjoint output slice — see
    :mod:`implementations.pypto.batched_program`), so that fixed overhead is paid
    once per batch. :meth:`measure` divides the batch dispatch by ``_B`` to report
    the marginal cost, mirroring simpler's batched ``measure`` exactly.

    NOTE: only ONE DistributedWorker may be prepared per device set at a time —
    two prepared workers fork chip processes on the same devices and their HCCL
    comms collide (``HcclCommInitRootInfo failed: 7``). So run()/verify reuse the
    batched worker and read ring slice 0 rather than preparing a second program.
    """

    name = "pypto"

    #: Number of AllScans dispatched per batched timing sample (and per dispatch).
    _MEASURE_BATCH = 16

    def __init__(self) -> None:
        self._rt = None

    def build(self, dk, dv, K, P, device_ids, platform):
        """Compile the batched program and prepare the DistributedWorker.

        Args as in :meth:`common.AllscanImpl.build`. Compiles a ``_B``-ring
        batched program (for amortized timing) and prepares a single reusable
        worker; ``run``/verify read ring slice 0.
        """
        # Release any DistributedWorker from a previous config first: it holds a
        # forked L3 Worker whose chip children leak if not closed.
        self.close()

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        self.P = P
        self._B = self._MEASURE_BATCH
        dist_cfg = DistributedConfig(device_ids=device_ids[:P], num_sub_workers=0)

        # Batched program: `_B` independent rings per dispatch under one comm
        # domain. Generated with `_B` explicit disjoint window-buffer pairs (see
        # batched_program); the single-ring program.py is its source of truth.
        program_b = make_batched_builder(self._B)(dk, dv, K, P)
        compiled_b = ir.compile(program_b, platform=platform, distributed_config=dist_cfg)

        # Shared-memory IO buffers must exist before prepare() forks the chip
        # workers; they are reused in place across every dispatch. Inputs are
        # shared across all `_B` rings; each ring writes its own output slice.
        self._host_s = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_g = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
        self._host_out_b = torch.zeros((self._B, P, dk, dv), dtype=torch.float32).share_memory_()
        self._rt = compiled_b.prepare()

    def _dispatch(self):
        """Copy inputs in and run one batched dispatch (`_B` rings)."""
        self._host_out_b.zero_()
        self._rt(self._host_s, self._host_g, self._host_out_b)

    def run(self, S_locals, gammas, outputs):
        """Forward AllScan; args as in :meth:`common.AllscanImpl.run`.

        All ``_B`` rings compute the same AllScan from the same inputs, so ring
        slice 0 is the result (used for the cold-start call and verification).
        """
        assert self._rt is not None, "call build() first"
        self._host_s.copy_(S_locals)
        self._host_g.copy_(gammas)
        self._dispatch()
        outputs.copy_(self._host_out_b[0])

    def run_batch(self, S_locals, gammas, n_iters: int) -> float:
        """Dispatch ``n_iters`` independent AllScans inside ONE dispatch under a
        single comm domain (each ring on its own window buffers + output slice).
        Returns total wall time (seconds); ``total / n_iters`` is the marginal
        kernel+comm cost. Requires n_iters == self._B (the compiled batch size).

        Args:
            S_locals: Per-rank local state, ``[P, dk, dv]`` (shared by all rings).
            gammas: Per-rank decay factors, ``[P, dk, 1]``.
            n_iters: Number of rings; must equal ``self._B``.

        Returns:
            Total wall time for the batched dispatch, in seconds.
        """
        assert self._rt is not None, "call build() first"
        assert n_iters == self._B, f"batched program compiled for B={self._B}, got {n_iters}"
        self._host_s.copy_(S_locals)
        self._host_g.copy_(gammas)
        self._host_out_b.zero_()
        t0 = time.perf_counter()
        self._rt(self._host_s, self._host_g, self._host_out_b)
        return time.perf_counter() - t0

    #: pypto amortizes the per-call comm-domain + drain overhead in measure(),
    #: matching the simpler backend so the two are directly comparable.
    amortized_timing = True

    def measure(self, S_locals, gammas, outputs, n_iters):
        """Per-iteration samples with per-dispatch orchestration overhead amortized.

        Each sample is one batched dispatch of ``self._B`` AllScans divided by the
        batch size; ``n_iters`` such samples form the distribution.
        """
        batch = self._B
        return [self.run_batch(S_locals, gammas, batch) / batch * 1e3 for _ in range(n_iters)]

    def close(self):
        if self._rt is not None:
            self._rt.close()
            self._rt = None


class PytoAllscanBackward(AllscanImpl):
    """PyPTO DSL-compiled AllScan *backward* pass on a reusable DistributedWorker.

    Separate from :class:`PytoAllscan` because only ONE DistributedWorker may be
    prepared per device set at a time (two prepared workers fork chip processes
    on the same devices and their HCCL comms collide, ``HcclCommInitRootInfo
    failed: 7``). This class prepares only the backward program; the forward
    class prepares only the forward one. Each is built/run/closed independently.

    Reverse-ring program (see :mod:`implementations.pypto.program_backward`):
    inputs g_out, gamma, out_prev; outputs dS, dgamma. ``out_prev[r] = out[r-1]``
    (the block rank r received during the forward pass; zeros for rank 0), so the
    dgamma row-reduction is fully local.
    """

    name = "pypto"

    #: Number of backward passes dispatched per batched timing sample.
    _MEASURE_BATCH = 16

    def __init__(self) -> None:
        self._rt = None

    def build(self, dk, dv, K, P, device_ids, platform):
        """Compile the batched backward program and prepare the worker.

        Args as in :meth:`common.AllscanImpl.build`. Prepares only the backward
        worker (one worker per device set); ``run_backward``/verify read slice 0.
        """
        self.close()

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        from implementations.pypto.batched_backward_program import make_batched_backward_builder

        self.P = P
        self._B = self._MEASURE_BATCH
        dist_cfg = DistributedConfig(device_ids=device_ids[:P], num_sub_workers=0)

        # Batched backward program: `_B` independent rings per dispatch under one
        # comm domain (each on its OWN window buffers + output slice), so the
        # fixed comm-domain overhead is amortized in measure_backward. The
        # single-ring program_backward.py is its source of truth. As in the
        # forward class, only ONE DistributedWorker may be prepared per device
        # set, so run_backward/verify reuse this worker and read ring slice 0.
        program_b = make_batched_backward_builder(self._B)(dk, dv, K, P)
        compiled_b = ir.compile(program_b, platform=platform, distributed_config=dist_cfg)

        # Shared-memory IO buffers must exist before prepare() forks chip workers.
        # Inputs are shared across all `_B` rings; each ring writes its own slice.
        self._host_gout = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_g = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
        self._host_outprev = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_dS_b = torch.zeros((self._B, P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_dgamma_b = torch.zeros((self._B, P, dk, 1), dtype=torch.float32).share_memory_()
        self._rt = compiled_b.prepare()

    def run(self, S_locals, gammas, outputs):
        """Not implemented — this class is backward-only (use :class:`PytoAllscan`
        for the forward pass). Args match the interface but always raise."""
        raise NotImplementedError("PytoAllscanBackward implements run_backward only")

    def _copy_inputs(self, g_out, gammas, outs):
        """Stage shared inputs into the host buffers.

        Args:
            g_out: Upstream gradient, ``[P, dk, dv]``.
            gammas: Per-rank decay factors, ``[P, dk, 1]``.
            outs: Retained forward outputs, ``[P, dk, dv]``; ``out_prev[r]`` is set
                to ``outs[r-1]`` (row 0 left zero — rank 0 has no predecessor).
        """
        self._host_gout.copy_(g_out)
        self._host_g.copy_(gammas)
        # out_prev[r] = out[r-1]; rank 0 has no predecessor (dgamma[0] == 0).
        self._host_outprev.zero_()
        self._host_outprev[1:].copy_(outs[:-1])

    def run_backward(self, g_out, gammas, outs, dS, dgamma):
        """Backward AllScan; args as in :meth:`common.AllscanImpl.run_backward`.

        All ``_B`` rings compute the same result from the same inputs, so ring
        slice 0 is the answer (used for the cold-start call and verification).
        """
        assert self._rt is not None, "call build() first"
        self._copy_inputs(g_out, gammas, outs)
        self._host_dS_b.zero_()
        self._host_dgamma_b.zero_()
        self._rt(self._host_gout, self._host_g, self._host_outprev, self._host_dS_b, self._host_dgamma_b)
        dS.copy_(self._host_dS_b[0])
        dgamma.copy_(self._host_dgamma_b[0])

    def run_batch_backward(self, g_out, gammas, outs, n_iters: int) -> float:
        """Dispatch ``n_iters`` backward passes in ONE dispatch under a single
        comm domain (each ring on its own window buffers + output slice). Returns
        total wall time (seconds); ``total / n_iters`` is the marginal cost.
        Requires n_iters == self._B (the compiled batch size).

        Args:
            g_out: Upstream gradient, ``[P, dk, dv]`` (shared by all rings).
            gammas: Per-rank decay factors, ``[P, dk, 1]``.
            outs: Retained forward outputs, ``[P, dk, dv]`` (for ``out_prev``).
            n_iters: Number of rings; must equal ``self._B``.

        Returns:
            Total wall time for the batched dispatch, in seconds.
        """
        assert self._rt is not None, "call build() first"
        assert n_iters == self._B, f"batched program compiled for B={self._B}, got {n_iters}"
        self._copy_inputs(g_out, gammas, outs)
        self._host_dS_b.zero_()
        self._host_dgamma_b.zero_()
        t0 = time.perf_counter()
        self._rt(self._host_gout, self._host_g, self._host_outprev, self._host_dS_b, self._host_dgamma_b)
        return time.perf_counter() - t0

    #: pypto amortizes the per-dispatch comm-domain overhead in measure_backward.
    amortized_timing = True

    def measure_backward(self, g_out, gammas, outs, dS, dgamma, n_iters):
        """Amortized backward samples: one batched dispatch of `_B` passes / `_B`."""
        batch = self._B
        return [self.run_batch_backward(g_out, gammas, outs, batch) / batch * 1e3 for _ in range(n_iters)]

    def close(self):
        if self._rt is not None:
            self._rt.close()
            self._rt = None
