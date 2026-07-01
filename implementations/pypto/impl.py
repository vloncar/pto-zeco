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
        # All `_B` rings compute the same AllScan from the same inputs, so ring
        # slice 0 is the result; used for the cold-start call and verification.
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

    def __init__(self) -> None:
        self._rt = None

    def build(self, dk, dv, K, P, device_ids, platform):
        self.close()

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        from implementations.pypto.program_backward import build_allscan_backward_program

        self.P = P
        dist_cfg = DistributedConfig(device_ids=device_ids[:P], num_sub_workers=0)
        program = build_allscan_backward_program(dk, dv, K, P)
        compiled = ir.compile(program, platform=platform, distributed_config=dist_cfg)

        # Shared-memory IO buffers must exist before prepare() forks chip workers.
        self._host_gout = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_g = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
        self._host_outprev = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_dS = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_dgamma = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
        self._rt = compiled.prepare()

    def run(self, S_locals, gammas, outputs):
        raise NotImplementedError("PytoAllscanBackward implements run_backward only")

    def run_backward(self, g_out, gammas, outs, dS, dgamma):
        assert self._rt is not None, "call build() first"
        self._host_gout.copy_(g_out)
        self._host_g.copy_(gammas)
        # out_prev[r] = out[r-1]; rank 0 has no predecessor (dgamma[0] == 0).
        self._host_outprev.zero_()
        self._host_outprev[1:].copy_(outs[:-1])
        self._host_dS.zero_()
        self._host_dgamma.zero_()
        self._rt(self._host_gout, self._host_g, self._host_outprev, self._host_dS, self._host_dgamma)
        dS.copy_(self._host_dS)
        dgamma.copy_(self._host_dgamma)

    def close(self):
        if self._rt is not None:
            self._rt.close()
            self._rt = None
