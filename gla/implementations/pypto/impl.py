"""PyPTO DSL ZeCO / GLA forward — benchmark/test adapter.

:class:`PyPtoZeCo` runs the **entire** ZeCO forward as ONE fully-fused distributed
``@pl.program`` (:mod:`.fused_program`). Per rank ``r`` (device ``r``) the single host
orchestrator runs three phases, all as distributed InCore chip kernels — no ``@pl.jit``,
no host round-trip:

  1. **stage1** — local end-of-slice state ``S_total[r]`` (chunk-recurrent scan from ``S=0``).
  2. **AllScan ring** — the exclusive-prefix boundary scan
     ``out[p] = S_local[p] + gamma[p]*out[p-1]``; rank ``r`` receives ``out[r-1]`` (its
     boundary ``S_recv[r]``, zero for rank 0) into a device-local window.
  3. **stage2** — ``O[r]`` = the same chunk recurrence initialised from ``S_recv[r]``.

``gamma`` (device total decay) is ``A.prod`` over tokens, computed host-side.

**Steady-state timing.** A ``DistributedWorker`` prepare/close (fork chip workers + HCCL
comm-domain setup) costs ~8–9 s and dominates a single ``forward``. To make the *operator*
comparison against ``simpler`` fair, :meth:`build` prepares the worker **once** and allocates
the shared-memory IO buffers reused in place; :meth:`forward`/:meth:`measure` then only copy
inputs and dispatch (mirrors :class:`allscan.implementations.pypto.impl.PyPtoAllscan`). The
one-time prepare is therefore paid at build, and :attr:`amortized_timing` is True.

``P == 1`` is a native path (single rank, no boundary, ``S_recv = 0``) built by a separate
factory (:func:`gla.implementations.pypto.fused_program._build_p1_forward_program`). It
usually still compiles to a distributed program (has ``prepare``); should a config ever
compile it non-distributed, :meth:`forward` falls back to the per-call
:func:`run_fused_forward` path and :meth:`measure` to the default per-call timing.

Runs on both a2a3sim and a2a3 hardware. Every distributed / HCCL run must set
``LD_PRELOAD=<cann>/lib64/libhccl.so`` or the rootinfo handshake hangs.
"""

from __future__ import annotations

import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gla.common import ZeCoImpl  # noqa: E402
from gla.implementations.pypto.fused_program import (  # noqa: E402
    build_fused_forward_program,
    run_fused_forward,
)


class PyPtoZeCo(ZeCoImpl):
    """PyPTO ZeCO forward: the entire forward as one fully-fused distributed program,
    dispatched on a reusable ``DistributedWorker`` (prepare once / dispatch many)."""

    name = "pypto"

    #: build() prepares the worker once, so measure() times only steady-state dispatch.
    amortized_timing = True

    def __init__(self) -> None:
        self._compiled = None
        self._rt = None

    def build(self, P, L, C, dk, dv, device_ids, platform):
        """Compile the fully-fused ZeCO forward and prepare the reusable worker.

        Args as in :meth:`gla.common.ZeCoImpl.build`. ``C`` is the real chunk size
        (``L`` divisible by ``C``; ``N = L // C``). Compiles once, then — for the
        distributed path — stands up the ``DistributedWorker`` and allocates the
        shared-memory IO buffers (reused in place by every ``forward``/``measure``).
        ``K=1`` is the ring pipeline depth.
        """
        assert L % C == 0, f"L={L} not divisible by C={C}"
        self.P, self.L, self.C, self.dk, self.dv = P, L, C, dk, dv
        self.platform = platform
        self.device_ids = list(device_ids[:P])

        # Release any worker from a previous config first (it holds forked chip
        # children that leak if not closed).
        self.close()

        # Host-side constant tiles for the chunk kernels: within-chunk lower-triangular ones
        # (cumprod matmul), causal mask, two all-ones matrices to broadcast the per-chunk
        # gamma without illegal 1-column tiles, and a zero S-init / rank-0 boundary. Shared
        # so they can be passed to the reusable worker in place.
        self._tril = torch.tril(torch.ones(C, C, dtype=torch.float32)).share_memory_()
        self._mask = torch.tril(torch.ones(C, C, dtype=torch.float32)).share_memory_()
        self._ones_cc = torch.ones(C, C, dtype=torch.float32).share_memory_()
        self._ones_cdv = torch.ones(C, dv, dtype=torch.float32).share_memory_()
        self._zero = torch.zeros(dk, dv, dtype=torch.float32).share_memory_()

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        program = build_fused_forward_program(L, C, dk, dv, 1, P)
        dist_cfg = DistributedConfig(device_ids=self.device_ids, num_sub_workers=0)
        self._compiled = ir.compile(program, platform=platform, distributed_config=dist_cfg)

        # Prepare-once: for the distributed path, allocate the shared IO buffers BEFORE
        # prepare() forks the chip workers, and reuse them in place. (P=1 that compiles
        # non-distributed has no prepare -> forward()/measure() fall back to per-call.)
        self._rt = None
        if hasattr(self._compiled, "prepare"):
            self._h_Q = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._h_K = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._h_V = torch.zeros((P, L, dv), dtype=torch.float32).share_memory_()
            self._h_A = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._h_g = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
            self._h_O = torch.zeros((P, L, dv), dtype=torch.float32).share_memory_()
            self._rt = self._compiled.prepare()

    def _stage_inputs(self, Q, K, V, A):
        """Copy the per-forward inputs (and host-side gamma) into the shared buffers."""
        gammas = A.prod(dim=1).reshape(self.P, self.dk, 1)
        self._h_Q.copy_(Q)
        self._h_K.copy_(K)
        self._h_V.copy_(V)
        self._h_A.copy_(A)
        self._h_g.copy_(gammas)

    def _dispatch(self):
        """Run one fused-forward dispatch on the prepared worker (inputs already staged)."""
        self._h_O.zero_()
        self._rt(self._h_Q, self._h_K, self._h_V, self._h_A, self._h_g,
                 self._tril, self._mask, self._ones_cc, self._ones_cdv, self._zero, self._h_O)

    def forward(self, Q, K, V, A):
        """ZeCO forward; args/return as in :meth:`gla.common.ZeCoImpl.forward`."""
        assert self._compiled is not None, "call build() first"
        if self._rt is not None:
            self._stage_inputs(Q, K, V, A)
            self._dispatch()
            return self._h_O.clone()
        # P=1 non-distributed fallback (rare): per-call compiled run.
        gammas = A.prod(dim=1).reshape(self.P, self.dk, 1)
        O = torch.zeros((self.P, self.L, self.dv), dtype=torch.float32)
        run_fused_forward(
            self._compiled, Q, K, V, A, gammas,
            self._tril, self._mask, self._ones_cc, self._ones_cdv, self._zero, O,
            platform=self.platform, device_ids=self.device_ids,
        )
        return O

    def measure(self, Q, K, V, A, n_iters):
        """Steady-state per-forward latency: prepare is already done, so time only the
        repeated dispatch on the reusable worker. Falls back to the default per-call
        timing for a non-distributed P=1 config."""
        if self._rt is None:
            return super().measure(Q, K, V, A, n_iters)
        self._stage_inputs(Q, K, V, A)
        samples: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self._dispatch()
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def close(self):
        if self._rt is not None:
            self._rt.close()
            self._rt = None
