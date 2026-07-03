"""PyPTO DSL ZeCO / GLA forward — benchmark/test adapter.

:class:`PytoZeCo` composes the chunk-recurrent GLA compute with the AllScan
boundary hand-off, exactly as :class:`gla.implementations.torch_ref.TorchZeCo`
composes ``TorchAllscan``. The composition is **hybrid**:

  1. **stage1 + AllScan ring**, *fused in one distributed ``@pl.program``*
     (:mod:`.dist_program`). Per rank ``r`` the host orchestrator computes the
     local end-of-slice state ``S_total[r]`` (chunk-recurrent scan from ``S=0``,
     as a distributed InCore chip kernel) and then rings it to
     ``out[p] = S_local[p] + gamma[p]*out[p-1]``. ``gamma`` is ``A.prod`` over
     tokens (host-side). The caller shifts ``S_recv[r] = out[r-1]`` (zero for r=0).
  2. **stage2** on each rank ``r`` -> ``O[r]`` via ``@pl.jit`` (:mod:`.program`),
     the same chunk recurrence **initialised from ``S_recv[r]``**.

**Why hybrid, not a single fully-fused program** (both learned on a2a3 hardware):
  * Fusing stage1 into the distributed program removes the only ``@pl.jit``
    dispatch that would otherwise run *before* ``DistributedWorker.prepare()``
    forks — the jit-dispatch-then-``prepare()`` coexistence segfaults at ``P>1``.
    Here the sole jit (stage2) runs *after* the worker ``close()``s, which is the
    safe ``prepare -> close -> jit`` order.
  * stage2 is a wide matmul-DAG kernel (loop-carry used as a matmul operand +
    intra-chunk attention) that **hangs as a distributed chip kernel** (AICore
    ``507018`` device-drain timeout). It only survives the ``@pl.jit`` CORE_GROUP
    path, so it must stay on jit.

At ``P == 1`` there is no boundary to exchange, so the fused distributed program
is skipped entirely (``S_recv = 0``) and only ``@pl.jit`` stage2 runs — this also
avoids a ``P=1`` loop-unroll codegen bug in the distributed ``device=r`` dispatch.

**Hardware-only:** the chunk-recurrent kernels deadlock the a2a3sim simulator
scheduler but run correctly on a2a3 hardware — see :mod:`.program`.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gla.common import ZeCoImpl  # noqa: E402
from gla.implementations.pypto.dist_program import build_stage1ring_program  # noqa: E402
from gla.implementations.pypto.program import make_zeco_jits  # noqa: E402


class PytoZeCo(ZeCoImpl):
    """PyPTO ZeCO forward: fused distributed stage1+AllScan-ring, then ``@pl.jit`` stage2."""

    name = "pypto"

    def __init__(self) -> None:
        self._stage2 = None

    def build(self, P, L, C, dk, dv, device_ids, platform):
        """Generate the shape-specialised chunk-recurrent GLA kernels and stash config.

        Args as in :meth:`gla.common.ZeCoImpl.build`. ``C`` is the real chunk size
        (``L`` must be divisible by ``C``; ``N = L // C`` chunks). Nothing HCCL is
        stood up here; the fused stage1+ring distributed program is compiled and
        prepared per ``forward`` (so its ``DistributedWorker`` never coexists with a
        ``@pl.jit`` dispatch on the same devices).

        Note: the chunk-recurrent InCore kernels run on **a2a3 hardware only** — their
        per-chunk body deadlocks the a2a3sim simulator scheduler (see
        :mod:`gla.implementations.pypto.program`).
        """
        assert L % C == 0, f"L={L} not divisible by C={C}"
        self.P, self.L, self.C, self.dk, self.dv = P, L, C, dk, dv
        self.platform = platform
        self.device_ids = list(device_ids[:P])
        # Only stage2 is a @pl.jit kernel; stage1 lives inside the distributed program.
        _stage1_unused, self._stage2 = make_zeco_jits(L, C, dk, dv)
        # Host-side constant tiles for the chunk kernels: within-chunk lower-triangular
        # ones (cumprod matmul), causal mask, and two all-ones matrices used to broadcast
        # the per-chunk gamma without illegal 1-column tiles. Plus a zero S-init.
        self._tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        self._mask = torch.tril(torch.ones(C, C, dtype=torch.float32))
        self._ones_cc = torch.ones(C, C, dtype=torch.float32)
        self._ones_cdv = torch.ones(C, dv, dtype=torch.float32)
        self._zero = torch.zeros(dk, dv, dtype=torch.float32)

    def _cfg(self, rank: int):
        from pypto.runtime.runner import RunConfig

        return RunConfig(platform=self.platform, device_id=self.device_ids[rank])

    def _stage1_ring(self, A, K, V, gammas):
        """Run the fused distributed stage1+AllScan-ring program (P>1) once.

        Compiles + prepares a fresh ``DistributedWorker`` on ``self.device_ids``,
        dispatches one host-orchestrated pass, then closes it (so the subsequent
        ``@pl.jit`` stage2 sees free devices). Returns the inclusive ring scan
        ``outputs`` ``[P, dk, dv]`` (the caller shifts it to ``S_recv``).
        """
        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        P, L, dk, dv = self.P, self.L, self.dk, self.dv
        program = build_stage1ring_program(L, self.C, dk, dv, 1, P)
        dist_cfg = DistributedConfig(device_ids=self.device_ids, num_sub_workers=0)
        compiled = ir.compile(program, platform=self.platform, distributed_config=dist_cfg)

        # Shared-memory IO buffers must exist before prepare() forks the chip workers.
        def sm(t):
            return t.clone().share_memory_()

        h_A, h_K, h_V, h_g = sm(A), sm(K), sm(V), sm(gammas)
        h_tril, h_occ = sm(self._tril), sm(self._ones_cc)
        h_ocdv, h_zero = sm(self._ones_cdv), sm(self._zero)
        h_out = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()

        rt = compiled.prepare()
        try:
            rt(h_A, h_K, h_V, h_g, h_tril, h_occ, h_ocdv, h_zero, h_out)
        finally:
            rt.close()
        return h_out.clone()

    def forward(self, Q, K, V, A):
        """ZeCO forward; args/return as in :meth:`gla.common.ZeCoImpl.forward`."""
        assert self._stage2 is not None, "call build() first"
        P, L, dk, dv = self.P, self.L, self.dk, self.dv

        # device total decay gamma = prod over tokens of A (== b[L-1]); host-side.
        gammas = A.prod(dim=1).reshape(P, dk, 1)

        # --- stage 1 + boundary exchange (fused distributed program, P>1 only) ---
        # S_recv[r] = out[r-1] (the global prefix state entering rank r); zero for r=0.
        S_recv = torch.zeros((P, dk, dv), dtype=torch.float32)
        if P > 1:
            ring_out = self._stage1_ring(A, K, V, gammas)
            S_recv[1:] = ring_out[:-1]

        # --- stage 2: reconstruct each rank's output (S_run initialised from S_recv) ---
        # @pl.jit, dispatched AFTER the distributed worker closed (safe order).
        O = torch.zeros((P, L, dv), dtype=torch.float32)
        for r in range(P):
            o_r = torch.zeros((L, dv), dtype=torch.float32)
            self._stage2(Q[r], K[r], V[r], A[r], self._tril, self._mask,
                         self._ones_cc, self._ones_cdv, S_recv[r], o_r, config=self._cfg(r))
            O[r] = o_r
        return O
