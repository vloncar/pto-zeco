"""PyPTO DSL ZeCO / GLA forward — benchmark/test adapter.

:class:`PytoZeCo` composes the two ``@pl.jit`` GLA kernels (:mod:`.program`) with
the existing PyPTO AllScan for the cross-device boundary hand-off, exactly as
:class:`gla.implementations.torch_ref.TorchZeCo` composes ``TorchAllscan``.

Per ``forward`` (all on-device via pypto):

  1. **stage1** on each rank ``r`` -> ``S_total[r]`` (local-only end-of-slice
     state) and ``g[r]`` (device total decay).
  2. **AllScan** of ``(S_total, g)`` -> ``out``; rank ``r`` receives the boundary
     state ``S_recv[r] = out[r-1]`` (zero for rank 0).
  3. **stage2** on each rank ``r`` -> ``O[r]``, adding the ``Qt @ S_recv[r]``
     cross-device term to the local GLA.

The single-device ``@pl.jit`` dispatches and the AllScan ``DistributedWorker``
are strictly sequenced (stage1 dispatches, then AllScan build/run/close, then
stage2 dispatches) so no ``@pl.jit`` runtime and the AllScan worker ever hold the
same devices at once.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gla.common import ZeCoImpl  # noqa: E402
from gla.implementations.pypto.program import make_zeco_jits  # noqa: E402


class PytoZeCo(ZeCoImpl):
    """PyPTO ZeCO forward: two ``@pl.jit`` GLA stages + PyPTO AllScan boundary."""

    name = "pypto"

    def __init__(self) -> None:
        self._stage1 = None

    def build(self, P, L, C, dk, dv, device_ids, platform):
        """Generate the shape-specialised GLA kernels and stash config.

        Args as in :meth:`gla.common.ZeCoImpl.build`. ``C`` is ignored — the
        pypto backend uses the quadratic whole-device form (``C = L``). The
        AllScan is built per ``forward`` (so it never coexists with the jit
        dispatches), so nothing HCCL is stood up here.
        """
        self.P, self.L, self.dk, self.dv = P, L, dk, dv
        self.platform = platform
        self.device_ids = list(device_ids[:P])
        self._stage1, self._stage2 = make_zeco_jits(L, dk, dv)
        # Host-side constant helpers: lower-triangular ones (for the cumprod
        # matmul) and the causal mask; identical [L, L] for this formulation.
        self._tril = torch.tril(torch.ones(L, L, dtype=torch.float32))
        self._mask = torch.tril(torch.ones(L, L, dtype=torch.float32))

    def _cfg(self, rank: int):
        from pypto.runtime.runner import RunConfig

        return RunConfig(platform=self.platform, device_id=self.device_ids[rank])

    def forward(self, Q, K, V, A):
        """ZeCO forward; args/return as in :meth:`gla.common.ZeCoImpl.forward`."""
        assert self._stage1 is not None, "call build() first"
        P, L, dk, dv = self.P, self.L, self.dk, self.dv

        # --- stage 1: local end-of-slice state per rank (on device) ---
        S_tot = torch.zeros((P, dk, dv), dtype=torch.float32)
        for r in range(P):
            s_r = torch.zeros((dk, dv), dtype=torch.float32)
            self._stage1(Q[r], K[r], V[r], A[r], self._tril, s_r, config=self._cfg(r))
            S_tot[r] = s_r
        # device total decay gamma = prod over tokens of A (== b[L-1]); host-side.
        gammas = A.prod(dim=1).reshape(P, dk, 1)

        # --- boundary exchange via PyPTO AllScan (out[r-1] is rank r's prefix) ---
        S_recv = torch.zeros((P, dk, dv), dtype=torch.float32)
        if P > 1:
            from allscan.implementations.pypto.impl import PytoAllscan

            outs = torch.zeros((P, dk, dv), dtype=torch.float32)
            allscan = PytoAllscan()
            allscan.build(dk, dv, 1, P, self.device_ids, self.platform)
            allscan.run(S_tot, gammas, outs)
            allscan.close()
            S_recv[1:] = outs[:-1]

        # --- stage 2: reconstruct each rank's output ---
        O = torch.zeros((P, L, dv), dtype=torch.float32)
        for r in range(P):
            o_r = torch.zeros((L, dv), dtype=torch.float32)
            self._stage2(Q[r], K[r], V[r], A[r], self._tril, self._mask, S_recv[r], o_r,
                         config=self._cfg(r))
            O[r] = o_r
        return O
