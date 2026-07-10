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

**History (why this used to be a hybrid).** Two framework limitations previously forced a
``dist(stage1+ring) -> close -> @pl.jit stage2`` hybrid: (a) a ``@pl.jit`` dispatch coexisting
with ``DistributedWorker.prepare()`` on one device segfaults/hangs, and (b) the wide matmul-DAG
``stage2`` kernel hung the card (AICore ``507018``) as a distributed chip kernel. Both the
sim-scheduler and HW dist-chip faults on that kernel class were fixed upstream, so ``stage2``
now runs as a distributed chip kernel on both a2a3sim and a2a3 — full single-program fusion,
verified on hardware at P=1/2/4. Since there is no ``@pl.jit`` at all, limitation (a) no longer
applies either.

``P == 1`` is a native path (single rank, no boundary, ``S_recv = 0``) built by a separate
factory (:func:`gla.implementations.pypto.fused_program._build_p1_forward_program`) — see that
module for why P=1 and P>1 must be separate factories.

Runs on both a2a3sim and a2a3 hardware. Every distributed / HCCL run must set
``LD_PRELOAD=<cann>/lib64/libhccl.so`` or the rootinfo handshake hangs.
"""

from __future__ import annotations

import os
import sys

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
    """PyPTO ZeCO forward: the entire forward as one fully-fused distributed program."""

    name = "pypto"

    def __init__(self) -> None:
        self._compiled = None

    def build(self, P, L, C, dk, dv, device_ids, platform):
        """Compile the fully-fused ZeCO forward program for this config.

        Args as in :meth:`gla.common.ZeCoImpl.build`. ``C`` is the real chunk size
        (``L`` must be divisible by ``C``; ``N = L // C`` chunks). The program is compiled
        once here; each :meth:`forward` prepares a fresh ``DistributedWorker``, runs one pass,
        and closes it (via :func:`run_fused_forward`). ``K=1`` is the ring pipeline depth.
        """
        assert L % C == 0, f"L={L} not divisible by C={C}"
        self.P, self.L, self.C, self.dk, self.dv = P, L, C, dk, dv
        self.platform = platform
        self.device_ids = list(device_ids[:P])

        # Host-side constant tiles for the chunk kernels: within-chunk lower-triangular ones
        # (cumprod matmul), causal mask, and two all-ones matrices used to broadcast the
        # per-chunk gamma without illegal 1-column tiles. Plus a zero S-init / rank-0 boundary.
        self._tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
        self._mask = torch.tril(torch.ones(C, C, dtype=torch.float32))
        self._ones_cc = torch.ones(C, C, dtype=torch.float32)
        self._ones_cdv = torch.ones(C, dv, dtype=torch.float32)
        self._zero = torch.zeros(dk, dv, dtype=torch.float32)

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        program = build_fused_forward_program(L, C, dk, dv, 1, P)
        dist_cfg = DistributedConfig(device_ids=self.device_ids, num_sub_workers=0)
        self._compiled = ir.compile(program, platform=platform, distributed_config=dist_cfg)

    def forward(self, Q, K, V, A):
        """ZeCO forward; args/return as in :meth:`gla.common.ZeCoImpl.forward`."""
        assert self._compiled is not None, "call build() first"
        P, L, dv = self.P, self.L, self.dv

        # device total decay gamma = prod over tokens of A (== b[L-1]); host-side.
        gammas = A.prod(dim=1).reshape(P, self.dk, 1)
        O = torch.zeros((P, L, dv), dtype=torch.float32)
        run_fused_forward(
            self._compiled, Q, K, V, A, gammas,
            self._tril, self._mask, self._ones_cc, self._ones_cdv, self._zero, O,
            platform=self.platform, device_ids=self.device_ids,
        )
        return O
