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

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import AllscanImpl  # noqa: E402
from implementations.pypto.program import build_allscan_program  # noqa: E402


class PytoAllscan(AllscanImpl):
    """PyPTO DSL-compiled AllScan, dispatched on a reusable DistributedWorker."""

    name = "pypto"

    def __init__(self) -> None:
        self._rt = None

    def build(self, dk, dv, K, P, device_ids, platform):
        # Release any DistributedWorker from a previous config first: it holds a
        # forked L3 Worker whose chip children leak if not closed.
        self.close()

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        self.P = P
        program = build_allscan_program(dk, dv, K, P)
        compiled = ir.compile(
            program,
            platform=platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:P],
                num_sub_workers=0,
            ),
        )

        # Shared-memory IO buffers must exist before prepare() forks the chip
        # workers; they are reused in place across every dispatch.
        self._host_s = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._host_g = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
        self._host_out = torch.zeros((P, dk, dv), dtype=torch.float32).share_memory_()
        self._rt = compiled.prepare()

    def run(self, S_locals, gammas, outputs):
        assert self._rt is not None, "call build() first"
        self._host_s.copy_(S_locals)
        self._host_g.copy_(gammas)
        self._host_out.zero_()
        self._rt(self._host_s, self._host_g, self._host_out)
        outputs.copy_(self._host_out)

    def close(self):
        if self._rt is not None:
            self._rt.close()
            self._rt = None
