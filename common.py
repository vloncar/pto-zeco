"""Shared building blocks for the AllScan implementations.

Every AllScan backend (torch reference, PyPTO DSL, direct PTO runtime "simpler")
plugs into the :class:`AllscanImpl` interface so the benchmark and the tests can
drive them uniformly.

The AllScan collective computes, over P ranks arranged in a ring::

    out[0] = S_local[0]
    out[p] = S_local[p] + gamma[p] * out[p-1]      (p = 1 .. P-1)

where ``gamma[p]`` is ``[dk, 1]`` and broadcasts across the ``dv`` columns of the
``[dk, dv]`` state. Work is pipelined over ``K`` blocks of ``dk // K`` rows.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import torch


# ---------------------------------------------------------------------------
# Sequential reference + shared input generation
# ---------------------------------------------------------------------------

def expected_allscan(S_locals: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """Pure sequential AllScan, used by every test/benchmark for verification."""
    P = S_locals.shape[0]
    out = torch.zeros_like(S_locals)
    out[0] = S_locals[0]
    for p in range(1, P):
        out[p] = S_locals[p] + gammas[p] * out[p - 1]
    return out


def make_inputs(
    P: int, dk: int, dv: int, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic inputs shared across implementations.

    Returns ``(S_locals[P,dk,dv], gammas[P,dk,1], outputs[P,dk,dv])`` with the
    same seed every backend uses, so results are directly comparable.
    """
    torch.manual_seed(seed)
    S_locals = torch.rand((P, dk, dv), dtype=torch.float32)
    gammas = torch.rand((P, dk, 1), dtype=torch.float32)
    outputs = torch.zeros((P, dk, dv), dtype=torch.float32)
    return S_locals, gammas, outputs


# ---------------------------------------------------------------------------
# Implementation interface
# ---------------------------------------------------------------------------

class AllscanImpl(ABC):
    """Interface for a single AllScan backend.

    Lifecycle per benchmark/test config::

        impl.build(dk, dv, K, P, device_ids, platform)   # once per config
        impl.run(S_locals, gammas, outputs)               # many times
        impl.close()                                      # cleanup
    """

    #: Short name used in ``--impl`` and result tables.
    name: str

    @abstractmethod
    def build(
        self,
        dk: int,
        dv: int,
        K: int,
        P: int,
        device_ids: list[int],
        platform: str,
    ) -> None:
        """Compile or initialise the implementation. Called once per config."""

    @abstractmethod
    def run(
        self,
        S_locals: torch.Tensor,
        gammas: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        """Execute the AllScan collective synchronously, writing into ``outputs``."""

    #: Whether :meth:`measure` amortizes one-time per-call setup (e.g. comm-domain
    #: allocation) across the timed iterations. False here means the timed numbers
    #: include full per-call orchestration overhead.
    amortized_timing: bool = False

    def measure(
        self,
        S_locals: torch.Tensor,
        gammas: torch.Tensor,
        outputs: torch.Tensor,
        n_iters: int,
    ) -> list[float]:
        """Return ``n_iters`` per-iteration latency samples in milliseconds.

        Default implementation times ``run`` once per sample, so the numbers
        carry the full per-call overhead. Backends whose per-call cost is
        dominated by fixed orchestration setup (e.g. comm-domain alloc/free)
        override this to amortize that setup across a batch and report the
        marginal kernel/comm time (set :attr:`amortized_timing` to True).
        """
        samples: list[float] = []
        for _ in range(n_iters):
            outputs.zero_()
            t0 = time.perf_counter()
            self.run(S_locals, gammas, outputs)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def close(self) -> None:
        """Release resources (override if needed)."""
