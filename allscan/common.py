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
    """Pure sequential AllScan, used by every test/benchmark for verification.

    Args:
        S_locals: Per-rank local state, shape ``[P, dk, dv]`` (rank ``p`` owns
            ``S_locals[p]``).
        gammas: Per-rank decay factors, shape ``[P, dk, 1]``; ``gammas[p]``
            broadcasts across the ``dv`` columns of ``out[p-1]``.

    Returns:
        The scan output ``out``, shape ``[P, dk, dv]``.
    """
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

    The same seed is used by every backend so results are directly comparable.

    Args:
        P: Number of ranks (ring participants).
        dk: Key/row dimension of the per-rank state.
        dv: Value/column dimension of the per-rank state.
        seed: RNG seed for the (reproducible) random inputs.

    Returns:
        ``(S_locals[P,dk,dv], gammas[P,dk,1], outputs[P,dk,dv])`` — random state
        and decays, plus a zeroed output buffer for the backend to fill.
    """
    torch.manual_seed(seed)
    S_locals = torch.rand((P, dk, dv), dtype=torch.float32)
    gammas = torch.rand((P, dk, 1), dtype=torch.float32)
    outputs = torch.zeros((P, dk, dv), dtype=torch.float32)
    return S_locals, gammas, outputs


def expected_allscan_backward(
    gammas: torch.Tensor, outs: torch.Tensor, g_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure sequential AllScan backward, the golden for every backend.

    Given the forward outputs ``outs`` (``out[p]``, retained from the forward
    pass) and the upstream gradient ``g_out[p] = dL/dout[p]``, the adjoint
    ``d[p] = dL/dout[p]`` (total, including the downstream chain) is a *reverse*
    scan with ``gamma`` shifted by one::

        d[P-1] = g_out[P-1]
        d[p]   = g_out[p] + gamma[p+1] * d[p+1]        (p = P-2 .. 0)

    from which the parameter gradients are local::

        dS_local[p] = d[p]                              (all p)
        dgamma[p]   = rowsum_dv( d[p] * out[p-1] )      (p = 1..P-1) -> [dk,1]
        dgamma[0]   = 0                                 (gamma[0] is unused)

    Args:
        gammas: Per-rank decay factors, shape ``[P, dk, 1]`` (same as forward).
        outs: Retained forward outputs ``out[p]``, shape ``[P, dk, dv]``.
        g_out: Upstream gradient ``dL/dout[p]``, shape ``[P, dk, dv]``.

    Returns:
        ``(dS[P,dk,dv], dgamma[P,dk,1])`` — gradients w.r.t. ``S_local`` and
        ``gamma``.
    """
    P = g_out.shape[0]
    dS = torch.zeros_like(g_out)
    dS[P - 1] = g_out[P - 1]
    for p in range(P - 2, -1, -1):
        dS[p] = g_out[p] + gammas[p + 1] * dS[p + 1]
    dgamma = torch.zeros_like(gammas)
    for p in range(1, P):
        dgamma[p] = (dS[p] * outs[p - 1]).sum(dim=1, keepdim=True)
    return dS, dgamma


def make_grad_inputs(P: int, dk: int, dv: int, seed: int = 1234) -> torch.Tensor:
    """Deterministic upstream gradient ``g_out[P,dk,dv]`` for backward tests.

    Uses a different default seed from :func:`make_inputs` so the upstream grad
    is not accidentally correlated with the forward inputs.

    Args:
        P: Number of ranks.
        dk: Key/row dimension.
        dv: Value/column dimension.
        seed: RNG seed (distinct from ``make_inputs`` by default).

    Returns:
        Random upstream gradient ``g_out``, shape ``[P, dk, dv]``.
    """
    torch.manual_seed(seed)
    return torch.rand((P, dk, dv), dtype=torch.float32)


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
        """Compile or initialise the implementation. Called once per config.

        Args:
            dk: Key/row dimension of the per-rank ``[dk, dv]`` state.
            dv: Value/column dimension of the state.
            K: Pipeline depth — the scan is blocked into ``K`` chunks of
                ``dk // K`` rows (``dk`` must be divisible by ``K``).
            P: Number of ranks; the first ``P`` of ``device_ids`` are used.
            device_ids: Physical device ids available to this run.
            platform: Target backend, e.g. ``"a2a3"`` (hardware) or
                ``"a2a3sim"`` (simulator).
        """

    @abstractmethod
    def run(
        self,
        S_locals: torch.Tensor,
        gammas: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        """Execute the AllScan collective synchronously, writing into ``outputs``.

        Args:
            S_locals: Per-rank local state, shape ``[P, dk, dv]``.
            gammas: Per-rank decay factors, shape ``[P, dk, 1]``.
            outputs: Destination for the scan result, shape ``[P, dk, dv]``;
                overwritten in place.
        """

    def run_backward(
        self,
        g_out: torch.Tensor,
        gammas: torch.Tensor,
        outs: torch.Tensor,
        dS: torch.Tensor,
        dgamma: torch.Tensor,
    ) -> None:
        """Execute the AllScan backward pass synchronously.

        Writes the input gradients (see :func:`expected_allscan_backward` for the
        exact math). Optional: backends implement it as they gain a backward
        kernel.

        Args:
            g_out: Upstream gradient ``dL/dout[p]``, shape ``[P, dk, dv]``.
            gammas: Per-rank decay factors, shape ``[P, dk, 1]`` (same as forward).
            outs: Retained forward outputs ``out[p]``, shape ``[P, dk, dv]``.
            dS: Destination for ``dL/dS_local``, shape ``[P, dk, dv]``; written
                in place.
            dgamma: Destination for ``dL/dgamma``, shape ``[P, dk, 1]``; written
                in place (``dgamma[0]`` is 0).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_backward"
        )

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

        Args:
            S_locals: Per-rank local state, shape ``[P, dk, dv]``.
            gammas: Per-rank decay factors, shape ``[P, dk, 1]``.
            outputs: Scratch output buffer, shape ``[P, dk, dv]`` (reused each
                sample).
            n_iters: Number of latency samples to collect.

        Returns:
            A list of ``n_iters`` per-iteration latencies in milliseconds.
        """
        samples: list[float] = []
        for _ in range(n_iters):
            outputs.zero_()
            t0 = time.perf_counter()
            self.run(S_locals, gammas, outputs)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def measure_backward(
        self,
        g_out: torch.Tensor,
        gammas: torch.Tensor,
        outs: torch.Tensor,
        dS: torch.Tensor,
        dgamma: torch.Tensor,
        n_iters: int,
    ) -> list[float]:
        """Return ``n_iters`` per-iteration backward latency samples (ms).

        Mirrors :meth:`measure` for the backward pass. Default times
        :meth:`run_backward` once per sample; backends dominated by fixed
        per-call orchestration override this to amortize it across a batch (and
        set :attr:`amortized_timing`).

        Args:
            g_out: Upstream gradient, shape ``[P, dk, dv]``.
            gammas: Per-rank decay factors, shape ``[P, dk, 1]``.
            outs: Retained forward outputs, shape ``[P, dk, dv]``.
            dS: Scratch buffer for ``dL/dS_local``, shape ``[P, dk, dv]``.
            dgamma: Scratch buffer for ``dL/dgamma``, shape ``[P, dk, 1]``.
            n_iters: Number of latency samples to collect.

        Returns:
            A list of ``n_iters`` per-iteration backward latencies in milliseconds.
        """
        samples: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.run_backward(g_out, gammas, outs, dS, dgamma)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def close(self) -> None:
        """Release resources (override if needed)."""
