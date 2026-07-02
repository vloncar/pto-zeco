"""Shared building blocks for the ZeCO / Gated-Linear-Attention (GLA) operator.

ZeCO is sequence parallelism for GLA. AllScan (see the sibling :mod:`allscan`
package) is its *communication* core; this module adds the *compute* around it.

Data-dependent decay GLA, single head. Per token ``t`` (row vectors ``q_t, k_t``
of size ``dk``, ``v_t`` of size ``dv``, per-key-dim gate ``a_t`` in ``(0, 1)``)::

    S_t = diag(a_t) @ S_{t-1} + k_t^T v_t        # state  S in R^{dk x dv}
    o_t = q_t @ S_t                               # output o_t in R^{dv}

:func:`expected_gla` is the sequential golden. ZeCO computes the same thing in
parallel over ``P`` devices, each holding a contiguous ``L``-token slice split
into ``N = L // C`` chunks, by (A) a local chunk-wise scan, (B) an intra-chunk
masked attention that overlaps communication, an AllScan of the per-device
boundary state/decay, then (C) an output reconstruction. The chunk math lives in
:func:`gla_chunk_scan` / :func:`gla_reconstruct` so every torch-level backend
(in-process and distributed) shares one implementation.

Mapping to AllScan (``out[p] = S_local[p] + gamma[p] * out[p-1]``):

    S_local[p] = S_total_p   (device p's end-of-slice local state, ``[dk, dv]``)
    gamma[p]   = g_total_p   (device p's total decay over its slice, ``[dk, 1]``)
    out[p-1]   = S_recv_p    (the global boundary state entering device p)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


# ---------------------------------------------------------------------------
# Sequential reference (golden) + input generation
# ---------------------------------------------------------------------------

def expected_gla(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, A: torch.Tensor
) -> torch.Tensor:
    """Sequential recurrent GLA over the full sequence — the golden for ZeCO.

    Args:
        Q: Queries, shape ``[T, dk]``.
        K: Keys, shape ``[T, dk]``.
        V: Values, shape ``[T, dv]``.
        A: Per-token per-key-dim decay gates in ``(0, 1)``, shape ``[T, dk]``.

    Returns:
        Outputs ``O``, shape ``[T, dv]``, where ``O[t] = q_t @ S_t`` and
        ``S_t = diag(a_t) S_{t-1} + k_t^T v_t``.
    """
    T, dk = Q.shape
    dv = V.shape[1]
    S = torch.zeros((dk, dv), dtype=Q.dtype)
    O = torch.zeros((T, dv), dtype=Q.dtype)
    for t in range(T):
        S = A[t].unsqueeze(1) * S + torch.outer(K[t], V[t])
        O[t] = Q[t] @ S
    return O


def make_gla_inputs(
    P: int,
    L: int,
    dk: int,
    dv: int,
    seed: int = 42,
    decay_lo: float = 0.9,
    decay_hi: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic rank-major GLA inputs shared across implementations.

    Decay gates are drawn in ``(decay_lo, decay_hi)`` (default a gentle
    ``(0.9, 1.0)``) so cumulative products over ``L`` (and ``P*L``) tokens stay
    well inside FP32 range — the chunk math divides by within-chunk cumulative
    decay, so severe underflow would spoil the golden comparison.

    Args:
        P: Number of ranks/devices (the leading, rank-major axis).
        L: Tokens per device (must be divisible by the chunk size ``C``).
        dk: Key/query dimension.
        dv: Value dimension.
        seed: RNG seed for reproducible inputs.
        decay_lo: Lower bound of the decay-gate range.
        decay_hi: Upper bound of the decay-gate range.

    Returns:
        ``(Q[P,L,dk], K[P,L,dk], V[P,L,dv], A[P,L,dk])``.
    """
    torch.manual_seed(seed)
    Q = torch.randn((P, L, dk), dtype=torch.float32)
    K = torch.randn((P, L, dk), dtype=torch.float32)
    V = torch.randn((P, L, dv), dtype=torch.float32)
    A = decay_lo + (decay_hi - decay_lo) * torch.sigmoid(torch.randn((P, L, dk)))
    return Q, K, V, A


def flatten_seq(X: torch.Tensor) -> torch.Tensor:
    """Flatten a rank-major ``[P, L, d]`` tensor to the full sequence ``[P*L, d]``.

    Args:
        X: Rank-major tensor, shape ``[P, L, d]`` (device ``p`` owns ``X[p]``,
            covering the contiguous token range ``[p*L, (p+1)*L)``).

    Returns:
        The concatenated sequence, shape ``[P*L, d]``.
    """
    P, L, d = X.shape
    return X.reshape(P * L, d)


# ---------------------------------------------------------------------------
# Per-device chunk-wise GLA (shared by every torch-level backend)
# ---------------------------------------------------------------------------

def gla_chunk_scan(
    Qp: torch.Tensor, Kp: torch.Tensor, Vp: torch.Tensor, Ap: torch.Tensor, C: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """One device's local chunk-wise GLA scan (Stage A + intra-chunk of Stage B).

    Splits the device's ``L`` tokens into ``N = L // C`` chunks and, for each
    chunk ``n``, computes the within-chunk cumulative decay ``b_t = prod_{j<=t}
    a_j`` (inclusive), the chunk total decay ``gamma = b_{C-1}``, the intra-chunk
    masked-attention output, and advances the local running state / cumulative
    decay. Nothing here needs communication.

    Args:
        Qp: This device's queries, shape ``[L, dk]``.
        Kp: This device's keys, shape ``[L, dk]``.
        Vp: This device's values, shape ``[L, dv]``.
        Ap: This device's decay gates in ``(0, 1)``, shape ``[L, dk]``.
        C: Chunk size (``L`` must be divisible by ``C``).

    Returns:
        ``(S_prev, c_prev, O_intra, S_total, g_total)`` where:
          * ``S_prev[n]`` ``[dk, dv]`` — local state *before* chunk ``n``
            (``S_prev[0]`` is zero);
          * ``c_prev[n]`` ``[dk]`` — cumulative decay of chunks ``0..n-1``
            (``c_prev[0]`` is one) that scales the cross-device prefix in Stage C;
          * ``O_intra`` ``[L, dv]`` — the intra-chunk (diagonal) attention output;
          * ``S_total`` ``[dk, dv]`` — the device's end-of-slice local state
            (feeds AllScan as ``S_local``);
          * ``g_total`` ``[dk]`` — the device's total decay over its slice
            (feeds AllScan as ``gamma``).
    """
    L, dk = Qp.shape
    dv = Vp.shape[1]
    assert L % C == 0, f"L={L} not divisible by C={C}"
    N = L // C

    S = torch.zeros((dk, dv), dtype=Qp.dtype)
    c = torch.ones((dk,), dtype=Qp.dtype)
    S_prev: list[torch.Tensor] = []
    c_prev: list[torch.Tensor] = []
    O_intra = torch.zeros((L, dv), dtype=Qp.dtype)
    mask = torch.tril(torch.ones((C, C), dtype=Qp.dtype))

    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        q, k, v, a = Qp[lo:hi], Kp[lo:hi], Vp[lo:hi], Ap[lo:hi]  # [C, ...]
        b = torch.cumprod(a, dim=0)          # [C, dk] inclusive within-chunk decay
        gamma = b[-1]                         # [dk] total chunk decay

        S_prev.append(S.clone())
        c_prev.append(c.clone())

        # --- intra-chunk masked attention: scores[t,s] = (q_t*b_t) . (k_s/b_s) ---
        Qt = q * b                            # [C, dk]
        Kbar_intra = k / b                     # [C, dk]
        scores = (Qt @ Kbar_intra.t()) * mask  # [C, C], causal (s <= t)
        O_intra[lo:hi] = scores @ v            # [C, dv]

        # --- state update: token s decays by gamma/b_s into end-of-chunk state ---
        Kbar_state = k * (gamma / b)          # [C, dk]
        S = gamma.unsqueeze(1) * S + Kbar_state.t() @ v
        c = c * gamma

    return S_prev, c_prev, O_intra, S, c


def gla_reconstruct(
    Qp: torch.Tensor,
    Ap: torch.Tensor,
    C: int,
    S_prev: list[torch.Tensor],
    c_prev: list[torch.Tensor],
    S_recv: torch.Tensor,
    O_intra: torch.Tensor,
) -> torch.Tensor:
    """One device's output reconstruction (Stage C), given the boundary state.

    Combines the inter-chunk history — local state plus the decayed cross-device
    prefix ``S_recv`` — with the cached intra-chunk output::

        O_inter[n] = (Q[n] * b) @ (S_prev[n] + diag(c_prev[n]) @ S_recv)
        O[n]       = O_inter[n] + O_intra[n]

    Args:
        Qp: This device's queries, shape ``[L, dk]``.
        Ap: This device's decay gates, shape ``[L, dk]`` (for the within-chunk
            cumulative decay ``b``).
        C: Chunk size.
        S_prev: Per-chunk local pre-states from :func:`gla_chunk_scan`.
        c_prev: Per-chunk cumulative decays from :func:`gla_chunk_scan`.
        S_recv: Global boundary state entering this device, shape ``[dk, dv]``
            (zeros for rank 0); this is AllScan's ``out[p-1]``.
        O_intra: Cached intra-chunk output from :func:`gla_chunk_scan`,
            shape ``[L, dv]``.

    Returns:
        This device's output ``O``, shape ``[L, dv]``.
    """
    L, dk = Qp.shape
    dv = O_intra.shape[1]
    N = L // C
    O = torch.zeros((L, dv), dtype=Qp.dtype)
    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        q, a = Qp[lo:hi], Ap[lo:hi]
        b = torch.cumprod(a, dim=0)
        Qt = q * b
        hist = S_prev[n] + c_prev[n].unsqueeze(1) * S_recv   # [dk, dv]
        O[lo:hi] = Qt @ hist + O_intra[lo:hi]
    return O


# ---------------------------------------------------------------------------
# Implementation interface
# ---------------------------------------------------------------------------

class ZeCoImpl(ABC):
    """Interface for a single ZeCO / GLA operator backend.

    Lifecycle per config::

        impl.build(P, L, C, dk, dv, device_ids, platform)   # once per config
        O = impl.forward(Q, K, V, A)                          # many times
        impl.close()                                          # cleanup

    Tensors are rank-major ``[P, L, dk/dv]`` (device ``p`` owns row ``p``),
    matching the ``[P, ...]`` convention of :class:`allscan.common.AllscanImpl`.
    """

    #: Short name used in registries and result tables.
    name: str

    @abstractmethod
    def build(
        self,
        P: int,
        L: int,
        C: int,
        dk: int,
        dv: int,
        device_ids: list[int],
        platform: str,
    ) -> None:
        """Compile / initialise the backend. Called once per config.

        Args:
            P: Number of ranks/devices.
            L: Tokens per device.
            C: Chunk size (``L`` must be divisible by ``C``).
            dk: Key/query dimension.
            dv: Value dimension.
            device_ids: Physical device ids available to this run.
            platform: Target backend, e.g. ``"a2a3"`` or ``"a2a3sim"``.
        """

    @abstractmethod
    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Run the ZeCO forward pass and return the outputs.

        Args:
            Q: Queries, shape ``[P, L, dk]``.
            K: Keys, shape ``[P, L, dk]``.
            V: Values, shape ``[P, L, dv]``.
            A: Decay gates in ``(0, 1)``, shape ``[P, L, dk]``.

        Returns:
            Outputs ``O``, shape ``[P, L, dv]`` (rank-major).
        """

    def close(self) -> None:
        """Release resources (override if needed)."""
