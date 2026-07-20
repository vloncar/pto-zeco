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

import time
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


def expected_gla_backward(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, A: torch.Tensor, dO: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic backward of :func:`expected_gla` — the golden for every ZeCO backend.

    Reverse of the forward recurrence ``S_t = diag(a_t) S_{t-1} + k_t^T v_t``,
    ``o_t = q_t S_t``.  Carrying the adjoint state ``dS`` (grad w.r.t. the running
    state) *backwards* over ``t``::

        dS_t   = q_t^T dO_t + diag(a_{t+1}) dS_{t+1}     (adjoint recurrence)
        dQ_t   = S_t  dO_t^T -> dQ_t = S_t @ dO_t
        dK_t   = dS_t V_t   (= dS_t @ v_t)
        dV_t   = K_t dS_t   (= k_t @ dS_t)
        dA_t   = rowsum_dv( dS_t * S_{t-1} )

    Independent of autograd; :func:`~gla.tests.test_gla_backward` cross-checks it
    against ``torch.autograd`` on :func:`expected_gla`.

    Args:
        Q, K, V, A: The forward inputs, shapes ``[T, dk]`` / ``[T, dk]`` /
            ``[T, dv]`` / ``[T, dk]``.
        dO: Upstream gradient of the outputs, shape ``[T, dv]``.

    Returns:
        ``(dQ, dK, dV, dA)`` matching the shapes of ``(Q, K, V, A)``.
    """
    T, dk = Q.shape
    dv = V.shape[1]

    # Re-run the forward, caching the state trajectory S_t (S_hist[t] == S_t;
    # S_{-1} == 0) needed for dQ_t (uses S_t) and dA_t (uses S_{t-1}).
    S = torch.zeros((dk, dv), dtype=Q.dtype)
    S_hist = torch.zeros((T, dk, dv), dtype=Q.dtype)
    for t in range(T):
        S = A[t].unsqueeze(1) * S + torch.outer(K[t], V[t])
        S_hist[t] = S

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dA = torch.zeros_like(A)

    dS = torch.zeros((dk, dv), dtype=Q.dtype)  # incoming diag(a_{t+1}) dS_{t+1}
    for t in range(T - 1, -1, -1):
        dS = dS + torch.outer(Q[t], dO[t])        # total adjoint of S_t
        dQ[t] = S_hist[t] @ dO[t]
        dK[t] = dS @ V[t]
        dV[t] = K[t] @ dS
        S_prev = S_hist[t - 1] if t > 0 else torch.zeros((dk, dv), dtype=Q.dtype)
        dA[t] = (dS * S_prev).sum(dim=1)
        dS = A[t].unsqueeze(1) * dS               # push through diag(a_t) to t-1
    return dQ, dK, dV, dA


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


def gla_chunk_backward(
    Qp: torch.Tensor,
    Kp: torch.Tensor,
    Vp: torch.Tensor,
    Ap: torch.Tensor,
    C: int,
    S_recv: torch.Tensor,
    dO: torch.Tensor,
    dS_total: torch.Tensor,
    dg_total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One device's explicit chunk-parallel GLA backward (no autograd).

    The hand-derived reverse of :func:`gla_chunk_scan` + :func:`gla_reconstruct`,
    written op-for-op so the simpler / pypto **kernels** can implement it directly
    (autograd is unavailable on-device). It is the local half of the SP backward:
    the cross-device coupling enters as ``dS_total`` / ``dg_total`` (the grads on
    this device's boundary outputs, produced by the AllScan-backward reverse ring)
    and leaves as ``dS_recv`` (the grad on the received boundary state, fed into
    that ring). For ``P == 1`` (``S_recv = dS_total = dg_total = 0``) it reduces to
    :func:`expected_gla_backward` on the device's own tokens.

    The blocks below are grouped by the kernel that will own them:

    * **grad_o** (output-stage, per chunk, embarrassingly parallel): reconstruct +
      intra adjoints — ``dQt``, ``dH_n = Qt^T dO``, ``dscores``, and the ``dS_recv``
      accumulation.
    * **grad_h** (state-stage, reverse chunk recurrence): carries the state adjoint
      ``dSloc`` / decay adjoint ``dcvec`` backwards across chunks, producing
      ``dKstate`` / the state-path ``dV`` / ``dgamma``.
    * **gate backward** (per chunk): ``db -> dA`` via ``reverse_cumsum(db*b)/a``.

    Args:
        Qp, Kp, Vp, Ap: This device's forward inputs, ``[L,dk]`` / ``[L,dk]`` /
            ``[L,dv]`` / ``[L,dk]``.
        C: Chunk size (``L % C == 0``).
        S_recv: Boundary state entering this device, ``[dk,dv]`` (zeros for rank 0).
        dO: Upstream output gradient, ``[L,dv]``.
        dS_total: Grad on the device's end-of-slice local state ``S_total``,
            ``[dk,dv]`` (zeros when nothing downstream needs it, e.g. ``P == 1``).
        dg_total: Grad on the device's total decay ``g_total``, ``[dk]``.

    Returns:
        ``(dQ, dK, dV, dA, dS_recv)`` — input grads matching ``(Q,K,V,A)`` shapes
        plus the received-boundary-state grad ``[dk,dv]`` for the reverse ring.
    """
    L, dk = Qp.shape
    dv = Vp.shape[1]
    assert L % C == 0, f"L={L} not divisible by C={C}"
    N = L // C
    mask = torch.tril(torch.ones((C, C), dtype=Qp.dtype))

    # --- forward recompute: cache the per-chunk quantities the backward reads ---
    b_l, gamma_l, Sprev_l, cprev_l = [], [], [], []
    scores_l, Kstate_l, Qt_l, Kintra_l = [], [], [], []
    S = torch.zeros((dk, dv), dtype=Qp.dtype)
    c = torch.ones((dk,), dtype=Qp.dtype)
    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        q, k, v, a = Qp[lo:hi], Kp[lo:hi], Vp[lo:hi], Ap[lo:hi]
        b = torch.cumprod(a, dim=0)
        gamma = b[-1]
        Sprev_l.append(S.clone())
        cprev_l.append(c.clone())
        Qt = q * b
        Kintra = k / b
        scores = (Qt @ Kintra.t()) * mask
        Kstate = k * (gamma / b)
        b_l.append(b); gamma_l.append(gamma); scores_l.append(scores)
        Kstate_l.append(Kstate); Qt_l.append(Qt); Kintra_l.append(Kintra)
        S = gamma.unsqueeze(1) * S + Kstate.t() @ v
        c = c * gamma

    dQ = torch.zeros_like(Qp)
    dK = torch.zeros_like(Kp)
    dV = torch.zeros_like(Vp)
    dA = torch.zeros_like(Ap)
    dS_recv = torch.zeros((dk, dv), dtype=Qp.dtype)
    db_l = [torch.zeros((C, dk), dtype=Qp.dtype) for _ in range(N)]
    dSprev_l: list[torch.Tensor] = [None] * N        # type: ignore[list-item]
    dcprev_l: list[torch.Tensor] = [None] * N        # type: ignore[list-item]

    # --- grad_o: output-stage adjoint (per chunk, independent) ---
    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        q, k, v = Qp[lo:hi], Kp[lo:hi], Vp[lo:hi]
        b, Qt, Kintra, scores = b_l[n], Qt_l[n], Kintra_l[n], scores_l[n]
        dO_n = dO[lo:hi]
        H_n = Sprev_l[n] + cprev_l[n].unsqueeze(1) * S_recv       # [dk,dv]
        dQt = dO_n @ H_n.t()                                       # [C,dk]
        dH = Qt.t() @ dO_n                                         # [dk,dv]
        dSprev_l[n] = dH
        dcprev_l[n] = (dH * S_recv).sum(dim=1)                     # [dk]
        dS_recv = dS_recv + cprev_l[n].unsqueeze(1) * dH
        # intra masked attention: O_intra = (scores∘mask) @ v
        dsc = (dO_n @ v.t()) * mask                                # [C,C]
        dV[lo:hi] += scores.t() @ dO_n
        dQt = dQt + dsc @ Kintra
        dKintra = dsc.t() @ Qt                                     # [C,dk]
        dK[lo:hi] += dKintra / b                                   # Kintra = k / b
        db = -dKintra * Kintra / b
        dQ[lo:hi] += dQt * b                                       # Qt = q * b
        db = db + dQt * q
        db_l[n] = db

    # --- grad_h: state-stage adjoint (reverse chunk recurrence) ---
    dSloc = dS_total.clone()
    dcvec = dg_total.clone()
    for n in reversed(range(N)):
        lo, hi = n * C, (n + 1) * C
        k, v = Kp[lo:hi], Vp[lo:hi]
        b, gamma, Kstate = b_l[n], gamma_l[n], Kstate_l[n]
        dKstate = v @ dSloc.t()                                    # [C,dk]
        dV[lo:hi] += Kstate @ dSloc
        dgamma_state = (dSloc * Sprev_l[n]).sum(dim=1)             # [dk]
        dSloc_prev = gamma.unsqueeze(1) * dSloc
        dgamma_c = dcvec * cprev_l[n]                              # [dk]
        dcvec_prev = dcvec * gamma
        dK[lo:hi] += dKstate * (gamma / b)                        # Kstate = k*gamma/b
        dgamma_kstate = (dKstate * k / b).sum(dim=0)              # [dk]
        db_l[n] += -dKstate * Kstate / b
        dgamma_n = dgamma_state + dgamma_c + dgamma_kstate        # gamma = b[-1]
        db_l[n][-1] += dgamma_n
        # S_{n-1}^loc / c_{n-1} feed both this update and reconstruct's H_n:
        dSloc = dSloc_prev + (dSprev_l[n] if n > 0 else 0)
        dcvec = dcvec_prev + (dcprev_l[n] if n > 0 else 0)

    # --- gate backward: b -> a  (dA = reverse_cumsum(db*b) / a, per chunk) ---
    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        prod = db_l[n] * b_l[n]                                    # [C,dk]
        rcs = prod.flip(0).cumsum(0).flip(0)                      # sum_{t>=j}
        dA[lo:hi] = rcs / Ap[lo:hi]

    return dQ, dK, dV, dA, dS_recv


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

    def backward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor,
        dO: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ZeCO backward pass and return the input gradients.

        Optional — backends that only implement the forward leave this raising
        ``NotImplementedError``. The reference recomputes the forward internally,
        so it is stateless (no dependence on a prior :meth:`forward` call).

        Args:
            Q, K, V, A: The forward inputs, shapes ``[P, L, dk/dv]`` (rank-major).
            dO: Upstream gradient of the outputs, shape ``[P, L, dv]``.

        Returns:
            ``(dQ, dK, dV, dA)`` matching the shapes of ``(Q, K, V, A)``.
        """
        raise NotImplementedError(f"{type(self).__name__} has no backward")

    #: Whether :meth:`measure` reports *steady-state* per-forward latency — i.e.
    #: the fixed per-call orchestration setup (comm-domain / worker prepare) is
    #: paid once at :meth:`build` time, not inside the timed loop. False means the
    #: timed numbers include whatever per-call setup :meth:`forward` does.
    amortized_timing: bool = False

    def measure(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor,
        n_iters: int,
    ) -> list[float]:
        """Return ``n_iters`` per-forward latency samples in milliseconds.

        Default implementation times :meth:`forward` once per sample, so the
        numbers carry the full per-call overhead. Backends whose per-call cost is
        dominated by fixed orchestration setup (e.g. a ``DistributedWorker``
        prepare/close) override this to prepare once and time only the repeated
        dispatch — the honest *steady-state* operator latency — and set
        :attr:`amortized_timing` to True.

        Args:
            Q: Queries, shape ``[P, L, dk]``.
            K: Keys, shape ``[P, L, dk]``.
            V: Values, shape ``[P, L, dv]``.
            A: Decay gates in ``(0, 1)``, shape ``[P, L, dk]``.
            n_iters: Number of latency samples to collect.

        Returns:
            A list of ``n_iters`` per-forward latencies in milliseconds.
        """
        samples: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.forward(Q, K, V, A)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def close(self) -> None:
        """Release resources (override if needed)."""
