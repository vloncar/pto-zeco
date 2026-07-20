"""Torch reference ZeCO / GLA implementations.

Two flavours, mirroring :mod:`allscan.implementations.torch_ref`:

* :class:`TorchZeCo` — an in-process CPU implementation. It runs the per-device
  chunk-wise GLA scan for every rank, hands the boundary states off through the
  :class:`~allscan.implementations.torch_ref.TorchAllscan` backend (so ZeCO
  literally composes the AllScan primitive), then reconstructs each device's
  output. Fast, deterministic, no NPU — the correctness anchor.

* :func:`run_distributed_zeco` — a ``torch.distributed`` (gloo) baseline that
  spawns one process per rank and exchanges the boundary state over a real
  send/recv ring. Run this module directly to exercise it
  (``python -m gla.implementations.torch_ref``).

Both share the chunk math in :mod:`gla.common`, so the two agree by construction
and both are checked against the sequential :func:`~gla.common.expected_gla`.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from allscan.implementations.torch_ref import (  # noqa: E402
    TorchAllscan,
    _all_scan_backward_p2p,
)
from gla.common import (  # noqa: E402
    ZeCoImpl,
    expected_gla,
    expected_gla_backward,
    flatten_seq,
    gla_chunk_scan,
    gla_reconstruct,
    make_gla_inputs,
)


# ---------------------------------------------------------------------------
# In-process CPU baseline (composes the AllScan backend)
# ---------------------------------------------------------------------------

class TorchZeCo(ZeCoImpl):
    """In-process CPU ZeCO that hands boundary state through :class:`TorchAllscan`."""

    name = "torch"

    def build(self, P, L, C, dk, dv, device_ids, platform):
        """Store config and stand up the AllScan backend for the state hand-off.

        Args are as in :meth:`gla.common.ZeCoImpl.build`; ``device_ids`` /
        ``platform`` are unused (pure in-process CPU). The composed AllScan runs
        with a single pipeline block (``K=1``) — blocking is an on-device
        optimisation and does not change the result.
        """
        self.P, self.L, self.C = P, L, C
        self.dk, self.dv = dk, dv
        self.allscan = TorchAllscan()
        self.allscan.build(dk, dv, 1, P, device_ids, platform)

    def forward(self, Q, K, V, A):
        """ZeCO forward; args/return as in :meth:`gla.common.ZeCoImpl.forward`."""
        P, C, dk, dv = self.P, self.C, self.dk, self.dv

        # --- Stage A + intra: local chunk scan per device ---
        S_locals = torch.zeros((P, dk, dv), dtype=Q.dtype)
        gammas = torch.zeros((P, dk, 1), dtype=Q.dtype)
        caches = []
        for p in range(P):
            S_prev, c_prev, O_intra, S_total, g_total = gla_chunk_scan(
                Q[p], K[p], V[p], A[p], C
            )
            S_locals[p] = S_total
            gammas[p] = g_total.unsqueeze(1)
            caches.append((S_prev, c_prev, O_intra))

        # --- Stage B stream 1: AllScan the boundary state/decay across devices ---
        outs = torch.zeros((P, dk, dv), dtype=Q.dtype)
        self.allscan.run(S_locals, gammas, outs)

        # --- Stage C: reconstruct each device's output from its boundary state ---
        O = torch.zeros((P, self.L, dv), dtype=Q.dtype)
        zero_recv = torch.zeros((dk, dv), dtype=Q.dtype)
        for p in range(P):
            S_recv = zero_recv if p == 0 else outs[p - 1]
            S_prev, c_prev, O_intra = caches[p]
            O[p] = gla_reconstruct(Q[p], A[p], C, S_prev, c_prev, S_recv, O_intra)
        return O

    def backward(self, Q, K, V, A, dO):
        """ZeCO backward; composes :meth:`TorchAllscan.run_backward` for the boundary.

        The SP decomposition of the gradient, mirroring the forward's three stages
        (each rank's stage A + stage C are differentiated *locally* by autograd;
        only the cross-rank boundary is manual — exactly what B2/B3/B4 distribute):

          1. **stage-C adjoint** (local): ``dS_recv[p] = dL/dS_recv_p`` from ``dO[p]``
             — how rank ``p``'s output depends on its incoming boundary state.
          2. **boundary** (reverse ring): the external grad on ``out[p]`` is
             ``dS_recv[p+1]`` (``out[p]`` is rank ``p+1``'s ``S_recv``);
             :meth:`run_backward` turns it into ``dS_total[p]`` / ``dgamma[p]``.
          3. **stage-A adjoint** (local): full ``dQ,dK,dV,dA[p]`` from ``dO[p]``
             *and* the boundary grads ``(dS_total[p], dgamma[p])`` on the outputs
             ``(O_p, S_total, g_total)``.

        Args/return as in :meth:`gla.common.ZeCoImpl.backward`.
        """
        P, C, dk, dv = self.P, self.C, self.dk, self.dv

        # Stage A (local, per rank): build the chunk-scan graph and collect the
        # local end-state / decay that feed the boundary AllScan.
        stageA = []
        S_locals = torch.zeros((P, dk, dv), dtype=Q.dtype)
        gammas = torch.zeros((P, dk, 1), dtype=Q.dtype)
        for p in range(P):
            Qp = Q[p].clone().requires_grad_(True)
            Kp = K[p].clone().requires_grad_(True)
            Vp = V[p].clone().requires_grad_(True)
            Ap = A[p].clone().requires_grad_(True)
            S_prev, c_prev, O_intra, S_total, g_total = gla_chunk_scan(Qp, Kp, Vp, Ap, C)
            stageA.append((Qp, Kp, Vp, Ap, S_prev, c_prev, O_intra, S_total, g_total))
            S_locals[p] = S_total.detach()
            gammas[p] = g_total.detach().unsqueeze(1)

        # Boundary AllScan (forward) — gives the actual S_recv values, and run_backward
        # later needs out[p-1] for dgamma.
        outs = torch.zeros((P, dk, dv), dtype=Q.dtype)
        self.allscan.run(S_locals, gammas, outs)

        # Stage C (local, per rank): reconstruct with S_recv held at its ACTUAL
        # boundary value (out[p-1]) as a fresh leaf — dQ/dA read hist = S_prev +
        # c_prev*S_recv, so the value matters (only dO/dS_recv is value-free).
        tapes = []
        for p in range(P):
            Qp, Kp, Vp, Ap, S_prev, c_prev, O_intra, S_total, g_total = stageA[p]
            S_recv = (outs[p - 1] if p > 0 else torch.zeros((dk, dv), dtype=Q.dtype))
            S_recv = S_recv.clone().detach().requires_grad_(True)
            O_p = gla_reconstruct(Qp, Ap, C, S_prev, c_prev, S_recv, O_intra)
            tapes.append((Qp, Kp, Vp, Ap, S_recv, O_p, S_total, g_total))

        # Phase 1: stage-C adjoint dS_recv[p] = dL/dS_recv_p.
        dS_recv = torch.zeros((P, dk, dv), dtype=Q.dtype)
        for p in range(P):
            _, _, _, _, S_recv, O_p, _, _ = tapes[p]
            (g,) = torch.autograd.grad(O_p, S_recv, dO[p], retain_graph=True)
            dS_recv[p] = g

        # Phase 2: boundary reverse ring. out[p] feeds rank p+1's S_recv, so the
        # external grad on out[p] is dS_recv[p+1]; out[P-1] is unused.
        g_out = torch.zeros((P, dk, dv), dtype=Q.dtype)
        if P > 1:
            g_out[: P - 1] = dS_recv[1:]
        dS = torch.zeros((P, dk, dv), dtype=Q.dtype)
        dgamma = torch.zeros((P, dk, 1), dtype=Q.dtype)
        self.allscan.run_backward(g_out, gammas, outs, dS, dgamma)

        # Phase 3: stage-A adjoint — full input grads from dO[p] + boundary grads.
        dQ, dK, dV, dA = (torch.zeros_like(t) for t in (Q, K, V, A))
        for p in range(P):
            Qp, Kp, Vp, Ap, _, O_p, S_total, g_total = tapes[p]
            gQ, gK, gV, gA = torch.autograd.grad(
                [O_p, S_total, g_total], [Qp, Kp, Vp, Ap],
                [dO[p], dS[p], dgamma[p].squeeze(1)],
            )
            dQ[p], dK[p], dV[p], dA[p] = gQ, gK, gV, gA
        return dQ, dK, dV, dA

    def close(self):
        self.allscan.close()


# ---------------------------------------------------------------------------
# torch.distributed baseline (standalone)
# ---------------------------------------------------------------------------

def _zeco_boundary_p2p(rank, world_size, S_total, g_total, device):
    """Exchange the global boundary state through a forward send/recv ring.

    Implements the AllScan recurrence one rank at a time: rank ``p`` receives the
    prefix ``S_recv = S_{(p-1)L}`` from ``p-1`` (zeros for rank 0), forms its
    outgoing global state ``S_send = S_total + diag(g_total) @ S_recv``, and
    sends it to ``p+1``. Returns the *received* prefix, which is exactly the
    boundary state ZeCO's reconstruction needs.

    Args:
        rank: This process's rank in the ring.
        world_size: Number of ring participants (P).
        S_total: This device's end-of-slice local state, ``[dk, dv]``.
        g_total: This device's total decay over its slice, ``[dk, 1]``.
        device: Torch device for the recv buffer (CPU here).

    Returns:
        The received boundary state ``S_recv``, ``[dk, dv]`` (zeros for rank 0).
    """
    import torch.distributed as dist

    S_recv = torch.zeros_like(S_total, device=device)
    if rank != 0:
        dist.recv(tensor=S_recv, src=rank - 1)
    S_send = S_total + g_total * S_recv
    if rank != world_size - 1:
        dist.send(tensor=S_send.contiguous(), dst=rank + 1)
    return S_recv


def _worker_zeco(rank, world_size, Q, K, V, A, C, output_queue):
    """Per-rank gloo process body for the ZeCO forward reference.

    Args:
        rank: This process's rank.
        world_size: Number of ranks (P).
        Q, K, V, A: All ranks' inputs (this rank uses row ``rank``); passed whole
            for pickling simplicity.
        C: Chunk size.
        output_queue: Queue to return ``(rank, O_rank.tolist())`` to the parent.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")

    S_prev, c_prev, O_intra, S_total, g_total = gla_chunk_scan(
        Q[rank], K[rank], V[rank], A[rank], C
    )
    S_recv = _zeco_boundary_p2p(rank, world_size, S_total, g_total.unsqueeze(1), device)
    O_rank = gla_reconstruct(Q[rank], A[rank], C, S_prev, c_prev, S_recv, O_intra)

    output_queue.put((rank, O_rank.cpu().tolist()))
    dist.destroy_process_group()


def run_distributed_zeco(
    P: int = 4, L: int = 16, C: int = 8, dk: int = 16, dv: int = 16
) -> int:
    """Spawn one gloo process per rank and verify ZeCO against the golden.

    Args:
        P: Number of ranks (processes) to spawn.
        L: Tokens per device.
        C: Chunk size (``L`` must be divisible by ``C``).
        dk: Key/query dimension.
        dv: Value dimension.

    Returns:
        Process exit code: ``0`` if the distributed ZeCO output matches
        :func:`~gla.common.expected_gla`, ``1`` otherwise.
    """
    import torch.multiprocessing as mp

    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    procs = []
    for rank in range(P):
        proc = ctx.Process(
            target=_worker_zeco, args=(rank, P, Q, K, V, A, C, output_queue)
        )
        proc.start()
        procs.append(proc)

    O = torch.zeros((P, L, dv), dtype=torch.float32)
    for _ in range(P):
        rank, O_list = output_queue.get()
        O[rank] = torch.tensor(O_list, dtype=torch.float32)
    for proc in procs:
        proc.join()

    expected = expected_gla(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)
    ).reshape(P, L, dv)
    max_diff = (O - expected).abs().max().item()
    print(f"[torch.distributed zeco] P={P} L={L} C={C} dk={dk} dv={dv}  max diff = {max_diff:.3e}")
    if not torch.allclose(O, expected, atol=1e-4):
        print("[torch.distributed zeco] FAILED")
        return 1
    print("[torch.distributed zeco] matches sequential GLA reference ✅")
    return 0


def _worker_zeco_backward(rank, world_size, Q, K, V, A, C, dO, output_queue):
    """Per-rank gloo process body for the ZeCO backward reference.

    Mirrors the forward worker, then runs the SP backward: local autograd for
    stage A/C, one ``dS_recv`` exchange to hand each rank's stage-C boundary grad
    down to its lower neighbour, and the reverse-ring AllScan backward for the
    boundary-state gradient. Returns this rank's ``(dQ, dK, dV, dA)``.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")

    Qp = Q[rank].clone().requires_grad_(True)
    Kp = K[rank].clone().requires_grad_(True)
    Vp = V[rank].clone().requires_grad_(True)
    Ap = A[rank].clone().requires_grad_(True)
    S_prev, c_prev, O_intra, S_total, g_total = gla_chunk_scan(Qp, Kp, Vp, Ap, C)
    gamma_r = g_total.detach().unsqueeze(1)

    # Forward boundary ring -> the actual S_recv value entering this rank.
    S_recv_val = _zeco_boundary_p2p(rank, world_size, S_total.detach(), gamma_r, device)
    S_recv = S_recv_val.clone().detach().requires_grad_(True)
    O_p = gla_reconstruct(Qp, Ap, C, S_prev, c_prev, S_recv, O_intra)

    # Phase 1 (local): stage-C boundary adjoint dS_recv[rank].
    (dS_recv_local,) = torch.autograd.grad(O_p, S_recv, dO[rank], retain_graph=True)

    # Exchange: this rank's g_out = dS_recv[rank+1] (received from the higher
    # neighbour); hand our own dS_recv[rank] down to the lower neighbour. Recv
    # before send (high->low chain) so the blocking p2p can't deadlock.
    g_out_r = torch.zeros_like(dS_recv_local)
    if rank != world_size - 1:
        dist.recv(tensor=g_out_r, src=rank + 1)
    if rank != 0:
        dist.send(tensor=dS_recv_local.contiguous(), dst=rank - 1)

    # Phase 2: reverse-ring AllScan backward -> (dS_total, dgamma) for this rank.
    dS_r, dgamma_r = _all_scan_backward_p2p(
        rank, world_size, g_out_r, gamma_r, S_recv_val, 1, device
    )

    # Phase 3 (local): full input gradients from dO + the boundary grads.
    gQ, gK, gV, gA = torch.autograd.grad(
        [O_p, S_total, g_total], [Qp, Kp, Vp, Ap],
        [dO[rank], dS_r, dgamma_r.squeeze(1)],
    )
    output_queue.put((rank, gQ.tolist(), gK.tolist(), gV.tolist(), gA.tolist()))
    dist.destroy_process_group()


def run_distributed_zeco_backward(
    P: int = 4, L: int = 16, C: int = 8, dk: int = 16, dv: int = 16
) -> int:
    """Spawn one gloo process per rank and verify the ZeCO backward vs the golden.

    Args are as in :func:`run_distributed_zeco`. Returns ``0`` if every rank's
    ``(dQ, dK, dV, dA)`` matches :func:`~gla.common.expected_gla_backward` on the
    full sequence, ``1`` otherwise.
    """
    import torch.multiprocessing as mp

    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    torch.manual_seed(P * 100 + dk + C)
    dO = torch.randn(P, L, dv)

    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    procs = []
    for rank in range(P):
        proc = ctx.Process(
            target=_worker_zeco_backward, args=(rank, P, Q, K, V, A, C, dO, output_queue)
        )
        proc.start()
        procs.append(proc)

    dQ, dK, dV = (torch.zeros(P, L, dk), torch.zeros(P, L, dk), torch.zeros(P, L, dv))
    dA = torch.zeros(P, L, dk)
    for _ in range(P):
        rank, gQ, gK, gV, gA = output_queue.get()
        dQ[rank], dK[rank] = torch.tensor(gQ), torch.tensor(gK)
        dV[rank], dA[rank] = torch.tensor(gV), torch.tensor(gA)
    for proc in procs:
        proc.join()

    gQ, gK, gV, gA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO)
    )
    ref = (gQ.reshape(P, L, dk), gK.reshape(P, L, dk),
           gV.reshape(P, L, dv), gA.reshape(P, L, dk))
    got = (dQ, dK, dV, dA)
    max_diff = max((g - r).abs().max().item() for g, r in zip(got, ref))
    print(f"[torch.distributed zeco-bwd] P={P} L={L} C={C} dk={dk} dv={dv}  max grad diff = {max_diff:.3e}")
    if max_diff > 1e-4:
        print("[torch.distributed zeco-bwd] FAILED")
        return 1
    print("[torch.distributed zeco-bwd] matches sequential GLA backward reference ✅")
    return 0


if __name__ == "__main__":
    sys.exit(run_distributed_zeco() or run_distributed_zeco_backward())
