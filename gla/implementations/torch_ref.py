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

from allscan.implementations.torch_ref import TorchAllscan  # noqa: E402
from gla.common import (  # noqa: E402
    ZeCoImpl,
    expected_gla,
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


if __name__ == "__main__":
    sys.exit(run_distributed_zeco())
