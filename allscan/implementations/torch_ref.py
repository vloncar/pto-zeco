"""Torch reference AllScan implementations.

Two flavours live here:

* :class:`TorchAllscan` — an in-process CPU implementation of the K-blocked ring,
  used as the benchmark/test baseline. It is fast, deterministic, and runs
  anywhere (no NPU, no process group), so it anchors correctness comparisons.

* :func:`run_distributed` — the original ``torch.distributed`` point-to-point
  baseline that actually spawns one process per rank and uses ``send``/``recv``.
  Kept as a faithful distributed reference; run this module directly to exercise
  it (``python -m implementations.torch_ref`` or ``python torch_ref.py``).
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from allscan.common import (  # noqa: E402
    AllscanImpl,
    expected_allscan,
    expected_allscan_backward,
    make_grad_inputs,
    make_inputs,
)


# ---------------------------------------------------------------------------
# In-process CPU baseline (benchmark/test adapter)
# ---------------------------------------------------------------------------

class TorchAllscan(AllscanImpl):
    """In-process CPU implementation of the K-blocked AllScan ring."""

    name = "torch"

    def build(self, dk, dv, K, P, device_ids, platform):
        """Store the config (no compilation needed for the CPU reference).

        Args are as in :meth:`common.AllscanImpl.build`; ``device_ids`` and
        ``platform`` are ignored (this backend is pure in-process CPU).
        """
        self.dk, self.dv, self.K, self.P = dk, dv, K, P

    def run(self, S_locals, gammas, outputs):
        """Forward scan; args as in :meth:`common.AllscanImpl.run`."""
        P, K = self.P, self.K
        dk = S_locals.shape[1]
        block = dk // K
        for k in range(K):
            lo, hi = k * block, (k + 1) * block
            # Walk the ring one block at a time, mirroring the pipelined kernels.
            outputs[0, lo:hi] = S_locals[0, lo:hi]
            for p in range(1, P):
                outputs[p, lo:hi] = (
                    S_locals[p, lo:hi] + gammas[p, lo:hi] * outputs[p - 1, lo:hi]
                )

    def run_backward(self, g_out, gammas, outs, dS, dgamma):
        """Backward scan; args as in :meth:`common.AllscanImpl.run_backward`.

        Reverse-ring adjoint scan, K-blocked to mirror the pipelined kernels.
        ``dS`` doubles as storage for the adjoint ``d[p]`` (``dS[p] == d[p]``);
        ``dgamma[0]`` is left zero since ``gamma[0]`` is unused in the forward.
        """
        P, K = self.P, self.K
        dk = g_out.shape[1]
        block = dk // K
        dgamma.zero_()
        for k in range(K):
            lo, hi = k * block, (k + 1) * block
            dS[P - 1, lo:hi] = g_out[P - 1, lo:hi]
            for p in range(P - 2, -1, -1):
                dS[p, lo:hi] = (
                    g_out[p, lo:hi] + gammas[p + 1, lo:hi] * dS[p + 1, lo:hi]
                )
            for p in range(1, P):
                dgamma[p, lo:hi] = (
                    dS[p, lo:hi] * outs[p - 1, lo:hi]
                ).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# torch.distributed point-to-point baseline (standalone)
# ---------------------------------------------------------------------------

def _all_scan_p2p(rank, world_size, S_local, gamma, K, device):
    """Distributed forward AllScan via point-to-point send/recv (one rank's work).

    Args:
        rank: This process's rank in the ring (``0 .. world_size-1``).
        world_size: Number of ring participants (P).
        S_local: This rank's local state, ``[dk, dv]``.
        gamma: This rank's decay factor, ``[dk, 1]``.
        K: Pipeline depth (number of ``dk // K``-row blocks).
        device: Torch device for the recv scratch buffer (CPU here).

    Returns:
        This rank's scan output ``S_send``, ``[dk, dv]``.
    """
    import torch.distributed as dist

    assert S_local.shape[0] % K == 0
    block = S_local.shape[0] // K
    S_send = torch.zeros_like(S_local)
    S_recv_k = torch.zeros((block, S_local.shape[1]), dtype=S_local.dtype, device=device)

    for k in range(K):
        lo, hi = k * block, (k + 1) * block
        if rank != 0:
            dist.recv(tensor=S_recv_k, src=rank - 1)
        else:
            S_recv_k.zero_()

        S_send_k = S_local[lo:hi] + gamma[lo:hi] * S_recv_k
        S_send[lo:hi] = S_send_k

        if rank != world_size - 1:
            dist.send(tensor=S_send_k, dst=rank + 1)
    return S_send


def _worker(rank, world_size, S_locals, gammas, K, output_queue):
    """Per-rank gloo process body for the forward reference.

    Args:
        rank: This process's rank.
        world_size: Number of ranks (P).
        S_locals: All ranks' local state, ``[P, dk, dv]`` (this rank uses row
            ``rank``); passed whole for pickling simplicity.
        gammas: All ranks' decay factors, ``[P, dk, 1]``.
        K: Pipeline depth.
        output_queue: Queue to return ``(rank, S_send.tolist())`` to the parent.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    S_send = _all_scan_p2p(rank, world_size, S_locals[rank], gammas[rank], K, device)
    output_queue.put((rank, S_send.cpu().tolist()))
    dist.destroy_process_group()


def run_distributed(P: int = 4, dk: int = 64, dv: int = 64, K: int = 4) -> int:
    """Spawn one gloo process per rank and verify against the sequential reference.

    Args:
        P: Number of ranks (processes) to spawn.
        dk: Key/row dimension.
        dv: Value/column dimension.
        K: Pipeline depth.

    Returns:
        Process exit code: ``0`` if the distributed result matches the sequential
        reference, ``1`` otherwise.
    """
    import torch.multiprocessing as mp

    S_locals, gammas, _ = make_inputs(P, dk, dv)
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    procs = []
    for rank in range(P):
        p = ctx.Process(target=_worker, args=(rank, P, S_locals, gammas, K, output_queue))
        p.start()
        procs.append(p)

    outputs = torch.zeros_like(S_locals)
    for _ in range(P):
        rank, S_send_list = output_queue.get()
        outputs[rank] = torch.tensor(S_send_list, dtype=torch.float32)
    for p in procs:
        p.join()

    expected = expected_allscan(S_locals, gammas)
    max_diff = (outputs - expected).abs().max().item()
    print(f"[torch.distributed] P={P} dk={dk} dv={dv} K={K}  max diff = {max_diff:.3e}")
    if not torch.allclose(outputs, expected, atol=1e-5):
        print("[torch.distributed] FAILED")
        return 1
    print("[torch.distributed] matches sequential reference ✅")
    return 0


# ---------------------------------------------------------------------------
# torch.distributed backward (reverse-ring) baseline (standalone)
# ---------------------------------------------------------------------------

def _all_scan_backward_p2p(rank, world_size, g_out_r, gamma_r, fwd_recv_r, K, device):
    """Distributed AllScan backward via reverse-ring send/recv.

    The forward ring flows ``rank -> rank+1``; the adjoint flows the other way,
    ``rank -> rank-1``. Rank ``p`` receives ``gamma[p+1] * d[p+1]`` from ``p+1``,
    adds its own upstream grad to form ``d[p]``, and forwards ``gamma[p] * d[p]``
    to ``p-1``. ``dgamma[p]`` reduces locally against ``fwd_recv_r == out[p-1]``,
    which is exactly the block rank ``p`` received during the forward pass.

    Args:
        rank: This process's rank in the ring.
        world_size: Number of ring participants (P).
        g_out_r: This rank's upstream gradient ``g_out[rank]``, ``[dk, dv]``.
        gamma_r: This rank's decay factor ``gamma[rank]``, ``[dk, 1]``.
        fwd_recv_r: ``out[rank-1]`` — the block this rank received in the forward
            pass (zeros for rank 0), ``[dk, dv]``; used for the local dgamma reduction.
        K: Pipeline depth.
        device: Torch device for scratch buffers (CPU here).

    Returns:
        ``(dS_r[dk,dv], dgamma_r[dk,1])`` — this rank's gradients (``dgamma_r`` is
        0 for rank 0).
    """
    import torch.distributed as dist

    dk = g_out_r.shape[0]
    assert dk % K == 0
    block = dk // K
    dS_r = torch.zeros_like(g_out_r)
    dgamma_r = torch.zeros((dk, 1), dtype=g_out_r.dtype, device=device)
    recv_k = torch.zeros((block, g_out_r.shape[1]), dtype=g_out_r.dtype, device=device)

    for k in range(K):
        lo, hi = k * block, (k + 1) * block
        # Receive gamma[p+1] * d[p+1] from the higher neighbour (none for the tail).
        if rank != world_size - 1:
            dist.recv(tensor=recv_k, src=rank + 1)
        else:
            recv_k.zero_()

        d_k = g_out_r[lo:hi] + recv_k          # d[p] block
        dS_r[lo:hi] = d_k

        if rank != 0:
            # dgamma[p] = rowsum_dv( d[p] * out[p-1] ); out[p-1] is the forward recv.
            dgamma_r[lo:hi] = (d_k * fwd_recv_r[lo:hi]).sum(dim=1, keepdim=True)
            # Forward gamma[p] * d[p] to the lower neighbour.
            dist.send(tensor=(gamma_r[lo:hi] * d_k).contiguous(), dst=rank - 1)

    return dS_r, dgamma_r


def _worker_backward(rank, world_size, g_out, gammas, fwd_recv, K, output_queue):
    """Per-rank gloo process body for the backward reference.

    Args:
        rank: This process's rank.
        world_size: Number of ranks (P).
        g_out: All ranks' upstream gradients, ``[P, dk, dv]`` (this rank uses row ``rank``).
        gammas: All ranks' decay factors, ``[P, dk, 1]``.
        fwd_recv: Per-rank ``out[rank-1]`` blocks, ``[P, dk, dv]`` (row 0 is zeros).
        K: Pipeline depth.
        output_queue: Queue to return ``(rank, dS_r.tolist(), dgamma_r.tolist())``.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    dS_r, dgamma_r = _all_scan_backward_p2p(
        rank, world_size, g_out[rank], gammas[rank], fwd_recv[rank], K, device
    )
    output_queue.put((rank, dS_r.cpu().tolist(), dgamma_r.cpu().tolist()))
    dist.destroy_process_group()


def run_distributed_backward(P: int = 4, dk: int = 64, dv: int = 64, K: int = 4) -> int:
    """Spawn one gloo process per rank for the backward pass and verify.

    Args:
        P: Number of ranks (processes) to spawn.
        dk: Key/row dimension.
        dv: Value/column dimension.
        K: Pipeline depth.

    Returns:
        Process exit code: ``0`` if ``(dS, dgamma)`` match the sequential
        backward reference, ``1`` otherwise.
    """
    import torch.multiprocessing as mp

    S_locals, gammas, _ = make_inputs(P, dk, dv)
    g_out = make_grad_inputs(P, dk, dv)
    outs = expected_allscan(S_locals, gammas)

    # Each rank's forward recv buffer holds out[rank-1] (zero for rank 0). This
    # is data already resident on the rank from the forward pass, not a fresh
    # exchange — mirroring how a fused fwd+bwd step would retain it.
    fwd_recv = torch.zeros_like(outs)
    fwd_recv[1:] = outs[:-1]

    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    procs = []
    for rank in range(P):
        p = ctx.Process(
            target=_worker_backward,
            args=(rank, P, g_out, gammas, fwd_recv, K, output_queue),
        )
        p.start()
        procs.append(p)

    dS = torch.zeros((P, dk, dv), dtype=torch.float32)
    dgamma = torch.zeros((P, dk, 1), dtype=torch.float32)
    for _ in range(P):
        rank, dS_list, dgamma_list = output_queue.get()
        dS[rank] = torch.tensor(dS_list, dtype=torch.float32)
        dgamma[rank] = torch.tensor(dgamma_list, dtype=torch.float32)
    for p in procs:
        p.join()

    exp_dS, exp_dgamma = expected_allscan_backward(gammas, outs, g_out)
    dS_diff = (dS - exp_dS).abs().max().item()
    dgamma_diff = (dgamma - exp_dgamma).abs().max().item()
    print(
        f"[torch.distributed bwd] P={P} dk={dk} dv={dv} K={K}  "
        f"max diff  dS={dS_diff:.3e}  dgamma={dgamma_diff:.3e}"
    )
    if not (
        torch.allclose(dS, exp_dS, atol=1e-5)
        and torch.allclose(dgamma, exp_dgamma, atol=1e-5)
    ):
        print("[torch.distributed bwd] FAILED")
        return 1
    print("[torch.distributed bwd] matches sequential reference ✅")
    return 0


if __name__ == "__main__":
    rc = run_distributed()
    rc |= run_distributed_backward()
    sys.exit(rc)
