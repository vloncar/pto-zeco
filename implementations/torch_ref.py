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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import AllscanImpl, expected_allscan, make_inputs  # noqa: E402


# ---------------------------------------------------------------------------
# In-process CPU baseline (benchmark/test adapter)
# ---------------------------------------------------------------------------

class TorchAllscan(AllscanImpl):
    """In-process CPU implementation of the K-blocked AllScan ring."""

    name = "torch"

    def build(self, dk, dv, K, P, device_ids, platform):
        self.dk, self.dv, self.K, self.P = dk, dv, K, P

    def run(self, S_locals, gammas, outputs):
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


# ---------------------------------------------------------------------------
# torch.distributed point-to-point baseline (standalone)
# ---------------------------------------------------------------------------

def _all_scan_p2p(rank, world_size, S_local, gamma, K, device):
    """Distributed All-Scan via point-to-point send/recv. ``S_local``: [dk, dv]."""
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
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    S_send = _all_scan_p2p(rank, world_size, S_locals[rank], gammas[rank], K, device)
    output_queue.put((rank, S_send.cpu().tolist()))
    dist.destroy_process_group()


def run_distributed(P: int = 4, dk: int = 64, dv: int = 64, K: int = 4) -> int:
    """Spawn one gloo process per rank and verify against the sequential reference."""
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


if __name__ == "__main__":
    sys.exit(run_distributed())
