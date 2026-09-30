"""Time megagdn-pto's tri_inverse, with every lookup hoisted out of the timed
region -- it exposes `launch_tri_inverse_kernel` for exactly this.

Configured to match models/gdn/solve_tril.py: BSND layout [T, H, C], lower
triangular, no varlen, so both kernels invert the same 1024 matrices of side 128.
"""
import argparse
import statistics

import torch
import torch_npu  # noqa: F401

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.fast_inverse import launch_tri_inverse_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=8192)
    ap.add_argument("-H", type=int, default=16)
    ap.add_argument("-C", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("-d", "--device", type=int, default=0)
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    dev = f"npu:{args.device}"
    T, H, C = args.T, args.H, args.C
    n_mat = (T // C) * H

    a = (0.1 * torch.rand(T, H, C)).to(torch.float16)
    rows = torch.arange(C)[:, None]
    cols = torch.arange(C)[None, :]
    a = (a.view(T // C, C, H, C) * (rows > cols).half()[None, :, None, :]).reshape(T, H, C)
    a_npu = a.contiguous().to(dev)
    out = torch.zeros(T, H, C, device=dev, dtype=torch.float32)
    minus_i = (-torch.eye(C)).to(torch.float16).to(dev)

    stream = torch.npu.current_stream()._as_parameter_

    def run():
        launch_tri_inverse_kernel(out, a_npu, minus_i, C, n_mat, H,
                                  cu_seqlens=None, block_dim=BLOCK_DIM,
                                  stream_ptr=stream, is_lower=True)

    run()
    torch.npu.synchronize()
    for _ in range(args.warmup):
        run()
    torch.npu.synchronize()

    bs = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    be = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    for i in range(args.iters):
        # Synchronise per batch. Queueing every batch and syncing once at the end
        # lets early start events be recorded before the device has begun, which
        # shows up as an implausibly low minimum.
        torch.npu.synchronize()
        bs[i].record()
        for _ in range(args.batch):
            run()
        be[i].record()
        be[i].synchronize()
    us = sorted(bs[i].elapsed_time(be[i]) * 1000.0 / args.batch for i in range(args.iters))
    print(f"[mega] tri_inverse T={T} H={H} C={C} ({n_mat} matrices)  "
          f"batched/{args.batch} us ({args.iters} iters) min={us[0]:.1f} "
          f"median={statistics.median(us):.1f} mean={statistics.fmean(us):.1f} "
          f"max={us[-1]:.1f}")

    # correctness, so a timing number is not quoted for a kernel that is wrong
    got = out.cpu().double()
    eye = torch.eye(C, dtype=torch.float64)
    worst = 0.0
    for ci in range(0, T, C):
        for hh in range(H):
            want = torch.linalg.inv(eye + a[ci : ci + C, hh, :].double())
            g = got[ci : ci + C, hh, :]
            worst = max(worst, float(torch.sqrt(((want - g) ** 2).sum() / (want ** 2).sum())))
    print(f"[mega] worst relative Frobenius error vs fp64 inverse: {worst:.3e}")


if __name__ == "__main__":
    main()
