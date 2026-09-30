"""Time megagdn-pto's scaled_dot_kkt at a fixed shape, for comparison with the
PyPTO port. Library/stream lookups are hoisted out of the timed region, exactly
as megagdn's own benchmarks/kernel/bench_gdn_kernels.py does.
"""
import argparse
import statistics

import torch
import torch_npu  # noqa: F401

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.kernel_libs import _vp, load_scaled_dot_kkt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=8192)
    ap.add_argument("-H", type=int, default=16)
    ap.add_argument("--Hg", type=int, default=None,
                    help="key heads (GQA); defaults to -H")
    ap.add_argument("-D", type=int, default=128)
    ap.add_argument("-C", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("-d", "--device", type=int, default=0)
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    dev = f"npu:{args.device}"
    T, H, D, C = args.T, args.H, args.D, args.C
    Hg = args.Hg if args.Hg is not None else H
    if H % Hg:
        raise SystemExit(f"H={H} must be divisible by Hg={Hg}")

    k = torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16)
    beta_t = torch.rand(H, T, device=dev, dtype=torch.float16)
    g_t = torch.randn(H, T, device=dev, dtype=torch.float32)
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    bd = BLOCK_DIM
    ws = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    A = torch.empty(1, T, H, C, device=dev, dtype=torch.float16)

    lib = load_scaled_dot_kkt(D, C)
    stream = torch.npu.current_stream()._as_parameter_
    kp, bp, gp, mp, wp, ap_ = _vp(k), _vp(beta_t), _vp(g_t), _vp(msk), _vp(ws), _vp(A)
    cup = _vp(None)

    def run():
        lib.call_kernel(bd, stream, kp, bp, gp, mp, wp, ap_, cup, 1, T, T, H, Hg)

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
    print(f"[mega] kkt T={T} H={H} Hg={Hg} D={D} C={C}  batched/{args.batch} us ({args.iters} iters) "
          f"min={us[0]:.1f} median={statistics.median(us):.1f} "
          f"mean={statistics.fmean(us):.1f} max={us[-1]:.1f}")


if __name__ == "__main__":
    main()
