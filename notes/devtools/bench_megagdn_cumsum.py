"""Time megagdn-pto's chunk_cumsum at a fixed shape, for comparison with the
PyPTO port. Event-pair timing, same warmup/iteration counts as the PyPTO side.
"""
import argparse
import statistics

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.kernel_libs import _vp, load_chunk_cumsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=1024)
    ap.add_argument("-H", type=int, default=16)
    ap.add_argument("-C", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20,
                    help="launches bracketed by ONE event pair; amortises per-launch "
                         "event/dispatch overhead so the number reflects steady-state work")
    args = ap.parse_args()

    torch.npu.set_device(args.device)          # mandatory: else the kernel launches on npu:0's stream
    torch.manual_seed(42)
    g = torch.randn(1, args.T, args.H, dtype=torch.float32).npu()
    g_sum = torch.zeros(1, args.T, args.H, dtype=torch.float32).npu()

    # Hoist every per-call lookup out of the timed region, exactly as megagdn's own
    # benchmarks/kernel/bench_gdn_kernels.py does. Timing the run_chunk_cumsum wrapper
    # instead measures its library lookup and stream query, which is ~210 us and
    # completely hides the kernel.
    lib = load_chunk_cumsum(args.C)
    stream = torch.npu.current_stream()._as_parameter_
    bd = BLOCK_DIM
    gp, sp, cup = _vp(g), _vp(g_sum), _vp(None)

    def run():
        lib.call_kernel(bd, stream, gp, sp, cup, 1, args.T, args.H)

    # correctness against the same reference the PyPTO kernel validates against
    run()
    torch.npu.synchronize()
    ref = torch.zeros_like(g.cpu())
    gc = g.cpu()
    for t0 in range(0, args.T, args.C):
        ref[0, t0 : t0 + args.C] = gc[0, t0 : t0 + args.C].cumsum(dim=0)
    err = (g_sum.cpu() - ref).abs().max().item()
    print(f"[mega] max_abs_err vs torch cumsum = {err:.3e}")

    for _ in range(args.warmup):
        run()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    for i in range(args.iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.npu.synchronize()

    us = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(args.iters))
    print(f"[mega] T={args.T} H={args.H} C={args.C} fp32  per-launch-event "
          f"us ({args.iters} iters, {args.warmup} warmup) "
          f"min={us[0]:.1f} median={statistics.median(us):.1f} "
          f"mean={statistics.fmean(us):.1f} max={us[-1]:.1f}")

    # Batched: one event pair around `batch` launches, repeated `iters` times.
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
    bus = sorted(bs[i].elapsed_time(be[i]) * 1000.0 / args.batch for i in range(args.iters))
    print(f"[mega] T={args.T} H={args.H} C={args.C} fp32  batched/{args.batch} "
          f"us ({args.iters} iters) "
          f"min={bus[0]:.1f} median={statistics.median(bus):.1f} "
          f"mean={statistics.fmean(bus):.1f} max={bus[-1]:.1f}")


if __name__ == "__main__":
    main()
