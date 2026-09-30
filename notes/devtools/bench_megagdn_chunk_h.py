"""Time megagdn-pto's chunk_h, with library/stream/workspace lookups hoisted out
of the timed region exactly as megagdn's own bench_gdn_kernels.py does.

Configured to match models/gdn/chunk_h.py: single sequence (no cu_seqlens), no
initial state, no final-state output -- so both kernels do the same work.
"""
import argparse
import statistics

import torch
import torch_npu  # noqa: F401

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.kernel_libs import _vp, load_chunk_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=8192)
    ap.add_argument("-H", type=int, default=16)
    ap.add_argument("--Hg", type=int, default=None,
                    help="key heads (GQA); defaults to -H")
    ap.add_argument("-D", type=int, default=128)
    ap.add_argument("-C", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
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
    bd = BLOCK_DIM
    nc = T // C

    k = torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_t = torch.randn(H, T, device=dev, dtype=torch.float32)
    s_out = torch.empty(nc * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    ws = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)

    lib = load_chunk_h(D, C)
    stream = torch.npu.current_stream()._as_parameter_
    ptrs = (_vp(k), _vp(w), _vp(u), _vp(g_t),
            _vp(s_out), _vp(v_out), _vp(None))

    def run():
        # ..., h0, has_initial_state, output_final_state, ws, cu_seqlens, ...
        lib.call_kernel(bd, stream, *ptrs, _vp(None), 0, 0, _vp(ws), _vp(None),
                        1, T, T, H, Hg)

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
    print(f"[mega] chunk_h T={T} H={H} Hg={Hg} D={D} C={C}  batched/{args.batch} us ({args.iters} iters) "
          f"min={us[0]:.1f} median={statistics.median(us):.1f} "
          f"mean={statistics.fmean(us):.1f} max={us[-1]:.1f}")


if __name__ == "__main__":
    main()
