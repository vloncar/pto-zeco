"""Vendor baseline for the GDN gated RMSNorm + per-token INT8 quant, on torch_npu.

No hand-written PTO-ISA counterpart exists for this piece, so the baseline is
the vendor library: torch_npu's fused RMSNorm and dynamic-quant ops where they
cover our arithmetic, plus eager silu. Both sides move the same bytes, so the
metric is GB/s.

    /root/mega-venv/bin/python bench_torchnpu_gated_rmsnorm.py --seq-len 8192
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

H, D, HIDDEN = 48, 128, 48 * 128
EPS = 1e-6


def timed(fn, rounds: int, warmup: int) -> float:
    """Median wall time in us, device-synchronised."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    t = args.seq_len
    o = torch.randn(t * H, D, dtype=torch.float16).npu()
    z = torch.randn(t * H, D, dtype=torch.bfloat16).npu()
    w = torch.ones(D, dtype=torch.bfloat16).npu()

    # Bytes our kernel moves: o and z in, int8 y and one fp32 scale per token out.
    moved = t * HIDDEN * 2 * 2 + t * HIDDEN + t * 4

    def eager():
        of = o.float()
        inv = torch.rsqrt((of * of).mean(-1, keepdim=True) + EPS)
        y = (of * inv * w.float() * torch.nn.functional.silu(z.float())).reshape(t, HIDDEN)
        amax = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        return torch.round(y * (127.0 / amax)).clamp(-127, 127).to(torch.int8), amax / 127.0

    variants = {"eager fp32 chain": eager}

    if hasattr(torch_npu, "npu_rms_norm"):
        def fused_norm():
            normed = torch_npu.npu_rms_norm(o, w.to(o.dtype), epsilon=EPS)[0]
            y = (normed.float() * torch.nn.functional.silu(z.float())).reshape(t, HIDDEN)
            amax = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
            return torch.round(y * (127.0 / amax)).clamp(-127, 127).to(torch.int8), amax / 127.0

        variants["npu_rms_norm + eager gate/quant"] = fused_norm

    if hasattr(torch_npu, "npu_dynamic_quant"):
        def fused_both():
            normed = torch_npu.npu_rms_norm(o, w.to(o.dtype), epsilon=EPS)[0]
            y = (normed.float() * torch.nn.functional.silu(z.float())).reshape(t, HIDDEN)
            return torch_npu.npu_dynamic_quant(y.to(torch.bfloat16))

        variants["npu_rms_norm + npu_dynamic_quant"] = fused_both

    print(f"T={t} H={H} D={D}, {moved / 2**20:.1f} MiB moved, "
          f"{args.rounds} rounds / {args.warmup} warmup, device {args.device}")
    for name, fn in variants.items():
        try:
            us = timed(fn, args.rounds, args.warmup)
        except Exception as exc:  # a vendor op this version does not have, or a dtype it rejects
            print(f"  {name:<40} unavailable: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        print(f"  {name:<40} {us:9.1f} us   {moved / us / 1e3:6.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
