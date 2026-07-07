#!/usr/bin/env python3
"""
AllScan **backward** benchmark suite.

Backward analogue of ``bench_allscan.py``. Primary head-to-head: ``simpler`` vs
``pypto`` reverse-ring backward, with fixed per-dispatch orchestration overhead
amortized across a batch (so Mean/Min reflect marginal kernel+comm cost). The
in-process torch reference has no real communication and is excluded here; time
the torch.distributed reverse ring separately for a comm floor.

Usage:
    # Real Ascend hardware (preload HCCL):
    LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \\
        python bench_allscan_backward.py --platform a2a3 --device 4,5,6,7

    # Simulator:
    python bench_allscan_backward.py --platform a2a3sim --device 0-3

    # Save JSON results:
    python bench_allscan_backward.py --platform a2a3 --device 4-7 --json bwd.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_THIS_DIR = Path(__file__).parent.parent  # repo root (pto-zeco/)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from allscan.common import (  # noqa: E402
    AllscanImpl,
    expected_allscan,
    expected_allscan_backward,
    make_grad_inputs,
    make_inputs,
)
from allscan.implementations.pypto.impl import PyPtoAllscanBackward  # noqa: E402
from allscan.implementations.simpler.impl import SimplerAllscan  # noqa: E402
from common.harness import parse_devices, percentile, print_table  # noqa: E402

# name -> backward-capable impl class (pypto uses the backward-only worker).
BACKWARD_IMPLS = {"simpler": SimplerAllscan, "pypto": PyPtoAllscanBackward}


def bench_one_backward(
    impl: AllscanImpl,
    dk: int, dv: int, K: int, P: int,
    device_ids: list[int], platform: str,
    n_warmup: int, n_iters: int, verify: bool,
) -> dict:
    """Run one (backward impl, config) combination and return a result dict.

    Args:
        impl: The backward-capable backend instance (built here).
        dk: Key/row dimension of the state.
        dv: Value/column dimension of the state.
        K: Pipeline depth (number of ``dk // K``-row blocks).
        P: Number of ranks.
        device_ids: Devices available; the first ``P`` are used.
        platform: Target backend (``"a2a3"`` / ``"a2a3sim"`` / ...).
        n_warmup: Warmup backward iterations after the cold-start call.
        n_iters: Timed backward iterations (latency samples) to collect.
        verify: If True, check ``(dS, dgamma)`` against the sequential backward
            reference first.

    Returns:
        A result dict with timing stats, bandwidth, correctness, and raw samples.
    """
    t0 = time.perf_counter()
    impl.build(dk, dv, K, P, device_ids, platform)
    build_s = time.perf_counter() - t0

    S_locals, gammas, _ = make_inputs(P, dk, dv)
    g_out = make_grad_inputs(P, dk, dv)
    outs = expected_allscan(S_locals, gammas)
    dS = torch.zeros((P, dk, dv), dtype=torch.float32)
    dgamma = torch.zeros((P, dk, 1), dtype=torch.float32)

    correct: Optional[bool] = None
    max_diff = float("nan")
    if verify:
        impl.run_backward(g_out, gammas, outs, dS, dgamma)
        exp_dS, exp_dgamma = expected_allscan_backward(gammas, outs, g_out)
        d1 = (dS - exp_dS).abs().max().item()
        d2 = (dgamma - exp_dgamma).abs().max().item()
        max_diff = max(d1, d2)
        correct = bool(
            torch.allclose(dS, exp_dS, atol=1e-3) and torch.allclose(dgamma, exp_dgamma, atol=1e-3)
        )

    # cold start
    t0 = time.perf_counter()
    impl.run_backward(g_out, gammas, outs, dS, dgamma)
    cold_ms = (time.perf_counter() - t0) * 1e3

    # warmup
    for _ in range(n_warmup - 1):
        impl.run_backward(g_out, gammas, outs, dS, dgamma)

    latencies_ms = impl.measure_backward(g_out, gammas, outs, dS, dgamma, n_iters)

    mean_ms = statistics.mean(latencies_ms)
    output_bytes = P * dk * dv * 4  # FP32, dS (dominant term); matches forward BW metric
    bw_mbs = (output_bytes / (mean_ms * 1e-3)) / 1e6

    return {
        "impl": impl.name, "P": P, "dk": dk, "dv": dv, "K": K,
        "build_s": build_s, "cold_ms": cold_ms,
        "mean_ms": mean_ms, "min_ms": min(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50), "p95_ms": percentile(latencies_ms, 95),
        "bw_mbs": bw_mbs, "correct": correct, "max_diff": max_diff,
        "amortized": bool(getattr(impl, "amortized_timing", False)),
        "raw_ms": latencies_ms,
    }


DEFAULT_CONFIGS: list[tuple[int, int, int, int]] = [
    (2, 64, 64, 1), (2, 64, 64, 4),
    (4, 64, 64, 1), (4, 64, 64, 4),
    (4, 128, 128, 1), (4, 128, 128, 4),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AllScan backward benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--device", default="4,5,6,7", help="Device IDs. Default: 4,5,6,7")
    parser.add_argument("--platform", default="a2a3", help="Platform. Default: a2a3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--impl", nargs="*", metavar="NAME",
                        help="Impls to run (default: simpler pypto).")
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args()

    device_ids = parse_devices(args.device)
    print(f"Devices : {device_ids}  ({len(device_ids)} available)")
    print(f"Platform: {args.platform}")
    print(f"Warmup  : {args.warmup}   Iters: {args.iters}   Verify: {not args.no_verify}")

    selected = args.impl if args.impl else list(BACKWARD_IMPLS)
    unknown = [n for n in selected if n not in BACKWARD_IMPLS]
    if unknown:
        sys.exit(f"Unknown impl(s) {unknown}. Available: {list(BACKWARD_IMPLS)}")

    configs = [(P, dk, dv, K) for (P, dk, dv, K) in DEFAULT_CONFIGS if P <= len(device_ids)]
    if not configs:
        sys.exit(f"Need at least 2 devices, got {len(device_ids)}")

    all_rows: list[dict] = []
    for name in selected:
        impl_obj = BACKWARD_IMPLS[name]()
        print(f"\n=== {name} (backward) ===")
        for (P, dk, dv, K) in configs:
            print(f"  P={P} dk={dk} dv={dv} K={K} ... ", end="", flush=True)
            try:
                row = bench_one_backward(
                    impl=impl_obj, dk=dk, dv=dv, K=K, P=P,
                    device_ids=device_ids, platform=args.platform,
                    n_warmup=args.warmup, n_iters=args.iters, verify=not args.no_verify,
                )
                ok = ("Y" if row["correct"] else "N") if row["correct"] is not None else "?"
                print(f"mean={row['mean_ms']:.3f}ms  cold={row['cold_ms']:.3f}ms  {ok}")
                all_rows.append(row)
            except Exception as exc:
                print(f"FAILED: {exc}")
        impl_obj.close()

    print_table(all_rows)

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2))
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
