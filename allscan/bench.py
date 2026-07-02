#!/usr/bin/env python3
"""
AllScan collective communication benchmark suite.

Compares every registered AllScan implementation (see ``implementations/``):

  torch    — in-process CPU K-blocked ring (reference baseline)
  pypto    — PyPTO DSL-compiled AllScan
  simpler  — direct PTO-runtime C++ AllScan

Usage:
    # Real Ascend hardware (preload HCCL):
    LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \\
        python bench_allscan.py --platform a2a3 --device 4,5,6,7

    # Simulator:
    python bench_allscan.py --platform a2a3sim --device 0-3

    # Pick specific implementations:
    python bench_allscan.py --platform a2a3sim --device 0-3 --impl simpler pypto

    # Save JSON results:
    python bench_allscan.py --platform a2a3sim --device 0-3 --json results.json
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

# Make the package importable when invoked from the repo root or any cwd.
_THIS_DIR = Path(__file__).parent.parent  # repo root (pto-zeco/)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from allscan.common import AllscanImpl, expected_allscan, make_inputs  # noqa: E402
from allscan.implementations import REGISTRY  # noqa: E402
from common.harness import parse_devices, percentile, print_table  # noqa: E402


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def bench_one(
    impl: AllscanImpl,
    dk: int,
    dv: int,
    K: int,
    P: int,
    device_ids: list[int],
    platform: str,
    n_warmup: int,
    n_iters: int,
    verify: bool,
) -> dict:
    """Run one (impl, config) combination and return a result dict.

    Args:
        impl: The backend instance to benchmark (built here, left built on return).
        dk: Key/row dimension of the state.
        dv: Value/column dimension of the state.
        K: Pipeline depth (number of ``dk // K``-row blocks).
        P: Number of ranks.
        device_ids: Devices available; the first ``P`` are used.
        platform: Target backend (``"a2a3"`` / ``"a2a3sim"`` / ...).
        n_warmup: Warmup iterations after the cold-start call (heats HCCS links).
        n_iters: Timed iterations (latency samples) to collect.
        verify: If True, check the result against the sequential reference first.

    Returns:
        A result dict with timing stats (mean/min/p50/p95 ms), bandwidth,
        correctness, and the raw per-iteration samples.
    """
    # --- compile / init ---
    t0 = time.perf_counter()
    impl.build(dk, dv, K, P, device_ids, platform)
    build_s = time.perf_counter() - t0

    S_locals, gammas, outputs = make_inputs(P, dk, dv)

    # --- optional correctness check ---
    correct: Optional[bool] = None
    max_diff = float("nan")
    if verify:
        outputs.zero_()
        impl.run(S_locals, gammas, outputs)
        expected = expected_allscan(S_locals, gammas)
        max_diff = (outputs - expected).abs().max().item()
        correct = bool(torch.allclose(outputs, expected, atol=1e-5))

    # --- cold start (first real timed call) ---
    outputs.zero_()
    t0 = time.perf_counter()
    impl.run(S_locals, gammas, outputs)
    cold_ms = (time.perf_counter() - t0) * 1e3

    # --- warmup (additional iterations to heat HCCS links) ---
    for _ in range(n_warmup - 1):
        outputs.zero_()
        impl.run(S_locals, gammas, outputs)

    # --- timed iterations ---
    # measure() defaults to one run() per sample (full per-call overhead), but
    # backends dominated by fixed orchestration setup amortize it across a batch
    # and report the marginal kernel/comm time (impl.amortized_timing == True).
    latencies_ms = impl.measure(S_locals, gammas, outputs, n_iters)

    mean_ms = statistics.mean(latencies_ms)
    output_bytes = P * dk * dv * 4  # FP32
    bw_mbs = (output_bytes / (mean_ms * 1e-3)) / 1e6

    return {
        "impl": impl.name,
        "P": P,
        "dk": dk,
        "dv": dv,
        "K": K,
        "build_s": build_s,
        "cold_ms": cold_ms,
        "mean_ms": mean_ms,
        "min_ms": min(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "bw_mbs": bw_mbs,
        "correct": correct,
        "max_diff": max_diff,
        "amortized": bool(getattr(impl, "amortized_timing", False)),
        "raw_ms": latencies_ms,
    }


# ---------------------------------------------------------------------------
# Default benchmark configurations
# ---------------------------------------------------------------------------

# Tuples of (P, dk, dv, K). Entries requiring P>2 are filtered at runtime
# based on the number of available devices.
DEFAULT_CONFIGS: list[tuple[int, int, int, int]] = [
    (2,  64,  64, 1),
    (2,  64,  64, 4),
    (4,  64,  64, 1),
    (4,  64,  64, 4),
    (4, 128, 128, 1),
    (4, 128, 128, 4),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AllScan collective communication benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device", default="4,5,6,7",
        help="Device IDs ('4,5,6,7' or '4-7'). Default: 4,5,6,7",
    )
    parser.add_argument(
        "--platform", default="a2a3",
        help="Target platform (a2a3, a2a3sim, a5, a5sim). Default: a2a3",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Extra warmup iterations after the cold-start call. Default: 5",
    )
    parser.add_argument(
        "--iters", type=int, default=20,
        help="Timed iterations per config. Default: 20",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip correctness check against sequential reference",
    )
    parser.add_argument(
        "--impl", nargs="*", metavar="NAME",
        help="Implementations to run (default: all). Choices: "
             + ", ".join(cls.name for cls in REGISTRY),
    )
    parser.add_argument(
        "--json", metavar="FILE",
        help="Write raw results to a JSON file",
    )
    args = parser.parse_args()

    device_ids = parse_devices(args.device)
    print(f"Devices : {device_ids}  ({len(device_ids)} available)")
    print(f"Platform: {args.platform}")
    print(f"Warmup  : {args.warmup}   Iters: {args.iters}   Verify: {not args.no_verify}")

    selected_names = set(args.impl) if args.impl else {cls.name for cls in REGISTRY}
    impls = [cls() for cls in REGISTRY if cls.name in selected_names]
    if not impls:
        available = [cls.name for cls in REGISTRY]
        sys.exit(f"No matching implementations. Available: {available}")

    configs = [(P, dk, dv, K) for (P, dk, dv, K) in DEFAULT_CONFIGS
               if P <= len(device_ids)]
    if not configs:
        sys.exit(f"Need at least 2 devices, got {len(device_ids)}")

    all_rows: list[dict] = []
    for impl_obj in impls:
        print(f"\n=== {impl_obj.name} ===")
        for (P, dk, dv, K) in configs:
            label = f"P={P} dk={dk} dv={dv} K={K}"
            print(f"  {label} ... ", end="", flush=True)
            try:
                row = bench_one(
                    impl=impl_obj,
                    dk=dk, dv=dv, K=K, P=P,
                    device_ids=device_ids,
                    platform=args.platform,
                    n_warmup=args.warmup,
                    n_iters=args.iters,
                    verify=not args.no_verify,
                )
                ok = ("Y" if row["correct"] else "N") if row["correct"] is not None else "?"
                print(f"mean={row['mean_ms']:.3f}ms  cold={row['cold_ms']:.3f}ms  {ok}")
                all_rows.append(row)
            except Exception as exc:
                print(f"FAILED: {exc}")
        impl_obj.close()

    print_table(all_rows)

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(all_rows, indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
