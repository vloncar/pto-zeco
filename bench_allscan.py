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
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import AllscanImpl, expected_allscan, make_inputs  # noqa: E402
from implementations import REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_devices(raw: str) -> list[int]:
    devices: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            devices.extend(range(int(a), int(b) + 1))
        else:
            devices.append(int(token))
    return list(dict.fromkeys(devices))


def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


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
    """Run one (impl, config) combination and return a result dict."""

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
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "bw_mbs": bw_mbs,
        "correct": correct,
        "max_diff": max_diff,
        "amortized": bool(getattr(impl, "amortized_timing", False)),
        "raw_ms": latencies_ms,
    }


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

_COLS = [
    # (key,        header,     width, fmt)
    ("impl",     "Impl",       7,  "s"),
    ("P",        "P",          2,  "d"),
    ("dk",       "dk",         4,  "d"),
    ("dv",       "dv",         4,  "d"),
    ("K",        "K",          2,  "d"),
    ("build_s",  "Build(s)",   8,  ".2f"),
    ("cold_ms",  "Cold(ms)",   9,  ".3f"),
    ("mean_ms",  "Mean(ms)",   9,  ".3f"),
    ("min_ms",   "Min(ms)",    8,  ".3f"),
    ("p50_ms",   "p50(ms)",    8,  ".3f"),
    ("p95_ms",   "p95(ms)",    8,  ".3f"),
    ("bw_mbs",   "BW(MB/s)",   9,  ".2f"),
    ("correct",  "OK",         4,  "s"),
]


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    header = "  ".join(f"{label:>{width}}" for _, label, width, _ in _COLS)
    sep    = "  ".join("-" * width          for _, _,     width, _ in _COLS)
    print("\n" + header)
    print(sep)
    any_amortized = False
    for row in rows:
        parts = []
        for key, _, width, fmt in _COLS:
            val = row[key]
            if key == "impl" and row.get("amortized"):
                val = f"{val}*"
                any_amortized = True
            if key == "correct":
                val = "?" if val is None else ("Y" if val else "N")
                parts.append(f"{val:>{width}}")
            else:
                parts.append(f"{val:{width}{fmt}}")
        print("  ".join(parts))
    print()
    if any_amortized:
        print("* timing amortizes fixed per-call orchestration setup (comm-domain "
              "alloc/free + drain) across a batch, so Mean/Min/etc. reflect the "
              "marginal kernel+comm cost rather than full per-call latency.\n")


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

    device_ids = _parse_devices(args.device)
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
