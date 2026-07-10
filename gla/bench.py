#!/usr/bin/env python3
"""ZeCO / GLA forward-pass benchmark suite.

Compares every registered ZeCO implementation (see ``gla/implementations/``):

  torch    — sequential chunk-recurrent reference (CPU baseline, not head-to-head)
  simpler  — hand-written PTO-ISA kernels + real AllScan (per-kernel worker cycles)
  pypto    — one fully-fused distributed @pl.program (single prepare/rt/close)

Every backend is timed **identically**: ``build()`` once, then ``forward(Q,K,V,A)``
end-to-end for warmup + timed iterations. No implementation gets a special amortized
path (``ZeCoImpl`` has no ``measure()`` hook), so the wall-clock is the honest
as-implemented forward latency — *including* each backend's per-call worker setup
(pypto: one ``DistributedWorker`` prepare/close; simpler: ``3P+1`` sequential
per-kernel worker init/close cycles + a separate AllScan worker). ``build_s`` (the
one-time compile) is reported separately.

Usage:
    # Real Ascend hardware (preload HCCL):
    LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \\
        python gla/bench.py --platform a2a3 --device 4,5,6,7

    # Simulator:
    python gla/bench.py --platform a2a3sim --device 0-3

    # Pick implementations / save JSON:
    python gla/bench.py --platform a2a3 --device 4-7 --impl simpler pypto --json out.json
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
# Repo root must precede the script dir (``gla/``) that Python auto-adds at sys.path[0]:
# ``gla/common.py`` would otherwise shadow the top-level ``common`` package. Insert at 0
# unconditionally (a harmless duplicate if it is already present via PYTHONPATH).
if sys.path[:1] != [str(_THIS_DIR)]:
    sys.path.insert(0, str(_THIS_DIR))

from common.harness import parse_devices, percentile, print_table  # noqa: E402
from gla.common import ZeCoImpl, expected_gla, flatten_seq, make_gla_inputs  # noqa: E402
from gla.implementations import REGISTRY  # noqa: E402


# (key, header, width, fmt) — GLA forward timing table.
GLA_COLS = [
    ("impl", "Impl", 7, "s"),
    ("P", "P", 2, "d"),
    ("L", "L", 5, "d"),
    ("C", "C", 4, "d"),
    ("D", "D", 4, "d"),
    ("build_s", "Build(s)", 8, ".2f"),
    ("cold_ms", "Cold(ms)", 10, ".2f"),
    ("mean_ms", "Mean(ms)", 10, ".2f"),
    ("min_ms", "Min(ms)", 9, ".2f"),
    ("p50_ms", "p50(ms)", 9, ".2f"),
    ("p95_ms", "p95(ms)", 9, ".2f"),
    ("correct", "OK", 4, "s"),
]


def bench_one(impl, P, L, C, dk, dv, device_ids, platform, n_warmup, n_iters, verify):
    """Run one (impl, config) and return a result dict with forward-latency stats."""
    t0 = time.perf_counter()
    impl.build(P, L, C, dk, dv, device_ids[:P], platform)
    build_s = time.perf_counter() - t0

    Q, K, V, A = make_gla_inputs(P, L, dk, dv)

    correct: Optional[bool] = None
    max_diff = float("nan")
    if verify:
        O = impl.forward(Q, K, V, A)
        exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)).reshape(P, L, dv)
        max_diff = (O - exp).abs().max().item()
        # on-device chunk math divides by within-chunk cumulative decay → looser than ref
        correct = bool(torch.allclose(O, exp, atol=1e-2))

    # cold start (first timed call)
    t0 = time.perf_counter()
    impl.forward(Q, K, V, A)
    cold_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(max(0, n_warmup - 1)):
        impl.forward(Q, K, V, A)

    lat_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        impl.forward(Q, K, V, A)
        lat_ms.append((time.perf_counter() - t0) * 1e3)

    return {
        "impl": impl.name, "P": P, "L": L, "C": C, "D": dk,
        "build_s": build_s, "cold_ms": cold_ms,
        "mean_ms": statistics.mean(lat_ms), "min_ms": min(lat_ms),
        "p50_ms": percentile(lat_ms, 50), "p95_ms": percentile(lat_ms, 95),
        "correct": correct, "max_diff": max_diff, "raw_ms": lat_ms,
    }


# (P, L, C, D) — D == dk == dv (the simpler kernels require it). L % C == 0.
# C is fixed at 32: the pypto fused kernels materialize full [C,C] tiles (no blocking),
# so C=64 overflows the 184KB vec-buffer limit; C=32 is pypto's ceiling. D up to 64 fits
# (only C drives the [C,C] tiles). We vary the sequence-parallel axes P and L (and the
# state size D) at that fixed C so both backends run the identical workload.
DEFAULT_CONFIGS: list[tuple[int, int, int, int]] = [
    (2, 128, 32, 32),
    (4, 128, 32, 32),
    (2, 256, 32, 32),
    (4, 256, 32, 32),
    (2, 128, 32, 64),
    (4, 128, 32, 64),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZeCO / GLA forward benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("--device", default="4,5,6,7", help="Device IDs ('4,5,6,7' or '4-7').")
    parser.add_argument("--platform", default="a2a3", help="Target platform. Default: a2a3")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iters after cold call. Default: 3")
    parser.add_argument("--iters", type=int, default=10, help="Timed iters per config. Default: 10")
    parser.add_argument("--no-verify", action="store_true", help="Skip correctness check")
    parser.add_argument("--impl", nargs="*", metavar="NAME",
                        help="Impls to run (default: all non-torch). Choices: "
                             + ", ".join(cls.name for cls in REGISTRY))
    parser.add_argument("--json", metavar="FILE", help="Write raw results to JSON")
    args = parser.parse_args()

    device_ids = parse_devices(args.device)
    print(f"Devices : {device_ids}  ({len(device_ids)} available)")
    print(f"Platform: {args.platform}")
    print(f"Warmup  : {args.warmup}   Iters: {args.iters}   Verify: {not args.no_verify}")

    if args.impl:
        selected = set(args.impl)
    else:
        # torch is a CPU reference, not head-to-head — exclude by default.
        selected = {cls.name for cls in REGISTRY if cls.name != "torch"}
    impls = [cls() for cls in REGISTRY if cls.name in selected]
    if not impls:
        sys.exit(f"No matching implementations. Available: {[cls.name for cls in REGISTRY]}")

    configs = [(P, L, C, D) for (P, L, C, D) in DEFAULT_CONFIGS if P <= len(device_ids)]
    if not configs:
        sys.exit(f"Need at least 2 devices, got {len(device_ids)}")

    all_rows: list[dict] = []
    for impl_obj in impls:
        print(f"\n=== {impl_obj.name} ===")
        for (P, L, C, D) in configs:
            print(f"  P={P} L={L} C={C} D={D} ... ", end="", flush=True)
            try:
                row = bench_one(impl_obj, P, L, C, D, D, device_ids, args.platform,
                                args.warmup, args.iters, not args.no_verify)
                ok = "?" if row["correct"] is None else ("Y" if row["correct"] else "N")
                print(f"mean={row['mean_ms']:.2f}ms cold={row['cold_ms']:.2f}ms "
                      f"build={row['build_s']:.2f}s {ok}")
                all_rows.append(row)
            except Exception as exc:
                print(f"FAILED: {exc}")
            finally:
                impl_obj.close()

    print_table(all_rows, cols=GLA_COLS)

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2))
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
