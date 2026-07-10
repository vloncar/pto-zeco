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
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import torch

#: Stale HCCL rendezvous files leaked by a killed distributed run — they make the
#: next run hang at "Timeout waiting for rootinfo" (see the ``hccl-rootinfo-timeout``
#: note). Cleaned between configs by default.
_RENDEZVOUS_GLOB = "/tmp/barrier_pto_multi_comm_*"


def clean_rendezvous() -> int:
    """Delete stale HCCL rendezvous files; return how many were removed."""
    removed = 0
    for path in glob.glob(_RENDEZVOUS_GLOB):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def check_preload(platform: str) -> None:
    """Warn if a hardware run is missing the mandatory HCCL ``LD_PRELOAD`` — without
    it every distributed pto run hangs at the rootinfo rendezvous."""
    if platform.endswith("sim"):
        return
    if "libhccl" not in os.environ.get("LD_PRELOAD", ""):
        print("  WARNING: LD_PRELOAD does not contain libhccl.so — distributed runs "
              "will hang at rootinfo rendezvous.\n"
              "           Prefix with: LD_PRELOAD=<cann>/aarch64-linux/lib64/libhccl.so",
              file=sys.stderr)

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
    ("steady", "SS", 4, "s"),
    ("correct", "OK", 4, "s"),
]


def bench_one(impl, P, L, C, dk, dv, device_ids, platform, n_warmup, n_iters, verify):
    """Run one (impl, config) and return a result dict with forward-latency stats.

    Timing is **steady-state**: it uses ``impl.measure`` (not a raw ``forward`` loop),
    so a backend that pays a fixed per-call orchestration setup (pypto's
    ``DistributedWorker`` prepare/close) can amortize it — prepare once at ``build``,
    time only the repeated dispatch. ``build_s`` (one-time compile + prepare) and
    ``cold_ms`` (first forward) are reported separately so the honest split between
    one-time cost, first-call cost, and steady-state operator latency is visible.
    ``SS`` marks whether the backend reports amortized (steady-state) numbers.
    """
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

    # cold start (first timed call). For an amortized backend prepare is already done in
    # build(), so this is close to steady-state; otherwise it carries per-call setup.
    t0 = time.perf_counter()
    impl.forward(Q, K, V, A)
    cold_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(max(0, n_warmup)):
        impl.forward(Q, K, V, A)

    # steady-state samples (prepare-once backends time only the dispatch)
    lat_ms = impl.measure(Q, K, V, A, n_iters)

    return {
        "impl": impl.name, "P": P, "L": L, "C": C, "D": dk,
        "build_s": build_s, "cold_ms": cold_ms,
        "mean_ms": statistics.mean(lat_ms), "min_ms": min(lat_ms),
        "p50_ms": percentile(lat_ms, 50), "p95_ms": percentile(lat_ms, 95),
        "steady": "Y" if getattr(impl, "amortized_timing", False) else "N",
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
    parser.add_argument("--no-clean", action="store_true",
                        help="Do not delete stale /tmp/barrier_pto_multi_comm_* between configs")
    parser.add_argument("--impl", nargs="*", metavar="NAME",
                        help="Impls to run (default: all non-torch). Choices: "
                             + ", ".join(cls.name for cls in REGISTRY))
    parser.add_argument("--json", metavar="FILE", help="Write raw results to JSON")
    args = parser.parse_args()

    device_ids = parse_devices(args.device)
    print(f"Devices : {device_ids}  ({len(device_ids)} available)")
    print(f"Platform: {args.platform}")
    print(f"Warmup  : {args.warmup}   Iters: {args.iters}   Verify: {not args.no_verify}")
    check_preload(args.platform)

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

    if not args.no_clean:
        n = clean_rendezvous()
        if n:
            print(f"Cleaned  : {n} stale rendezvous file(s) before start")

    all_rows: list[dict] = []
    for impl_obj in impls:
        print(f"\n=== {impl_obj.name} ===")
        for (P, L, C, D) in configs:
            # Clean stale rendezvous before each config so a prior config's killed/leaked
            # comm domain can't hang this one at rootinfo (F5 operational hardening).
            if not args.no_clean:
                clean_rendezvous()
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
                # Always release the worker AND clear any rendezvous it leaked, so a
                # failed config (e.g. device 507018) can't poison the next one.
                impl_obj.close()
                if not args.no_clean:
                    clean_rendezvous()

    print_table(all_rows, cols=GLA_COLS)

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2))
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
