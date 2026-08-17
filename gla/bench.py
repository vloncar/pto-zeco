#!/usr/bin/env python3
"""ZeCO / GLA benchmark suite — forward and backward.

Compares every registered ZeCO implementation (see ``gla/implementations/``):

  torch    — sequential chunk-recurrent reference (CPU baseline, not head-to-head)
  simpler  — hand-written PTO-ISA kernels + real AllScan (per-kernel worker cycles)
  pypto    — one fully-fused distributed @pl.program (single prepare/rt/close)

Every backend is timed the same way: ``build()`` once, then repeated end-to-end calls
of the selected direction. Timing goes through ``measure`` / ``measure_backward``, so a
backend whose per-call cost is dominated by fixed orchestration setup can amortize it
(prepare at ``build``, time only the dispatch) — the ``SS`` column says, **per
direction**, whether that happened, and ``build_s`` / ``cold_ms`` keep the one-time and
first-call costs visible separately.

**Reading the backward rows (B5.4).** The backends place the backward's work
differently: simpler runs the snapshot shifts and the reverse chunk recurrence as
host-side torch glue between device dispatches, while pypto's backward is one fused
device program. These numbers are therefore end-to-end per-call wall clock, which counts
that host work — the only basis on which the two are comparable. **Do not derive a
compute-vs-comm or kernel-vs-kernel split from them** while F6.6 (work-placement parity)
is open; a device-only figure flatters whichever backend offloads more to the host.

Also note pypto's ``P>1`` backward currently needs a locally carried pypto codegen patch
(comm-dispatch ordering, upstream issue #2397 / PR #2398). On stock pypto every ``P>1``
backward config deadlocks, so any backward number here is measured on a patched
toolchain until that merges.

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
import subprocess
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


def is_shape_ceiling(exc: BaseException) -> bool:
    """True if ``exc`` is a compile-time "this shape does not fit" rejection.

    pypto raises it from ``ir.compile`` when the live tiles overflow the platform's vector
    buffer. It is a property of the (C, D) shape, not a defect, so the benchmark reports it
    as a skip: the backward's ``grad_o`` is the widest kernel in either direction, so a
    shape the forward runs comfortably can be out of range for the backward.
    """
    msg = str(exc)
    return "exceeds platform limit" in msg or "Vec buffer usage" in msg


def is_drain_timeout(exc: BaseException) -> bool:
    """True if ``exc`` looks like the ``507018`` AICore drain-timeout.

    That error wedges the AICore(s) it ran on: follow-on configs reusing the same
    device then fail with cascade errors (``-1``). So a config that hits it marks
    its devices suspect and later configs skip them (F5 operational hardening).
    """
    return "507018" in str(exc)


def npu_smi_snapshot() -> str:
    """Best-effort ``npu-smi info`` chip summary for diagnostics; ``''`` if absent.

    (AICore% is unreliable on driver 25.5.1 — it reads ~100% with no processes —
    so this is for eyeballing HBM / process leaks on failure, not an auto-gate.)
    """
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        lines = [ln for ln in out.stdout.splitlines() if "910" in ln or "HBM-Usage" in ln]
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - diagnostics only, never fatal
        return ""

_THIS_DIR = Path(__file__).parent.parent  # repo root (pto-zeco/)
# Repo root must precede the script dir (``gla/``) that Python auto-adds at sys.path[0]:
# ``gla/common.py`` would otherwise shadow the top-level ``common`` package. Insert at 0
# unconditionally (a harmless duplicate if it is already present via PYTHONPATH).
if sys.path[:1] != [str(_THIS_DIR)]:
    sys.path.insert(0, str(_THIS_DIR))

from common.harness import parse_devices, percentile, print_table  # noqa: E402
from gla.common import (  # noqa: E402
    expected_gla,
    expected_gla_backward,
    flatten_seq,
    make_gla_inputs,
)
from gla.implementations import REGISTRY  # noqa: E402


# (key, header, width, fmt) — GLA timing table. ``Dir`` distinguishes forward from backward
# rows so a --direction both sweep prints one comparable table.
GLA_COLS = [
    ("impl", "Impl", 7, "s"),
    ("dir", "Dir", 4, "s"),
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


def bench_one(impl, P, L, C, dk, dv, device_ids, platform, n_warmup, n_iters, verify,
              direction="forward"):
    """Run one (impl, config, direction) and return a result dict with latency stats.

    Timing is **steady-state where the backend supports it**: it goes through
    ``impl.measure`` / ``impl.measure_backward`` rather than a raw call loop, so a
    backend that pays a fixed per-call orchestration setup (pypto's
    ``DistributedWorker`` prepare/close) can amortize it — prepare once at ``build``,
    time only the repeated dispatch. ``build_s`` (one-time compile + prepare) and
    ``cold_ms`` (first call of this direction) are reported separately so the honest
    split between one-time cost, first-call cost and steady-state operator latency
    stays visible. ``SS`` marks whether *this direction* is amortized — it is read per
    direction, because a backend can be amortized forward and not backward.

    **On comparing backward numbers across backends (B5.4/F6.6).** The two backends put
    the backward's work in different places: simpler runs the snapshot shifts and the
    reverse chunk recurrence as host-side torch glue between device dispatches, while
    pypto's backward is one fused device program. The timings below are therefore
    deliberately **end-to-end per-call wall clock**, which counts that host work. Do not
    turn them into a compute-vs-comm or kernel-vs-kernel split — F6.6 is still open, and
    a device-only figure would flatter whichever backend offloads more to the host.
    """
    backward = direction == "backward"
    t0 = time.perf_counter()
    impl.build(P, L, C, dk, dv, device_ids[:P], platform)
    build_s = time.perf_counter() - t0

    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    # make_gla_inputs seeds torch itself, so dO is deterministic for a given (P, L, dv).
    dO = torch.randn(P, L, dv) if backward else None

    correct: Optional[bool] = None
    max_diff = float("nan")
    if verify:
        if backward:
            got = impl.backward(Q, K, V, A, dO)
            exp = expected_gla_backward(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                                        flatten_seq(A), flatten_seq(dO))
            # Worst relative error over all four gradients, scaled by each one's own
            # magnitude: dQ/dK/dV/dA differ by orders of magnitude, so a single absolute
            # tolerance would be vacuous on the small ones and unreachable on the large.
            rels = []
            for g, e in zip(got, exp):
                e_r = e.reshape(g.shape)
                denom = e_r.abs().max().item()
                rels.append((g - e_r).abs().max().item() / max(denom, 1e-12))
            max_diff = max(rels)
            correct = bool(max_diff < 1e-2)
        else:
            O = impl.forward(Q, K, V, A)
            exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                               flatten_seq(A)).reshape(P, L, dv)
            max_diff = (O - exp).abs().max().item()
            # on-device chunk math divides by within-chunk cumulative decay → looser than ref
            correct = bool(torch.allclose(O, exp, atol=1e-2))

    def call():
        if backward:
            impl.backward(Q, K, V, A, dO)
        else:
            impl.forward(Q, K, V, A)

    # cold start (first timed call). For an amortized backend prepare is already done in
    # build(), so this is close to steady-state; otherwise it carries per-call setup.
    # For pypto's backward this also absorbs the one-time forward->backward worker swap.
    t0 = time.perf_counter()
    call()
    cold_ms = (time.perf_counter() - t0) * 1e3

    for _ in range(max(0, n_warmup)):
        call()

    if backward:
        lat_ms = impl.measure_backward(Q, K, V, A, dO, n_iters)
        steady = getattr(impl, "amortized_timing_backward", False)
    else:
        lat_ms = impl.measure(Q, K, V, A, n_iters)
        steady = getattr(impl, "amortized_timing", False)

    return {
        "impl": impl.name, "dir": "bwd" if backward else "fwd",
        "P": P, "L": L, "C": C, "D": dk,
        "build_s": build_s, "cold_ms": cold_ms,
        "mean_ms": statistics.mean(lat_ms), "min_ms": min(lat_ms),
        "p50_ms": percentile(lat_ms, 50), "p95_ms": percentile(lat_ms, 95),
        "steady": "Y" if steady else "N",
        "correct": correct, "max_diff": max_diff, "raw_ms": lat_ms,
    }


# (P, L, C, D) — D == dk == dv (the simpler kernels require it). L % C == 0.
# C is fixed at 32 so both backends run the identical workload while P, L and D vary.
# The pypto fused kernels materialize full [C,C] tiles (no blocking), so C is what bounds
# them: F3.1 raised the FORWARD to C=64/D=64 (96% of the 184 KB vector buffer), but the
# B4 backward is roughly twice as wide and tops out at C=32 with D<=64 — C=32/D=32 is the
# largest shape BOTH directions reach, which is what a forward-plus-backward comparison
# has to be run at. See ../devtools/b4_shape_probe.py for the measured per-shape bytes.
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
    parser.add_argument("--direction", default="forward",
                        choices=("forward", "backward", "both"),
                        help="Which pass to benchmark. Default: forward. Note 'both' times each "
                             "direction in its own block, NOT alternating per iteration: pypto "
                             "holds one prepared worker at a time, so alternating would pay a "
                             "forward<->backward worker swap every call.")
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
    print(f"Direction: {args.direction}")
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

    # Devices wedged by a 507018 drain-timeout — skip later configs that reuse them
    # so one bad config can't cascade -1 failures across the rest of the sweep.
    suspect: set[int] = set()

    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)

    all_rows: list[dict] = []
    # Direction is the OUTER loop over configs so each (impl, direction) block runs its
    # configs back to back. pypto holds one prepared worker at a time, so interleaving the
    # two directions would pay a forward<->backward swap on every call.
    for impl_obj in impls:
        for direction in directions:
            print(f"\n=== {impl_obj.name} [{direction}] ===")
            for (P, L, C, D) in configs:
                used = device_ids[:P]
                wedged = suspect.intersection(used)
                if wedged:
                    print(f"  P={P} L={L} C={C} D={D} ... SKIP (devices {sorted(wedged)} "
                          f"wedged by an earlier 507018; reset them to re-enable)")
                    continue
                # Clean stale rendezvous before each config so a prior config's killed/leaked
                # comm domain can't hang this one at rootinfo (F5 operational hardening).
                if not args.no_clean:
                    clean_rendezvous()
                print(f"  P={P} L={L} C={C} D={D} ... ", end="", flush=True)
                try:
                    row = bench_one(impl_obj, P, L, C, D, D, device_ids, args.platform,
                                    args.warmup, args.iters, not args.no_verify, direction)
                    ok = "?" if row["correct"] is None else ("Y" if row["correct"] else "N")
                    print(f"mean={row['mean_ms']:.2f}ms cold={row['cold_ms']:.2f}ms "
                          f"build={row['build_s']:.2f}s {ok}")
                    all_rows.append(row)
                except Exception as exc:
                    if is_shape_ceiling(exc):
                        # Not a defect: this shape does not fit the backend's vector
                        # budget. Distinguished from FAILED so a config that is simply out
                        # of range does not read as a broken backend. The backward's
                        # `grad_o` is the widest kernel in either direction, so a shape the
                        # forward reaches can be out of range for the backward.
                        print(f"SKIP (shape exceeds vector budget): {str(exc)[:90]}")
                        continue
                    print(f"FAILED: {exc}")
                    if is_drain_timeout(exc):
                        suspect.update(used)
                        print(f"    -> 507018 drain-timeout: marking devices {used} suspect "
                              f"(skipping later configs that reuse them).")
                        snap = npu_smi_snapshot()
                        if snap:
                            print("    npu-smi at failure:\n      " + snap.replace("\n", "\n      "))
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
