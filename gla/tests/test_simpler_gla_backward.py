"""System test for the simpler-runtime ZeCO / GLA **backward** (B3).

Runs the hand-written PTO-ISA backward kernels (grad_o + grad_h) in the simpler
runtime and checks (dQ,dK,dV,dA) against :func:`gla.common.expected_gla_backward`.

``P == 1`` runs on one device with no boundary (reduces to the sequential GLA
backward), so it is CI-safe on the simulator.  ``P > 1`` exchanges the boundary
gradient over the **real multi-device HCCL AllScan backward** (reverse ring), so
it needs hardware + ``LD_PRELOAD=libhccl.so``::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so \
        python gla/tests/test_simpler_gla_backward.py --platform a2a3 --devices 6,7

The pytest entry honours ``GLA_TEST_PLATFORM`` (default ``a2a3sim``) and
``GLA_TEST_DEVICES`` (default ``0`` -> P=1).
"""

import argparse
import os
import sys
import time

import torch

from gla.common import expected_gla_backward, flatten_seq, make_gla_inputs
from gla.implementations.simpler.impl import SimplerZeCo


def _golden(Q, K, V, A, dO):
    P, L, dk = Q.shape
    dv = V.shape[2]
    gQ, gK, gV, gA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO)
    )
    return (gQ.reshape(P, L, dk), gK.reshape(P, L, dk),
            gV.reshape(P, L, dv), gA.reshape(P, L, dk))


def _worst_rel(dQ, dK, dV, dA, ref, tag="", verbose=True):
    """Max relative error of (dQ,dK,dV,dA) vs the golden ``ref`` tuple."""
    worst = 0.0
    for nm, got, r in zip(("dQ", "dK", "dV", "dA"), (dQ, dK, dV, dA), ref):
        rel = ((got - r).abs().max() / (r.abs().max() + 1e-6)).item()
        worst = max(worst, rel)
        if verbose:
            print(f"[simpler-gla-bwd]{tag} {nm}: max_rel = {rel:.3e}")
    return worst


def run(device_ids, platform="a2a3sim", L=256, C=128, D=128, dv=None):
    P = len(device_ids)
    dk = D
    dv = dv if dv is not None else D
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    torch.manual_seed(1234)
    dO = torch.randn(P, L, dv, dtype=torch.float32)

    impl = SimplerZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids, platform=platform)
    try:
        dQ, dK, dV, dA = impl.backward(Q, K, V, A, dO)
    finally:
        impl.close()

    ref = _golden(Q, K, V, A, dO)
    worst = _worst_rel(dQ, dK, dV, dA, ref, tag=f" P={P} L={L} C={C} dk={dk} dv={dv}")
    ok = worst < 2e-2
    print("[simpler-gla-bwd] RESULT:", "PASS" if ok else "FAIL", f"(worst {worst:.3e})")
    return ok


# Curated backward correctness sweep — (C, D, L, dv); square + rectangular C!=D +
# dk!=dv; larger L => more chunks (N=L/C up to 16).  Runs at P = len(device_ids),
# so --devices 6 sweeps P=1 and --devices 6,7 sweeps P=2 (real AllScan boundary).
SWEEP = [
    (16, 16, 256, None),    # N=16 small square
    (32, 32, 512, None),    # N=16
    (32, 64, 256, None),    # N=8  rect C<D
    (64, 64, 512, None),    # N=8
    (64, 128, 512, None),   # N=8  rect C<D (realistic)
    (128, 128, 512, None),  # N=4  full square
    (128, 64, 256, None),   # N=2  rect C>D
    (16, 128, 256, None),   # N=16 rect C<<D
    (32, 32, 256, 64),      # N=8  dk=32 dv=64
    (64, 32, 256, 16),      # N=4  dk=64 dv=32 (both rectangular)
]

# Multi-device subset: the compute kernels are identical to P=1 (fully covered by
# SWEEP), so for P>1 we only re-exercise the boundary ring across a representative
# spread (small square/large N, rect C<D, 128^2, dk!=dv) — sparing the shared box
# a full 10-shape multi-device sweep.
SWEEP_MULTI = [
    (16, 16, 256, None),    # N=16 small square
    (64, 128, 512, None),   # N=8  rect C<D
    (128, 128, 512, None),  # N=4  full square
    (32, 32, 256, 64),      # N=8  dk=32 dv=64
]


def run_sweep(device_ids, platform="a2a3"):
    """Run the curated backward shape sweep; True iff all pass.

    P=1 sweeps the full shape set (SWEEP); P>1 sweeps the boundary-focused
    SWEEP_MULTI subset (the compute is identical to P=1)."""
    shapes = SWEEP if len(device_ids) == 1 else SWEEP_MULTI
    results = []
    for C, D, L, dv in shapes:
        try:
            ok = run(device_ids, platform=platform, L=L, C=C, D=D, dv=dv)
        except Exception as e:  # noqa: BLE001 - report + continue the sweep
            print(f"[simpler-gla-bwd-sweep] C={C} D={D} L={L} dv={dv} ERROR: {e}")
            ok = False
        results.append((C, D, L, dv, ok))
    print("\n[simpler-gla-bwd-sweep] summary:")
    for C, D, L, dv, ok in results:
        dvv = dv if dv is not None else D
        print(f"  C={C:<4} dk={D:<4} dv={dvv:<4} L={L:<5} N={L // C:<3} -> {'PASS' if ok else 'FAIL'}")
    n_pass = sum(ok for *_, ok in results)
    print(f"[simpler-gla-bwd-sweep] {n_pass}/{len(results)} passed")
    return n_pass == len(results)


def run_stress(device_ids, platform="a2a3", reps=16, L=256, C=128, D=128, dv=None):
    """Back-to-back backward: build once, dispatch ``reps`` backwards on fresh
    random inputs, assert every one matches the golden.  Exercises repeated use of
    the two-AllScan-session pipeline (forward AllScan + reverse-ring run_backward)
    plus the per-kernel worker cycling — the stability check the F1 stress gave the
    collective, now for the full operator backward."""
    P = len(device_ids)
    dk = D
    dv = dv if dv is not None else D
    impl = SimplerZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids, platform=platform)
    worst_all = 0.0
    n_ok = 0
    try:
        for it in range(reps):
            Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=1000 + it)
            torch.manual_seed(7000 + it)
            dO = torch.randn(P, L, dv, dtype=torch.float32)
            dQ, dK, dV, dA = impl.backward(Q, K, V, A, dO)
            ref = _golden(Q, K, V, A, dO)
            worst = _worst_rel(dQ, dK, dV, dA, ref, verbose=False)
            worst_all = max(worst_all, worst)
            ok = worst < 2e-2
            n_ok += ok
            print(f"[simpler-gla-bwd-stress] iter {it:2d}: worst_rel = {worst:.3e} {'ok' if ok else 'FAIL'}")
    finally:
        impl.close()
    print(f"[simpler-gla-bwd-stress] {n_ok}/{reps} correct (worst {worst_all:.3e})  "
          f"P={P} L={L} C={C} dk={dk} dv={dv}")
    return n_ok == reps


def bench(device_ids, platform="a2a3", iters=10, warmup=2, L=256, C=128, D=128, dv=None):
    """Per-backward wall-clock characterization (build once, time backward()).

    The simpler runtime cycles a fresh single-callable worker per kernel dispatch
    (device-exclusive), so a backward pays many worker init/close round-trips plus
    two AllScan comm-domain alloc/free cycles; the compile is session-cached.  This
    reports the honest per-call latency at steady state (build excluded)."""
    P = len(device_ids)
    dk = D
    dv = dv if dv is not None else D
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    torch.manual_seed(1234)
    dO = torch.randn(P, L, dv, dtype=torch.float32)
    impl = SimplerZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids, platform=platform)
    samples = []
    try:
        for it in range(warmup + iters):
            t0 = time.perf_counter()
            impl.backward(Q, K, V, A, dO)
            dt = (time.perf_counter() - t0) * 1e3
            if it >= warmup:
                samples.append(dt)
    finally:
        impl.close()
    mean = sum(samples) / len(samples)
    print(f"[simpler-gla-bwd-bench] P={P} L={L} C={C} dk={dk} dv={dv} N={L // C}: "
          f"mean={mean:.1f}ms min={min(samples):.1f}ms max={max(samples):.1f}ms (n={len(samples)})")
    return samples


def test_simpler_zeco_backward():
    """pytest entry: P=1 on the simulator by default (no NPU / no LD_PRELOAD)."""
    platform = os.environ.get("GLA_TEST_PLATFORM", "a2a3sim")
    devs = [int(x) for x in os.environ.get("GLA_TEST_DEVICES", "0").split(",") if x != ""]
    assert run(devs, platform=platform)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0")
    ap.add_argument("--platform", default="a2a3sim")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--D", type=int, default=128, help="key/query dim dk")
    ap.add_argument("--dv", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="run the curated backward shape sweep")
    ap.add_argument("--stress", type=int, default=0, metavar="REPS",
                    help="back-to-back backward stress: REPS fresh-input dispatches")
    ap.add_argument("--bench", type=int, default=0, metavar="ITERS",
                    help="time ITERS backward passes (build excluded)")
    args = ap.parse_args()
    devs = [int(x) for x in args.devices.split(",") if x != ""]
    if args.sweep:
        sys.exit(0 if run_sweep(devs, args.platform) else 1)
    if args.stress:
        sys.exit(0 if run_stress(devs, args.platform, args.stress,
                                 args.L, args.C, args.D, args.dv) else 1)
    if args.bench:
        bench(devs, args.platform, args.bench, L=args.L, C=args.C, D=args.D, dv=args.dv)
        sys.exit(0)
    sys.exit(0 if run(devs, args.platform, args.L, args.C, args.D, args.dv) else 1)
