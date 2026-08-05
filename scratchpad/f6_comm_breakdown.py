#!/usr/bin/env python3
"""F6.4 — split the two blocks the phase breakdown left lumped.

`f6_phase_breakdown.py` showed simpler's per-call time is 92% orchestration, but two
entries were still composites and neither supports a claim on its own:

  as_run  2411.8 ms  — "the boundary collective", but measured as ONE run on a freshly
                       built worker, so it may be first-run warmup rather than comm.
  stage2  2240.8 ms  — chunk_o dispatch PLUS the compute-worker reopen the boundary forced.

Both are decomposed here by repeating the operation on an already-warm worker:

  A. compute: time open() explicitly, then each kernel dispatch individually, x K.
     Separates "reopen tax" from "what chunk_o actually costs".
  B. comm:    build ONE AllScan worker, then run the boundary K times on it.
     If run[0] >> run[1..], the 2.4 s was per-worker warmup, not per-call comm — which
     is what [[allscan-amortized-benchmark]] (comm tied at P=4/128^2) would predict.

A and B run in separate phases because a device hosts one worker at a time.

Usage: python3 scratchpad/f6_comm_breakdown.py <dev_csv> [platform] [K]
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

from gla.common import make_gla_inputs
from gla.implementations.simpler.impl import SimplerZeCo

P_CFG, L, C, D = 2, 128, 32, 32


def _stats(xs):
    return (f"first={xs[0] * 1e3:8.1f} ms   rest mean={statistics.mean(xs[1:]) * 1e3:8.1f} ms   "
            f"min={min(xs[1:]) * 1e3:8.1f} ms   (n={len(xs) - 1})") if len(xs) > 1 else \
           f"single={xs[0] * 1e3:8.1f} ms"


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    P = min(P_CFG, len(devices))

    torch.manual_seed(P * 100 + L)
    Q, K_t, V, A = make_gla_inputs(P, L, C, D)

    impl = SimplerZeCo()
    impl.build(P, L, C, C, D, devices[:P], platform)

    print(f"=== F6.4 decomposition  P={P} L={L} C={C} D={D}  ({K} repeats) ===\n")

    # ---------------- A. compute: reopen tax vs kernel dispatch ----------------
    impl._use_persistent_runners()
    r0 = impl._runners[0]

    t = time.perf_counter()
    r0.open()
    open_s = time.perf_counter() - t
    print("A. compute (rank 0 device)")
    print(f"   worker open()          {open_s * 1e3:8.1f} ms   <- the 'reopen tax' stage2 pays")

    g_log = torch.log(A[0]).contiguous()
    g_cs = torch.zeros(L, C)
    s_snap = torch.zeros(L // C, C, D)
    o = torch.zeros(L, D)
    from gla.implementations.simpler.impl import (CHUNK_H_SPEC, CHUNK_O_SPEC,
                                                  GATE_CUMSUM_SPEC)

    per_kernel = {}
    for name, sig, args in (
        ("gate_cumsum", GATE_CUMSUM_SPEC["orchestration"]["signature"],
         [("tril", impl._tril), ("g", g_log), ("g_cs", g_cs), ("config", impl._config)]),
        ("chunk_h", CHUNK_H_SPEC["orchestration"]["signature"],
         [("k", K_t[0].contiguous()), ("v", V[0].contiguous()), ("g_cs", g_cs),
          ("s_snap", s_snap), ("config", impl._config)]),
        ("chunk_o", CHUNK_O_SPEC["orchestration"]["signature"],
         [("q", Q[0].contiguous()), ("k", K_t[0].contiguous()), ("v", V[0].contiguous()),
          ("g_cs", g_cs), ("s_snap", s_snap), ("tril", impl._tril), ("o", o),
          ("config", impl._config)]),
    ):
        xs = []
        for _ in range(K):
            t = time.perf_counter()
            r0.run(name, sig, args)
            xs.append(time.perf_counter() - t)
        per_kernel[name] = xs
        print(f"   {name:<22} {_stats(xs)}")

    dispatch_total = sum(statistics.mean(v[1:] if len(v) > 1 else v) for v in per_kernel.values())
    print(f"   -> all three kernels, warm: {dispatch_total * 1e3:.1f} ms\n")

    for r in impl._runners:
        r.close()

    # ---------------- B. comm: build once, run the boundary K times ----------------
    print("B. boundary AllScan (compute workers closed)")
    if P < 2:
        print("   SKIP (needs P>=2)\n")
    else:
        S_totals = [torch.randn(C, D) for _ in range(P)]
        t = time.perf_counter()
        allscan = impl._make_allscan()
        build_s = time.perf_counter() - t
        print(f"   worker build           {build_s * 1e3:8.1f} ms   <- paid EVERY call today")
        try:
            xs = []
            for _ in range(K):
                t = time.perf_counter()
                impl._boundary_on(allscan, S_totals, A)
                xs.append(time.perf_counter() - t)
            print(f"   {'boundary run':<22} {_stats(xs)}")
            warm = statistics.mean(xs[1:]) if len(xs) > 1 else xs[0]
            print(f"\n   VERDICT: first run {xs[0] * 1e3:.1f} ms vs warm {warm * 1e3:.1f} ms -> "
                  f"{'first-run WARMUP, not per-call comm' if xs[0] > 3 * warm else 'genuinely per-run cost'}")
        finally:
            t = time.perf_counter()
            allscan.close()
            print(f"   worker close           {(time.perf_counter() - t) * 1e3:8.1f} ms")

    impl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
