#!/usr/bin/env python3
"""Where does simpler's per-call time actually go, once workers are amortized?

The amortized bench still shows ~29-32 s/call for simpler vs 12-200 ms for pypto.
Since the persistent path removed the 3P+1 per-kernel worker cycling, the remainder
must be elsewhere. This splits one persistent forward into its phases so the fair
comparison can separate *compute* from *runtime orchestration*:

  stage1      — gate_cumsum + chunk_h dispatches on held workers (P devices)
  close       — tearing the compute workers down (forced: a device hosts one worker)
  as_build    — standing up the HCCL AllScan distributed worker
  as_run      — the actual boundary collective
  as_close    — tearing it down
  stage2      — chunk_o dispatches (reopens the compute workers)

pypto pays none of the as_* / close phases per call: one DistributedWorker serves its
whole fused program and is held across calls. So the honest headline is which part is
kernel time and which is the runtime integration.

Usage: python3 scratchpad/f6_phase_breakdown.py <dev_csv> [platform] [iters]
"""

from __future__ import annotations

import sys
import time

import torch

from gla.common import make_gla_inputs
from gla.implementations.simpler.impl import SimplerZeCo, _shift_snaps

CONFIGS = [(2, 128, 32, 32), (4, 128, 32, 32)]


def timed_forward(impl, Q, K, V, A, acc):
    """One persistent forward, accumulating per-phase seconds into ``acc``."""
    P, L, C, dk, dv = impl.P, impl.L, impl.C, impl.dk, impl.dv

    t = time.perf_counter()
    s_snaps, g_css, S_totals = [], [], []
    for p in range(P):
        s_snap, g_cs, S_total = impl._stage1(p, Q[p], K[p], V[p], A[p])
        s_snaps.append(s_snap); g_css.append(g_cs); S_totals.append(S_total)
    acc["stage1"] += time.perf_counter() - t

    if P == 1:
        S_recvs = [torch.zeros(dk, dv, dtype=torch.float32)]
    else:
        t = time.perf_counter()
        for r in impl._runners:
            r.close()
        acc["close"] += time.perf_counter() - t

        t = time.perf_counter()
        allscan = impl._make_allscan()
        acc["as_build"] += time.perf_counter() - t
        try:
            t = time.perf_counter()
            out = impl._boundary_on(allscan, S_totals, A)
            acc["as_run"] += time.perf_counter() - t
        finally:
            t = time.perf_counter()
            allscan.close()
            acc["as_close"] += time.perf_counter() - t
        S_recvs = [torch.zeros(dk, dv, dtype=torch.float32) if p == 0 else out[p - 1]
                   for p in range(P)]

    t = time.perf_counter()
    O = torch.zeros(P, L, dv, dtype=torch.float32)
    for p in range(P):
        s_shift = _shift_snaps(s_snaps[p], A[p], S_recvs[p], L, C, dk)
        O[p] = impl._stage2(p, Q[p], K[p], V[p], g_css[p], s_shift)
    acc["stage2"] += time.perf_counter() - t
    return O


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    order = ["stage1", "close", "as_build", "as_run", "as_close", "stage2"]

    for (P, L, C, D) in CONFIGS:
        if P > len(devices):
            print(f"P={P}: SKIP (needs {P} devices)")
            continue
        torch.manual_seed(P * 100 + L)
        Q, K, V, A = make_gla_inputs(P, L, C, D)

        impl = SimplerZeCo()
        impl.build(P, L, C, C, D, devices[:P], platform)
        acc = dict.fromkeys(order, 0.0)
        try:
            impl._use_persistent_runners()
            timed_forward(impl, Q, K, V, A, dict.fromkeys(order, 0.0))   # warm
            for _ in range(iters):
                timed_forward(impl, Q, K, V, A, acc)
            for r in impl._runners:
                r.close()
        finally:
            impl.close()

        total = sum(acc.values())
        print(f"\nP={P} L={L} C={C} D={D}  ({iters} iters, mean per call)")
        print(f"  {'phase':<10} {'ms':>10}  {'share':>7}")
        for k in order:
            ms = acc[k] / iters * 1e3
            print(f"  {k:<10} {ms:>10.1f}  {100 * acc[k] / total:>6.1f}%")
        print(f"  {'TOTAL':<10} {total / iters * 1e3:>10.1f}")
        compute = (acc["stage1"] + acc["stage2"]) / iters * 1e3
        orch = (total - acc["stage1"] - acc["stage2"]) / iters * 1e3
        print(f"  -> compute dispatch {compute:.1f} ms | runtime orchestration {orch:.1f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
