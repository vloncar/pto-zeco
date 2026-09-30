#!/usr/bin/env python3
"""Did key-row blocking make the shapes that ALREADY worked slower?

The new stage2 accumulates the within-chunk term straight into the output, which means that
matmul runs once per HEAD BLOCK instead of once per chunk. At one head block that is identical
to before; at more it is extra work. Against that, dropping the [C,C] score accumulator and
the full tril tile frees enough vector buffer that the search often picks COARSER head
blocking -- fewer head blocks, fewer of those matmuls. Which effect wins is a measurement, not
an argument.

Reports median and two-sided spread (common/harness.latency_stats) -- a mean over a bimodal
distribution is what this project got wrong before.

Usage: a4_perf_check.py <platform> [iters]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/workspace/allscan/pto-zeco")

from common.harness import latency_stats              # noqa: E402
from gla.common import make_gla_inputs                # noqa: E402
from gla.implementations.pypto.impl import PyPtoZeCo  # noqa: E402

CASES = [(256, 64, 64, 64), (256, 64, 128, 128), (256, 64, 64, 128), (128, 32, 32, 32)]


def main() -> int:
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print(f"{'shape':<26} {'plan':<22} {'p50 ms':>9} {'p05':>9} {'p95':>9}")
    for (L, C, dk, dv) in CASES:
        impl = PyPtoZeCo()
        try:
            impl.build(1, L, C, dk, dv, device_ids=[0], platform=platform)
            Q, K, V, A = make_gla_inputs(1, L, dk, dv, seed=42)
            st = latency_stats(impl.measure(Q, K, V, A, n_iters=iters))
            b = getattr(impl, "blocking", None)
            plan = "x".join(str(x) for x in b) if b else "unblocked"
            print(f"L={L} C={C} dk={dk} dv={dv:<6} {plan:<22} "
                  f"{st['p50_ms']:>9.2f} {st['p05_ms']:>9.2f} {st['p95_ms']:>9.2f}"
                  f"{'  DISPERSED' if st['dispersed'] else ''}", flush=True)
        except Exception as e:  # noqa: BLE001 -- one shape failing must not kill the sweep
            print(f"L={L} C={C} dk={dk} dv={dv:<6} {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:60]}", flush=True)
        finally:
            impl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
