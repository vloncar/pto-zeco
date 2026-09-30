#!/usr/bin/env python3
"""Did blocking make the shapes that ALREADY worked slower?

The chunk kernels now walk the head dim in a nested loop and slice/reassemble the carried
state even when there is only one block, and the plan search may pick a shallower cube<->vector
ring than the old default. Either could cost time on a shape that needed none of it. Compare
against the pre-blocking program, same box, same inputs.

Reports the median and the two-sided spread, per common/harness.latency_stats -- a mean over a
bimodal distribution is what this project got wrong before.

Usage: python3 devtools/t5_perf_check.py <platform> [iters]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/workspace/allscan/pto-zeco")

from common.harness import latency_stats            # noqa: E402
from gla.common import make_gla_inputs              # noqa: E402
from gla.implementations.pypto.impl import PyPtoZeCo  # noqa: E402

CASES = [(256, 64, 64, 64), (128, 32, 64, 64), (128, 32, 32, 32)]


def main() -> int:
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print(f"{'shape':<26} {'plan':<16} {'p50 ms':>9} {'p05':>9} {'p95':>9} {'spread':>7}")
    for (L, C, dk, dv) in CASES:
        impl = PyPtoZeCo()
        try:
            impl.build(1, L, C, dk, dv, device_ids=[0], platform=platform)
            Q, K, V, A = make_gla_inputs(1, L, dk, dv, seed=42)
            st = latency_stats(impl.measure(Q, K, V, A, n_iters=iters))
            # getattr: the pre-blocking build() sets no such field, and reading it
            # directly turned the whole "before" leg into AttributeErrors.
            b = getattr(impl, "blocking", None)
            plan = f"{b[0]}x depth{b[1]}" if b else "unblocked"
            flag = "  <-- unreliable" if st["dispersed"] else ""
            print(f"{f'L={L} C={C} d={dk}/{dv}':<26} {plan:<16} "
                  f"{st['p50_ms']:>9.2f} {st['p05_ms']:>9.2f} {st['p95_ms']:>9.2f} "
                  f"{st['spread']:>7.2f}{flag}")
        except Exception as exc:  # noqa: BLE001
            print(f"{f'L={L} C={C} d={dk}/{dv}':<26} ERROR {type(exc).__name__}: {str(exc)[:70]}")
        finally:
            impl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
