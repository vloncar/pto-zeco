#!/usr/bin/env python3
"""Task 5: do the shapes the blocking UNLOCKS actually give the right answer on hardware?

The regression suite proves nothing was broken. This proves something was gained: head dim
128 on either side or both, plus a head dim that is a multiple of 16 without being a power of
two. Reports the blocking each shape landed on, so the search's choice is visible rather than
implied.

Usage: python3 devtools/t5_newshapes.py <platform> [P]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/workspace/allscan/pto-zeco")

from gla.common import make_gla_inputs, expected_gla, flatten_seq   # noqa: E402
from gla.implementations.pypto.impl import PyPtoZeCo                # noqa: E402

# (L, C, dk, dv) -- all previously refused by the vector budget, except the last two which
# check that an odd-but-legal head dim works and that the old ceiling still does.
CASES = [
    (256, 64, 128,  64),   # keys 128
    (256, 64,  64, 128),   # values 128
    (256, 64, 128, 128),   # both 128 -- the F6.5 target
    (192, 48,  48,  48),   # multiple of 16, not a power of two
    (256, 64,  64,  64),   # control: the old ceiling, must still be right
]


def golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                        flatten_seq(A)).reshape(P, L, dv)


def main() -> int:
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    device_ids = [0, 1, 2, 3][:P]
    bad = 0
    print(f"{'shape':<30} {'blocking':<22} result")
    for (L, C, dk, dv) in CASES:
        tag = f"L={L} C={C} dk={dk} dv={dv}"
        impl = PyPtoZeCo()
        try:
            impl.build(P, L, C, dk, dv, device_ids=device_ids, platform=platform)
            plan = f"blocks={impl.blocking[0]} depth={impl.blocking[1]}"
            worst = 0.0
            for s in (42, 43, 44):
                Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=s)
                worst = max(worst, (impl.forward(Q, K, V, A) - golden(Q, K, V, A)).abs().max().item())
            ok = worst < 1e-2
            bad += 0 if ok else 1
            print(f"{tag:<30} {plan:<22} {'PASS' if ok else 'FAIL'}  err={worst:.3e}")
        except Exception as exc:  # noqa: BLE001 - the point is to report, not to stop
            bad += 1
            print(f"{tag:<30} {'-':<22} ERROR {type(exc).__name__}: {str(exc)[:120]}")
        finally:
            impl.close()
    print(f"\n{len(CASES) - bad}/{len(CASES)} shapes correct at P={P}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
