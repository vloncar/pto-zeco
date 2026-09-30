#!/usr/bin/env python3
"""Run one PyPTO ZeCO forward shape end to end against the torch golden.

Fast single-shape gate for the A1 rewrite: build + dispatch + compare, no pytest.
Usage: a1_case.py <platform> <P> <L> <C> <dk> <dv> [device_ids csv] [repeats]
"""
import os
import sys

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

# ZECO_FORCE_PLAN="nb,nv,slot" pins the blocking instead of letting the search pick it, so a
# split that a small shape would never choose on its own can still be exercised on hardware.
_FORCE = os.environ.get("ZECO_FORCE_PLAN")
if _FORCE:
    import gla.implementations.pypto.fused_program as _fp
    _plan = tuple(int(x) for x in _FORCE.split(","))
    _fp.blocking_plans = lambda C, dk, dv, distributed=False, _p=_plan: [_p]


def main():
    platform = sys.argv[1]
    P, L, C, dk, dv = (int(x) for x in sys.argv[2:7])
    devs = [int(x) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else list(range(P))
    repeats = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=devs[:P], platform=platform)
    worst, worst_seed = 0.0, 42
    try:
        for i in range(repeats):
            s = 42 + i
            Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=s)
            g = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                             flatten_seq(A)).reshape(P, L, dv)
            err = (impl.forward(Q, K, V, A) - g).abs().max().item()
            if err > worst:
                worst, worst_seed = err, s
    finally:
        impl.close()
    ok = worst < 1e-2
    print(f"P={P} L={L} C={C} dk={dk} dv={dv} blocking={getattr(impl, 'blocking', None)} "
          f"worst={worst:.3e} seed={worst_seed} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
