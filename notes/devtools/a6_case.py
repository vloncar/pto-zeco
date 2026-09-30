#!/usr/bin/env python3
"""Run one PyPTO ZeCO BACKWARD shape end to end against the analytic golden.

Fast single-shape gate for the A6 rewrite: build + dispatch + compare, no pytest.
Usage: a6_case.py <platform> <P> <L> <C> <dk> <dv> [device_ids csv] [repeats]

ZECO_FORCE_PLAN="nb,nv,slot,rb,nc" pins the blocking instead of letting the search pick it,
so a split a small shape would never choose on its own can still be exercised on hardware --
which is how the value/key-row splits get tested for BIT-IDENTITY against the unsplit answer
rather than merely for "it fits".
"""
import os
import sys

import torch

from gla.common import expected_gla_backward, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

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
    worst, worst_name, worst_seed = 0.0, "", 42
    try:
        for i in range(repeats):
            s = 42 + i
            Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=s)
            torch.manual_seed(s + 991)
            dO = torch.randn(P, L, dv, dtype=torch.float32)
            got = impl.backward(Q, K, V, A, dO)
            ref = expected_gla_backward(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                                        flatten_seq(A), flatten_seq(dO))
            for nm, g, r in zip(("dQ", "dK", "dV", "dA"), got, ref):
                r = r.reshape(g.shape)
                rel = ((g - r).abs().max() / (r.abs().max() + 1e-6)).item()
                if rel > worst:
                    worst, worst_name, worst_seed = rel, nm, s
    finally:
        impl.close()
    ok = worst < 1e-3
    print(f"P={P} L={L} C={C} dk={dk} dv={dv} blocking={getattr(impl, 'bblocking', None)} "
          f"worst={worst:.3e} on {worst_name} seed={worst_seed} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
