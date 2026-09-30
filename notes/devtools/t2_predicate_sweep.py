#!/usr/bin/env python3
"""Task 2: pin the exact shape predicate, with enough dispatches to trust a clean verdict.

The failure is intermittent -- observed per-process rates run from 0/6 to 10/10. At a 1-in-6
rate three repeats miss it 57% of the time, so the earlier 3-repeat matrix cannot rule any
shape clean. This dispatches N times (default 20) and draws FRESH inputs every dispatch, which
maximises the chance of hitting the failure: the question here is "does this shape EVER
corrupt", not "what is its exact rate".

One config per process (the driver forks per line), so no in-process build history can carry.

Usage: python3 devtools/t2_predicate_sweep.py <device_csv> <platform> <N> <P> <L> <C> <dk> <dv>
"""

from __future__ import annotations

import sys

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2]
    n = int(sys.argv[3])
    P, L, C, dk, dv = (int(x) for x in sys.argv[4:9])

    rel_dk = "dk<C" if dk < C else ("dk=C" if dk == C else "dk>C")
    rel_dv = "dv<C" if dv < C else ("dv=C" if dv == C else "dv>C")
    tag = f"C={C:3d} dk={dk:3d} dv={dv:3d} L={L:3d} P={P}  [{rel_dk},{rel_dv}]"

    impl = PyPtoZeCo()
    diffs = []
    try:
        impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform=platform)
        for _ in range(n):
            Q, K, V, A = make_gla_inputs(P, L, dk, dv)
            exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                               flatten_seq(A)).reshape(P, L, dv)
            O = impl.forward(Q, K, V, A)
            diffs.append((O - exp).abs().max().item())
    except AssertionError as exc:
        print(f"SWEEP {tag}  GUARDED ({str(exc).splitlines()[0][:60]})")
        return 0
    except Exception as exc:  # noqa: BLE001 - a failure is a result here
        print(f"SWEEP {tag}  ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
        return 3
    finally:
        impl.close()

    bad = [d for d in diffs if d >= 1e-2]
    verdict = "*** WRONG ***" if bad else "clean"
    extra = f"  worst {max(bad):.3e}" if bad else f"  max {max(diffs):.2e}"
    print(f"SWEEP {tag}  {len(bad):2d}/{n} wrong  {verdict}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
