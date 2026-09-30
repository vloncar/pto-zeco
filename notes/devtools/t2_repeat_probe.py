#!/usr/bin/env python3
"""Task 2: dispatch one config many times and count how often it is wrong.

The failure is nondeterministic -- within a single build, dispatches alternate between exact
(~3.8e-05) and catastrophic (~1.9e+02). A single pytest run therefore proves nothing about
either the trigger or a candidate fix, which is how the first A/B misled. This builds ONCE and
dispatches N times, reporting the failure rate.

Usage: python3 devtools/t2_repeat_probe.py <device_csv> <platform> <N> <P> <L> <C> <dk> <dv>
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2]
    n = int(sys.argv[3])
    P, L, C, dk, dv = (int(x) for x in sys.argv[4:9])

    torch.manual_seed(1234)
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                       flatten_seq(A)).reshape(P, L, dv)

    impl = PyPtoZeCo()
    diffs = []
    try:
        impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform=platform)
        for _ in range(n):
            O = impl.forward(Q, K, V, A)
            diffs.append((O - exp).abs().max().item())
    except Exception as exc:  # noqa: BLE001 - a failure is a result here
        print(f"ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        return 3
    finally:
        impl.close()

    bad = [d for d in diffs if d >= 1e-2]
    print(f"RESULT P={P} L={L} C={C} dk={dk} dv={dv}: "
          f"{len(bad)}/{n} dispatches WRONG")
    print("  diffs: " + " ".join(f"{d:.2e}" for d in diffs))
    if bad:
        print(f"  worst {max(bad):.4e}   best-bad {min(bad):.4e}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
