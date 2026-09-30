#!/usr/bin/env python3
"""Task 2: separate the three confounds behind the inconsistent reproduction.

Same shape (C=64, dk=32, dv=64) has now both failed 3/3 (axis matrix) and passed 12/12
(repeat probe) on the stock header. Three differences between those runs:

  1. INPUTS   the pytest harness `_run_case` does NOT seed -- every run gets fresh random
              data. The probes seeded 1234. So the failure may be input-dependent.
  2. HISTORY  the matrix built and closed other configs in the same process first (a canary,
              then earlier shapes); the repeat probe built once in a clean process.
  3. CARD     different jobs got different physical cards.

This varies (1) and (2) explicitly within one job on one card, so whatever is left is (3).

  mode=seeded     : fixed seed, no prior build
  mode=random     : fresh random inputs each dispatch, no prior build
  mode=warm-canary: fixed seed, but a canary build+close first (mimics the matrix)
  mode=warm-same  : fixed seed, but a build+close of the SAME shape first

Usage: python3 devtools/t2_confound_probe.py <device_csv> <platform> <N> <mode>
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

P, L, C, DK, DV = 1, 128, 64, 32, 64


def _golden(Q, K, V, A, dv):
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                        flatten_seq(A)).reshape(-1, L, dv)


def _build_and_close(devices, platform, p, l, c, dk, dv):
    """Stand a config up and tear it down, to leave whatever state that leaves."""
    impl = PyPtoZeCo()
    try:
        impl.build(p, l, c, dk, dv, device_ids=devices[:p], platform=platform)
        Q, K, V, A = make_gla_inputs(p, l, dk, dv)
        impl.forward(Q, K, V, A)
    finally:
        impl.close()


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2]
    n = int(sys.argv[3])
    mode = sys.argv[4]

    if mode == "warm-canary":
        _build_and_close(devices, platform, 1, 32, 16, 16, 16)
    elif mode == "warm-same":
        _build_and_close(devices, platform, P, L, C, DK, DV)

    seeded = mode != "random"
    if seeded:
        torch.manual_seed(1234)

    impl = PyPtoZeCo()
    diffs = []
    try:
        impl.build(P, L, C, DK, DV, device_ids=devices[:P], platform=platform)
        for _ in range(n):
            # `random` regenerates inputs per dispatch, like the unseeded pytest harness.
            Q, K, V, A = make_gla_inputs(P, L, DK, DV)
            exp = _golden(Q, K, V, A, DV)
            O = impl.forward(Q, K, V, A)
            diffs.append((O - exp).abs().max().item())
    except Exception as exc:  # noqa: BLE001 - a failure is a result here
        print(f"MODE {mode}: ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:140]}")
        return 3
    finally:
        impl.close()

    bad = [d for d in diffs if d >= 1e-2]
    print(f"MODE {mode:12s}: {len(bad)}/{n} WRONG   "
          + " ".join(f"{d:.2e}" for d in diffs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
