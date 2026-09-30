#!/usr/bin/env python3
"""Run PyPtoZeCo at arbitrary (P, L, C, D) and report the max error vs the golden.

Deliberately uses only the public `PyPtoZeCo` API (build/forward/close), so the SAME
script runs against a pre-F3.1 worktree and the current one — which is what makes an
A/B possible for shapes both trees can compile.

Usage: python3 devtools/f31_shape_check.py <device_csv> <platform> <P:L:C:D> [...]
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2]
    specs = [tuple(int(v) for v in s.split(":")) for s in sys.argv[3:]]

    bad = 0
    for (P, L, C, D) in specs:
        tag = f"P={P} L={L} C={C} D={D}"
        if P > len(devices):
            print(f"{tag}: SKIP (needs {P} devices)")
            continue
        torch.manual_seed(P * 100 + L)
        # make_gla_inputs takes (P, L, dk, dv) — NOT the chunk size. Passing C here silently
        # builds dk=C inputs for a dk=D program, which blows up later in gammas.reshape.
        Q, K, V, A = make_gla_inputs(P, L, D, D)
        exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                           flatten_seq(A)).reshape(P, L, D)
        impl = PyPtoZeCo()
        try:
            impl.build(P, L, C, D, D, device_ids=devices[:P], platform=platform)
            err = (impl.forward(Q, K, V, A) - exp).abs().max().item()
        except Exception as exc:  # noqa: BLE001 - a compile/run failure is a result here
            print(f"{tag}: ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            bad += 1
            continue
        finally:
            impl.close()
        ok = err < 1e-2
        bad += 0 if ok else 1
        print(f"{tag}: max diff {err:.4e}  {'OK' if ok else '*** WRONG ***'}")

    print(f"\n{len(specs) - bad}/{len(specs)} OK")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
