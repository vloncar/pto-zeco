#!/usr/bin/env python3
"""A3: what exactly stops C=128, and by how much, across blockings?

The plan search reports only the FIRST buffer to overflow, so it says "Left" and stops. To
choose between narrower operands and an exact restructuring, we need the whole picture: how
far over each space is, at the settings that come closest.

Compile-only. Reports the compiler's own overflow number for each attempt.
Usage: a3_c128_budget.py [C] [dk] [dv]
"""
from __future__ import annotations

import contextlib
import io
import re
import sys

from pypto import ir

from gla.implementations.pypto.fused_program import _splits, build_fused_forward_program

LIMITS = {"Vec": 188416, "Left": 65536, "Right": 65536, "Acc": 65536, "Mat": 524288}
_OVER = re.compile(r"(\w+) buffer usage \((\d+) bytes\) exceeds platform limit \((\d+) bytes\)")


def main():
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    dk = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    dv = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    L = 4 * C
    print(f"C={C} dk={dk} dv={dv}   limits: " +
          "  ".join(f"{k}={v}" for k, v in LIMITS.items()))
    print(f"  {'head':>4} {'val':>4} {'ring':>4}  result")
    best = None
    for nb in _splits(dk):
        for nv in _splits(dv):
            for slot in (2, 1):
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        ir.compile(build_fused_forward_program(L, C, dk, dv, 1, 1, nb, nv, slot),
                                   platform="a2a3")
                    print(f"  {nb:>4} {nv:>4} {slot:>4}  FITS")
                    return 0
                except Exception as e:  # noqa: BLE001 -- reporting
                    m = _OVER.search(str(e))
                    if not m:
                        print(f"  {nb:>4} {nv:>4} {slot:>4}  {type(e).__name__}: {str(e)[:70]}")
                        continue
                    space, used, lim = m.group(1), int(m.group(2)), int(m.group(3))
                    over = used - lim
                    print(f"  {nb:>4} {nv:>4} {slot:>4}  {space:<6} {used:>7} B "
                          f"= {used / lim:>5.2f}x limit, over by {over} B")
                    if best is None or over < best[0]:
                        best = (over, space, nb, nv, slot)
    if best:
        over, space, nb, nv, slot = best
        print(f"\n  closest: {space} over by {over} B at head={nb} value={nv} ring_depth={slot}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
