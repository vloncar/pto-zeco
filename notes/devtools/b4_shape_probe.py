#!/usr/bin/env python3
"""B4: where does the fused backward's vector-buffer ceiling actually sit?

The backward's `grad_o` is the widest kernel in either direction, so it must top out below
the forward's `C=D=64`. Guessing the ceiling is unnecessary: a Vec-buffer overflow is a
COMPILE-time failure (that is how F3.1's `C=64 needs 200704 B` surfaced), so the reachable
set can be mapped with no NPU at all — which matters on a box where cards are contended.

Reports, per shape, whether the P=1 and P>1 programs compile, and quotes the allocator's
byte figure when they do not, so the result is a budget number rather than "it broke".

Usage: python3 devtools/b4_shape_probe.py [--platform a2a3]
"""

from __future__ import annotations

import argparse
import re
import sys
import time


def try_compile(L, C, dk, dv, P, platform):
    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    from gla.implementations.pypto.fused_backward_program import (
        build_fused_backward_program,
    )
    try:
        prog = build_fused_backward_program(L, C, dk, dv, 1, P)
        ir.compile(prog, platform=platform,
                   distributed_config=DistributedConfig(device_ids=list(range(P)),
                                                        num_sub_workers=0))
        return True, ""
    except Exception as exc:  # noqa: BLE001 - a failure IS the measurement here
        msg = " ".join(str(exc).split())
        # Pull out the allocator's byte figure when there is one: "... 200704 B ..."
        nums = re.findall(r"(\d{5,})", msg)
        short = msg[:150]
        if nums:
            short = f"needs/limit {' vs '.join(nums[:2])} B | {short}"
        return False, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="a2a3")
    args = ap.parse_args()

    # C drives the three [C,C] tiles; D drives the [dk,dv] state tiles. Walk both, plus the
    # rectangles, so the ceiling is attributed to the right axis rather than to "size".
    shapes = [
        (128, 16, 16, 16), (128, 16, 32, 32), (128, 16, 64, 64),
        (128, 32, 32, 32), (128, 32, 64, 32), (128, 32, 32, 64), (128, 32, 64, 64),
        (256, 64, 32, 32), (256, 64, 64, 32), (256, 64, 32, 64), (256, 64, 64, 64),
        (256, 128, 128, 128),
    ]
    print(f"=== fused backward compile ceiling ({args.platform}) — no NPU needed\n")
    print(f"{'L':>5} {'C':>4} {'dk':>4} {'dv':>4} | {'P=1':>6} {'P=2':>6} | note")
    print("-" * 92)
    for (L, C, dk, dv) in shapes:
        row, note = [], ""
        for P in (1, 2):
            t0 = time.time()
            ok, msg = try_compile(L, C, dk, dv, P, args.platform)
            row.append(f"{'ok' if ok else 'FAIL':>6}")
            if not ok and not note:
                note = msg
            del t0
        print(f"{L:>5} {C:>4} {dk:>4} {dv:>4} | {row[0]} {row[1]} | {note[:60]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
