#!/usr/bin/env python3
"""A6: does the blocked backward compile, and at which blocking does each shape land?

A Vec-buffer overflow is a COMPILE-time failure, so the reachable set can be mapped with no
NPU at all -- which matters on a box whose cards are contended. Reports the plan the search
settles on (or the byte figure when nothing fits), which is both the ceiling map and a check
that the search is not quietly picking a finer blocking than the shape needs.

Usage: a6_compile_probe.py [--platform a2a3] [--shapes L,C,dk,dv;...] [--p 1,2]
"""
from __future__ import annotations

import argparse
import re
import sys
import time


def try_compile(L, C, dk, dv, P, platform, verbose=False):
    from pypto.ir.distributed_compiled_program import DistributedConfig

    from gla.implementations.pypto.fused_backward_program import compile_fused_backward
    notes = []
    try:
        _, plan = compile_fused_backward(
            L, C, dk, dv, P, platform=platform,
            distributed_config=DistributedConfig(device_ids=list(range(P)), num_sub_workers=0),
            log=notes.append if verbose else None)
        if verbose:
            for line in notes:
                print(f"      {line}", flush=True)
        return True, "head=%d value=%d ring_depth=%d ring=%d key_row=%d" % plan
    except Exception as exc:  # noqa: BLE001 - a failure IS the measurement here
        msg = " ".join(str(exc).split())
        if verbose:
            for line in notes:
                print(f"      {line}", flush=True)
            print("      " + msg[:1500], flush=True)
        nums = re.findall(r"(\d{5,})", msg)
        return False, (f"needs {nums[0]} B" if nums else msg[:70])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="a2a3")
    ap.add_argument("--p", default="1")
    ap.add_argument("--verbose", action="store_true",
                    help="print why each plan was rejected -- buffer overflow or cube placement")
    ap.add_argument("--shapes", default=(
        "128,16,16,16;128,32,32,32;128,32,64,32;128,32,32,64;128,32,16,32;"
        "256,64,64,64;256,64,128,128;512,128,128,128"))
    args = ap.parse_args()

    ranks = [int(x) for x in args.p.split(",")]
    print(f"{'L':>5} {'C':>4} {'dk':>4} {'dv':>4} {'P':>2} | {'':>4} | plan / why", flush=True)
    rc = 0
    for spec in args.shapes.split(";"):
        L, C, dk, dv = (int(x) for x in spec.split(","))
        for P in ranks:
            t0 = time.time()
            ok, note = try_compile(L, C, dk, dv, P, args.platform, args.verbose)
            print(f"{L:>5} {C:>4} {dk:>4} {dv:>4} {P:>2} | {'ok' if ok else 'FAIL':>4} | "
                  f"{note}   ({time.time() - t0:.0f}s)", flush=True)
            rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
