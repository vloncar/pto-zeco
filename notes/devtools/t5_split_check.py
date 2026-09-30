#!/usr/bin/env python3
"""Which shapes put global<->vector memory traffic on the CUBE core?

The cube core has no vector buffer, so a TLOAD/TSTORE emitted into its half of a mixed kernel
cannot build: "copy_gm_to_ubuf_align_b32 does not support the given target feature". That
failure only appears when the DEVICE kernels are compiled, which happens at prepare() -- long
after ir.compile has returned success -- so it is invisible to anything that only compiles the
program. It is also invisible on a2a3sim.

The generated .cpp is written by ir.compile though, so the same defect is detectable
statically: bracket the `#if defined(__DAV_CUBE__)` region and look for the ops.

Usage: python3 devtools/t5_split_check.py [platform]
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "/root/workspace/allscan/pto-zeco")

BUILD = pathlib.Path("/root/workspace/allscan/pto-zeco/build_output")
GM_UB_OPS = ("TLOAD", "TSTORE", "TMULS")


def cube_gm_ops(cpp: pathlib.Path) -> list[str]:
    src = cpp.read_text().splitlines()
    try:
        s = next(i for i, l in enumerate(src) if "__DAV_CUBE__" in l)
        e = next(i for i, l in enumerate(src) if "#endif // __DAV_CUBE__" in l)
    except StopIteration:
        return []
    return [src[i].strip()[:70] for i in range(s, e)
            if any(op + "(" in src[i] for op in GM_UB_OPS)]


def main() -> int:
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    from pypto import ir
    from gla.implementations.pypto.fused_program import build_fused_forward_program

    cases = [(256, 64, 64, 64), (256, 64, 128, 64), (256, 64, 64, 128), (256, 64, 128, 128),
             (128, 16, 32, 16), (192, 48, 48, 48)]
    plans = [(1, 2), (2, 2), (4, 1), (8, 1)]
    print(f"{'shape':<26} {'plan':<12} {'cube GM<->UB ops':<18} verdict")
    for (L, C, dk, dv) in cases:
        for nb, slot in plans:
            if dk % nb or (dk // nb) % 16:
                continue
            before = set(BUILD.glob("*/next_levels/chip_orch_stage2/kernels/aic/gla_stage2_aic.cpp"))
            try:
                ir.compile(build_fused_forward_program(L, C, dk, dv, 1, 1, nb, 1, slot),
                           platform=platform)
            except Exception as exc:  # noqa: BLE001
                why = "too big" if "exceeds platform limit" in str(exc) else type(exc).__name__
                print(f"{f'C={C} dk={dk} dv={dv}':<26} {f'{nb}x depth{slot}':<12} {'-':<18} {why}")
                continue
            fresh = set(BUILD.glob("*/next_levels/chip_orch_stage2/kernels/aic/gla_stage2_aic.cpp")) - before
            if not fresh:
                print(f"{f'C={C} dk={dk} dv={dv}':<26} {f'{nb}x depth{slot}':<12} {'?':<18} no generated kernel found")
                continue
            ops = cube_gm_ops(max(fresh, key=lambda p: p.stat().st_mtime))
            verdict = "OK" if not ops else "WILL NOT BUILD ON HW"
            print(f"{f'C={C} dk={dk} dv={dv}':<26} {f'{nb}x depth{slot}':<12} {len(ops):<18} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
