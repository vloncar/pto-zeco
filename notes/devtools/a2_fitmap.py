#!/usr/bin/env python3
"""Which shapes does the plan search find a fit for? ir.compile ONLY -- not a device build.

A fit here means the tiles are placeable; it is NOT proof the device kernel builds (that
happens at prepare(), and has caught designs this stage called clean). Use it to map the
frontier, then verify the headline shapes on hardware.
"""
import contextlib
import io
import sys

from pypto import ir

from gla.implementations.pypto.fused_program import blocking_plans, build_fused_forward_program

CASES = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]] or [(64, 128, 128)]
for C, dk, dv in CASES:
    L = 4 * C
    why, found = "", None
    for nb, nv, slot in blocking_plans(dk, dv):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ir.compile(build_fused_forward_program(L, C, dk, dv, 1, 1, nb, nv, slot),
                           platform="a2a3")
            found = (nb, nv, slot)
            break
        except Exception as e:  # noqa: BLE001 -- reporting, not handling
            s = str(e)
            if "buffer usage" not in s:
                why = f"{type(e).__name__}: {s[:100]}"
                break
            why = [w for w in s.split("\n") if "buffer usage" in w][0].split("exceeds")[0].strip()
    print(f"C={C:<4} dk={dk:<5} dv={dv:<5} -> " +
          (f"FITS at head_blocks={found[0]} value_blocks={found[1]} ring_depth={found[2]}"
           if found else f"no fit ({why})"), flush=True)
print("NOTE: ir.compile only -- a fit here is not proof the device kernel builds.")
