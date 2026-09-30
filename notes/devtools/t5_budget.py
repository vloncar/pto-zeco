#!/usr/bin/env python3
"""Peak vector-buffer bytes per kernel, for one shape and one blocking plan.

Reports the number for a plan that FITS (from the AllocateMemoryAddr dump, which reflects the
real reuse-aware packing) or the compiler's figure for one that does not (the legacy
non-reused packing, which overstates by ~25-35% -- never size a shape from it).

Usage: python3 devtools/t5_budget.py C dk dv [blocks] [depth] [L]
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

sys.path.insert(0, "/root/workspace/allscan/pto-zeco")

LIMIT = 188416
BUILD = pathlib.Path(os.environ.get("PTO_BUILD_DIR", pathlib.Path.cwd() / "build_output"))
_MEMREF = re.compile(r"pl\.MemRef\(mem_(\w+?)_\d+, pl\.const\((\d+), pl\.INT64\), (\d+)\)")
_FUNC = re.compile(r"^\s*def (\w+)\(")


def peak(dump: pathlib.Path) -> dict[str, int]:
    out: dict[str, int] = {}
    fn = "<module>"
    for line in dump.read_text().splitlines():
        m = _FUNC.match(line)
        if m:
            fn = m.group(1)
        for space, off, size in _MEMREF.findall(line):
            if space.capitalize() == "Vec":
                out[fn] = max(out.get(fn, 0), int(off) + int(size))
    return out


def main() -> int:
    C, dk, dv = (int(x) for x in sys.argv[1:4])
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    slot = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    L = int(sys.argv[6]) if len(sys.argv) > 6 else 4 * C

    from pypto import ir
    from gla.implementations.pypto.fused_program import build_fused_forward_program

    g = "*/passes_dump/*_after_AllocateMemoryAddr.py"
    before = set(BUILD.glob(g))
    try:
        ir.compile(build_fused_forward_program(L, C, dk, dv, 1, 1, nb, 1, slot), platform="a2a3")
    except Exception as exc:  # noqa: BLE001
        m = re.search(r"Vec buffer usage \((\d+) bytes\)", str(exc))
        if m:
            print(f"C={C} dk={dk} dv={dv} blocks={nb} depth={slot}: "
                  f"OVER by {int(m.group(1)) - LIMIT} ({m.group(1)} vs {LIMIT}, legacy packing)")
            return 1
        print(f"C={C} dk={dk} dv={dv} blocks={nb} depth={slot}: {type(exc).__name__}: {str(exc)[:110]}")
        return 2
    fresh = set(BUILD.glob(g)) - before
    if not fresh:
        print(f"C={C} dk={dk} dv={dv} blocks={nb} depth={slot}: FITS (no dump found -- no bytes)")
        return 0
    u = peak(max(fresh, key=lambda p: p.stat().st_size))
    parts = "  ".join(f"{fn} {b}B ({100.0 * b / LIMIT:.0f}%)"
                      for fn, b in sorted(u.items()) if fn.startswith("gla_") and b)
    print(f"C={C} dk={dk} dv={dv} blocks={nb} depth={slot}: FITS   {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
