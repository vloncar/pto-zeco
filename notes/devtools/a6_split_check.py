#!/usr/bin/env python3
"""A6: which of the backward's kernels put VECTOR work on the CUBE core?

`ir.compile` returning success does not mean the kernel builds: the device kernels are built
at prepare(), and a vector op emitted into the cube half of a mixed kernel dies there with
"'set_vector_mask' ... does not support the given target feature". a2a3sim does not catch it
either. But ir.compile WRITES the generated .cpp, so the defect is visible statically.

The test is an ALLOW-list, not a deny-list, and it is measured rather than guessed: a build
that runs on hardware emits only ``TASSIGN`` / ``TMOV`` / ``TMATMUL`` / ``TLOAD`` inside
``#if defined(__DAV_CUBE__)`` (the ``TLOAD`` being a matmul operand staged into L1, which is
cube work by rights). A plan that fails to build adds a ``TMULS``. Anything outside the list
is therefore reported, which also catches the GM->GM copy landing on the cube.

Tokens are matched exactly -- ``TMATMUL`` contains ``TMUL`` as a substring, and a deny-list
built with ``in`` flags every matmul in the program.

Usage: a6_split_check.py <plan nb,nv,slot,rb,nc> [L C dk dv] [platform]
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

sys.path.insert(0, "/root/workspace/allscan/pto-zeco")

BUILD = pathlib.Path("/root/workspace/allscan/pto-zeco/build_output")
CUBE_OK = {"TASSIGN", "TMOV", "TMATMUL", "TLOAD", "TFREE", "TPUSH", "TPOP", "TSTORE"}
VEC_TILE = "TileType::Vec"   # the real test: UB tiles the cube cannot address
TOKEN = re.compile(r"\b(T[A-Z_0-9]+)\(")


def cube_ops(cpp: pathlib.Path) -> collections.Counter:
    src = cpp.read_text().splitlines()
    try:
        s = next(i for i, l in enumerate(src) if "__DAV_CUBE__" in l)
        e = next(i for i, l in enumerate(src) if "#endif // __DAV_CUBE__" in l)
    except StopIteration:
        return collections.Counter()
    return collections.Counter(m for i in range(s, e) for m in TOKEN.findall(src[i]))


def main() -> int:
    plan = tuple(int(x) for x in sys.argv[1].split(","))
    L, C, dk, dv = ((int(x) for x in sys.argv[2:6]) if len(sys.argv) > 5 else (128, 32, 32, 32))
    platform = sys.argv[6] if len(sys.argv) > 6 else "a2a3"
    from pypto import ir

    from gla.implementations.pypto.fused_backward_program import build_fused_backward_program

    nb, nv, slot, rb, nc = plan
    before = set(BUILD.glob("*/next_levels/*/kernels/aic/*.cpp"))
    ir.compile(build_fused_backward_program(L, C, dk, dv, rb, 1, nb, nv, slot, nc),
               platform=platform)
    fresh = sorted(set(BUILD.glob("*/next_levels/*/kernels/aic/*.cpp")) - before)
    if not fresh:
        print("no freshly generated cube kernels found")
        return 2
    total = 0
    for cpp in fresh:
        ops = cube_ops(cpp)
        lines = cpp.read_text().splitlines()
        s = next(i for i, l in enumerate(lines) if "__DAV_CUBE__" in l)
        e = next(i for i, l in enumerate(lines) if "#endif // __DAV_CUBE__" in l)
        vec = [lines[i - 1].strip()[9:].strip() for i in range(s, e) if VEC_TILE in lines[i]]
        odd = {k: v for k, v in ops.items() if k not in CUBE_OK}
        total += len(vec)
        print(f"{cpp.name:<28} {'BAD ' + str(vec) if vec else 'clean':<40} "
              f"ops={dict(ops)}{' odd=' + str(odd) if odd else ''}")
    print(f"TOTAL cube-side vector tiles: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
