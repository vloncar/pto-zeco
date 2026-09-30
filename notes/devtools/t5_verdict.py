#!/usr/bin/env python3
"""The trustworthy triple for one shape: bytes, wrong-core ops, and a REAL device build.

`ir.compile` returning success means only that the program compiled -- the device kernels are
built later, at prepare(). Twice today a change looked like a win on `ir.compile` bytes alone
and could not actually build. So report all three, and let the build be the verdict.

Usage: python3 devtools/t5_verdict.py C dk dv [blocks] [depth] [L]
  omit blocks/depth to let the plan search choose (then only the build is meaningful).
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
# TLOAD/TSTORE are the GM<->vector-buffer moves the cube core has no instruction for.
# TMULS was in this list and cried wolf: it appears on the cube side in builds that
# compile perfectly well, so counting it reported 1 "bad" op for every healthy shape.
GM_UB = ("TLOAD", "TSTORE")


def spaces(dump: pathlib.Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    fn = "<module>"
    for line in dump.read_text().splitlines():
        m = _FUNC.match(line)
        if m:
            fn = m.group(1)
        for sp, off, size in _MEMREF.findall(line):
            k = (fn, sp.capitalize())
            out[k] = max(out.get(k, 0), int(off) + int(size))
    return out


def cube_ops(cpp: pathlib.Path) -> int:
    src = cpp.read_text().splitlines()
    try:
        s = next(i for i, l in enumerate(src) if "__DAV_CUBE__" in l)
        e = next(i for i, l in enumerate(src) if "#endif // __DAV_CUBE__" in l)
    except StopIteration:
        return 0
    return sum(1 for i in range(s, e) if any(op + "(" in src[i] for op in GM_UB))


def main() -> int:
    C, dk, dv = (int(x) for x in sys.argv[1:4])
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else None
    slot = int(sys.argv[5]) if len(sys.argv) > 5 else None
    L = int(sys.argv[6]) if len(sys.argv) > 6 else 4 * C
    tag = f"C={C} dk={dk} dv={dv}" + (f" {nb}x depth{slot}" if nb else " (search)")

    from pypto import ir
    from gla.implementations.pypto.fused_program import build_fused_forward_program

    dg = "*/passes_dump/*_after_AllocateMemoryAddr.py"
    kg = "*/next_levels/chip_orch_stage2/kernels/aic/gla_stage2_aic.cpp"
    if nb:
        before_d, before_k = set(BUILD.glob(dg)), set(BUILD.glob(kg))
        try:
            ir.compile(build_fused_forward_program(L, C, dk, dv, 1, 1, nb, 1, slot), platform="a2a3")
        except Exception as exc:  # noqa: BLE001
            m = re.search(r"(\w+) buffer usage \((\d+) bytes\)", str(exc))
            print(f"{tag}: ir.compile OVER -> {m.group(1)} {m.group(2)}B" if m
                  else f"{tag}: {type(exc).__name__}: {str(exc)[:90]}")
            return 1
        fd = set(BUILD.glob(dg)) - before_d
        fk = set(BUILD.glob(kg)) - before_k
        parts = ""
        if fd:
            u = spaces(max(fd, key=lambda p: p.stat().st_size))
            parts = "  ".join(f"{sp} {b}B" for (fn, sp), b in sorted(u.items())
                              if fn == "gla_stage2_aiv" or (fn == "gla_stage2_aic" and sp != "Ddr"))
        bad = cube_ops(max(fk, key=lambda p: p.stat().st_mtime)) if fk else -1
        print(f"{tag}: ir.compile FITS | wrong-core ops {bad} | {parts}")

    # the verdict: a real device build
    from gla.implementations.pypto.impl import PyPtoZeCo
    impl = PyPtoZeCo()
    try:
        impl.build(1, L, C, dk, dv, device_ids=[0], platform="a2a3")
        print(f"{tag}: DEVICE BUILD OK (plan {impl.blocking})")
        return 0
    except Exception as exc:  # noqa: BLE001
        errs = [l.strip() for l in str(exc).splitlines() if "error:" in l and "warning" not in l]
        print(f"{tag}: DEVICE BUILD FAILED :: {(errs[0][:100] if errs else str(exc)[:100])}")
        return 2
    finally:
        impl.close()


if __name__ == "__main__":
    raise SystemExit(main())
