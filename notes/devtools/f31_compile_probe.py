#!/usr/bin/env python3
"""F3.1 — map the pypto fused-forward shape ceiling and report the buffer headroom.

Compile-only (no device), so this runs anywhere. For each (L, C, dk, dv) it compiles the
fused forward and reports, per generated kernel, the peak usage of each on-core memory
space against the platform limit:

  Vec    188416 B   vector buffer (holds the whole per-chunk working set)
  Left    65536 B   matmul A operand (L0A)
  Right   65536 B   matmul B operand (L0B)

On failure the compiler names the offending function and its byte count directly. On
success the numbers come from the ``33_after_AllocateMemoryAddr`` pass dump: every tile
carries ``pl.MemRef(<ptr>, <offset>, <size>)``, so peak usage per space is
``max(offset + size)``. That is what tells you how much room the *next* size increase has
— the failure message alone only ever reports the legacy (non-reused) packing, which
badly overstates what a passing config actually occupies.

Usage: python3 devtools/f31_compile_probe.py [platform] [P]
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

# (L, C, dk, dv) — the corners that bound the reachable region.
CASES = [
    (128, 32, 32, 32),
    (128, 32, 64, 64),
    (128, 64, 32, 32),
    (256, 64, 64, 64),
    (256, 64, 128, 128),
    (256, 128, 64, 64),
    (256, 128, 128, 128),
]

LIMITS = {"Vec": 188416, "Left": 65536, "Right": 65536}
# pypto writes build_output/ under the CURRENT WORKING DIRECTORY, not next to this file.
# This used to be `__file__/../../build_output`, which was the same place only while devtools
# still lived inside the repo; after the move it pointed at an empty directory and every row
# reported no dump.
BUILD_DIR = pathlib.Path(os.environ.get("PTO_BUILD_DIR", pathlib.Path.cwd() / "build_output"))

_MEMREF = re.compile(r"pl\.MemRef\(mem_(\w+?)_\d+, pl\.const\((\d+), pl\.INT64\), (\d+)\)")
_FUNC = re.compile(r"^\s*def (\w+)\(")


def peak_usage(dump: pathlib.Path) -> dict[str, dict[str, int]]:
    """Peak bytes per (function, memory space) = max(offset + size) over its tiles."""
    out: dict[str, dict[str, int]] = {}
    fn = "<module>"
    for line in dump.read_text().splitlines():
        m = _FUNC.match(line)
        if m:
            fn = m.group(1)
        for space, off, size in _MEMREF.findall(line):
            key = space.capitalize()
            if key in LIMITS:
                cur = out.setdefault(fn, {})
                cur[key] = max(cur.get(key, 0), int(off) + int(size))
    return out


# Match the pass by NAME, not by its number: upstream inserted two passes and
# `33_after_AllocateMemoryAddr` became `35_after_AllocateMemoryAddr`, at which point a
# literal-number glob found nothing and every row printed a bare "PASS" with no numbers --
# a probe that had gone blind but still looked like it was working.
def dump_dirs() -> set[pathlib.Path]:
    return set(BUILD_DIR.glob("*/passes_dump/*_after_AllocateMemoryAddr.py"))


def new_dump(before: set[pathlib.Path]) -> pathlib.Path | None:
    """The dump this compile produced.

    Deliberately "whatever appeared since `before`", not "newest by mtime": one compile can
    emit several program directories, and picking by mtime silently read a *different*
    config's dump (a 135424 B stage2 was reported as 10304 B that way). If a compile somehow
    yields more than one, take the largest — the outer program, not an inner one.
    """
    fresh = dump_dirs() - before
    if not fresh:
        return None
    return max(fresh, key=lambda p: p.stat().st_size)


def main() -> int:
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    from gla.implementations.pypto.fused_program import build_fused_forward_program

    for (L, C, dk, dv) in CASES:
        tag = f"L={L:>3} C={C:>3} dk={dk:>3} dv={dv:>3}"
        before = dump_dirs()
        try:
            program = build_fused_forward_program(L, C, dk, dv, 1, P)
            cfg = DistributedConfig(device_ids=list(range(P)), num_sub_workers=0)
            ir.compile(program, platform=platform, distributed_config=cfg)
        except Exception as exc:  # noqa: BLE001 - probing the ceiling is the point
            print(f"{tag}  FAIL")
            for msg in re.findall(r"Message: (.*)", str(exc)):
                print(f"     {msg}")
            continue

        dump = new_dump(before)
        if dump is None:
            print(f"{tag}  PASS   !! no AllocateMemoryAddr dump found under {BUILD_DIR} -- "
                  "numbers unavailable, do NOT read this row as headroom")
            continue
        usage = peak_usage(dump)
        worst = []
        for fn in sorted(usage):
            if not fn.startswith("gla_"):
                continue
            for space, used in sorted(usage[fn].items()):
                pct = 100.0 * used / LIMITS[space]
                worst.append(f"{fn}.{space} {used}B ({pct:.0f}%)")
        if not worst:
            print(f"{tag}  PASS   !! dump {dump.name} held no gla_* tiles -- "
                  "numbers unavailable, do NOT read this row as headroom")
            continue
        print(f"{tag}  PASS   " + "  ".join(worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
