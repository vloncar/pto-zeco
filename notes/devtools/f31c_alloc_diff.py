#!/usr/bin/env python3
"""F3.1c: does the allocator place a layout-changing op's source and destination at the
SAME address, for the shapes that fail on hardware?

Compile-only (no device). Compiles ONE shape per invocation, in a private working
directory, and reports the buffer address of every tile in the two layout-changing chains
of ``gla_stage2``:

  * ``gamma``:  ``col_sum(la)[1, DK]`` -> ``reshape -> [DK, 1]`` -> ``exp``
  * ``kbt``:    ``transpose(kb)``

Both rewrite a tile's *layout*, not just its values: ``[1, N] -> [N, 1]`` moves element i
from byte 4i to byte i*row_stride, and a transpose moves (r, c) to (c, r). Neither is safe
in place — a scatter overwrites source elements before they are read — so if MemoryReuse
assigns the source and the destination the same address, the result is silently wrong.
That would be invisible to a2a3sim (which models tiles as logical arrays, not as bytes in
a reused buffer) and would depend on the exact shape, which is the signature this bug has.

These chains are new in F3.1: ``col_sum`` + ``reshape`` + ``row_expand_mul`` replaced the
old matmul-by-ones broadcast, and ``kbt`` became a shared tile. So they are the F3.1-
introduced code the failure has to be explained by.

IMPORTANT: run one config per process into a private cwd. pypto writes ``build_output/``
relative to the cwd, and identifying "this compile's dump" by set-difference is wrong the
moment any other pypto process (an HW test run, say) is compiling at the same time.

Usage: python3 devtools/f31c_alloc_diff.py <L> <C> <dk> <dv> [platform]
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import tempfile

_TILE = re.compile(
    r"^\s*(\w+): pl\.Tile\[\[([0-9, ]+)\][^]]*?pl\.MemRef\((mem_\w+), "
    r"pl\.const\((\d+), pl\.INT64\), (\d+)\)"
)
_ASSIGN = re.compile(r"=\s*pl\.tile\.(\w+)\(([^)]*)")
_FUNC = re.compile(r"^\s*def (\w+)\(")
_PIPE = re.compile(r"pl\.system\.(ai[cv])_initialize_pipe\((.*)\)")
_RESERVE = re.compile(r'pl\.system\.reserve_buffer\(name="(\w+)", size=(\d+)')
_XFER = re.compile(r"pl\.(?:tile|system)\.(tpush_to_ai[cv]|tpop_from_ai[cv]|tfree_to_ai[cv])\(([^)]*)")

# The ops that rewrite layout rather than just values.
LAYOUT_OPS = {"reshape", "transpose", "col_sum", "row_sum"}


def pipe_geometry(dump: pathlib.Path) -> None:
    """Report the cross-core pipe geometry against the tiles actually pushed through it.

    stage2 is a MIXED kernel: every matmul operand crosses the cube<->vector DIR_BOTH pipe.
    The ring indexes entry i as (i % slot_num) * slot_size, so a tile larger than slot_size
    runs off the end of its slot into the next one, and a reservation smaller than
    slot_num * slot_size runs off the end of the ring. Both corrupt silently and both are
    shape-dependent — F2 was exactly this class of bug on the same pipe.
    """
    fn = "<module>"
    shapes: dict[str, tuple[int, int]] = {}
    pipes: dict[str, list[str]] = {}
    bufs: dict[str, list[tuple[str, int]]] = {}
    moved: dict[str, list[tuple[str, str, int]]] = {}   # fn -> [(op, name, bytes)]

    for raw in dump.read_text().splitlines():
        m = _FUNC.match(raw)
        if m:
            fn = m.group(1)
            continue
        t = _TILE.match(raw)
        if t:
            dims = [int(x) for x in t.group(2).replace(" ", "").split(",")]
            shapes[t.group(1)] = (dims[0], dims[-1])
        for core, args in _PIPE.findall(raw):
            pipes.setdefault(fn, []).append(f"{core}: {args.strip()}")
        for bname, bsize in _RESERVE.findall(raw):
            bufs.setdefault(fn, []).append((bname, int(bsize)))
        for op, arg in _XFER.findall(raw):
            arg = arg.split(",")[0].strip()
            r, c = shapes.get(arg, (0, 0))
            moved.setdefault(fn, []).append((op, arg, r * c * 4))

    for fn in sorted(set(pipes) | set(bufs) | set(moved)):
        if "stage2" not in fn:
            continue
        print(f"  --- {fn}")
        for p in pipes.get(fn, []):
            print(f"      {p}")
        for bname, bsize in bufs.get(fn, []):
            print(f"      reserve_buffer {bname} = {bsize} B")
        slot = 0
        for p in pipes.get(fn, []):
            m = re.search(r"slot_size=(\d+)", p)
            if m:
                slot = int(m.group(1))
        by_op: dict[str, int] = {}
        for op, _name, nbytes in moved.get(fn, []):
            by_op[op] = max(by_op.get(op, 0), nbytes)
        for op, mx in sorted(by_op.items()):
            flag = "  <<< EXCEEDS slot_size" if slot and mx > slot else ""
            print(f"      max {op:<16} = {mx} B  (slot_size {slot}){flag}")


def analyse(dump: pathlib.Path, want: tuple[int, int, int]) -> int:
    """Print the layout-op chains in gla_stage2_aiv. Returns the number of in-place hits."""
    L, dk, dv = want
    lines = dump.read_text().splitlines()

    fn = "<module>"
    addr: dict[str, tuple[str, int, int, str]] = {}   # name -> (sym, off, size, shape)
    hits = 0
    in_stage2 = False

    for raw in lines:
        m = _FUNC.match(raw)
        if m:
            fn = m.group(1)
            in_stage2 = "stage2_aiv" in fn
            if in_stage2:
                print(f"  --- {fn}")
            continue
        if not in_stage2:
            continue

        t = _TILE.match(raw)
        if not t:
            continue
        name, shape, sym, off, size = t.group(1), t.group(2).replace(" ", ""), t.group(3), int(t.group(4)), int(t.group(5))
        addr[name] = (sym, off, size, shape)

        a = _ASSIGN.search(raw)
        if not a:
            continue
        op, args = a.group(1), a.group(2)
        if op not in LAYOUT_OPS:
            continue

        srcs = [s.strip() for s in args.split(",") if s.strip() in addr]
        note = ""
        for s in srcs:
            if addr[s][:2] == (sym, off):
                note = "   <<< IN-PLACE: source and destination share an address"
                hits += 1
        src_desc = ", ".join(f"{s}{addr[s][3]}@{addr[s][1]}" for s in srcs) or "?"
        print(f"      {op:<10} {src_desc:<34} -> {name}[{shape}]@{off} (+{size}){note}")

    return hits


def main() -> int:
    L, C, dk, dv = (int(v) for v in sys.argv[1:5])
    platform = sys.argv[5] if len(sys.argv) > 5 else "a2a3"

    workdir = tempfile.mkdtemp(prefix=f"f31c_{C}_{dk}_{dv}_")
    os.chdir(workdir)

    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    from gla.implementations.pypto.fused_program import build_fused_forward_program

    print(f"=== L={L} C={C} dk={dk} dv={dv}  (N={L // C})   [{workdir}]")
    program = build_fused_forward_program(L, C, dk, dv, 1, 1)
    cfg = DistributedConfig(device_ids=[0], num_sub_workers=0)
    ir.compile(program, platform=platform, distributed_config=cfg)

    dumps = list(pathlib.Path(workdir).glob("**/passes_dump/33_after_AllocateMemoryAddr.py"))
    if not dumps:
        print("  no dump produced")
        return 2
    # Private cwd, so every dump here belongs to this compile; if a compile emits more than
    # one program dir, the outer program is the largest.
    dump = max(dumps, key=lambda p: p.stat().st_size)
    pipe_geometry(dump)
    hits = analyse(dump, (L, dk, dv))
    print(f"  => {hits} in-place layout op(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
