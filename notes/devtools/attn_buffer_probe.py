"""Q6-0: what does head_dim 256 actually cost in on-chip buffers?

The roadmap states the Q6d risk as head_dim 256 against the ~184 KB vector
buffer. The competing reading is that head_dim sizes the QK^T / PV *matmul*
tiles -- L1 (`Mat`) and the L0C accumulator (`Acc`) -- while the softmax
working set is [rows, kv] and does not scale with head_dim at all.

This compiles one prefill attention tile at a given (head_dim, row, kv)
and reads the compiler's own buffer report. No device, no queue.

The shape is the real one otherwise: GQA 24/4, causal, fp16 operands, fp32
scores and accumulator. Online-softmax rescaling is omitted -- it adds two
[row, 1] vectors, which cannot move a KB-scale conclusion -- so this measures
allocation, not numerics.
"""
import argparse
import json

import sys
from pathlib import Path

sys.path.insert(0, "/root/workspace/pypto/pypto-lib")

import pypto.language as pl  # noqa: E402
import torch  # noqa: E402

T = 1024                 # tokens; only the tile shapes matter here
HQ = 24                  # query heads
HKV = 4                  # KV heads
KV_STEPS = 2             # unrolled kv tiles, so the accumulator must persist


def build(d: int, row_tile: int, kv_tile: int, slot_num: int = 0):
    scale = float(d) ** -0.5
    kv_span = KV_STEPS * kv_tile

    @pl.jit
    def attn_tile(
        q: pl.Tensor[[T, HQ, d], pl.FP16],
        k: pl.Tensor[[T, HKV, d], pl.FP16],
        v: pl.Tensor[[T, HKV, d], pl.FP16],
        o_out: pl.Out[pl.Tensor[[T, HQ, d], pl.FP16]],
    ):
        q_flat = pl.reshape(q, [T, HQ * d])
        k_flat = pl.reshape(k, [T, HKV * d])
        v_flat = pl.reshape(v, [T, HKV * d])
        o_flat = pl.reshape(o_out, [T, HQ * d])
        group = HQ // HKV
        for r in pl.spmd(T // row_tile, name_hint="attn_tile",
                         optimizations=[pl.cross_core_slot(slot_num=slot_num)]):
            r0 = r * row_tile
            for hh in pl.range(HQ):
                qd0 = hh * d
                kd0 = (hh // group) * d
                qc = q_flat[r0 : r0 + row_tile, qd0 : qd0 + d]
                # The first kv tile opens the accumulator; the rest add into it.
                # Hoisted rather than branched, because the DSL traces Python
                # control flow and rejects the `is None` test.
                k0 = k_flat[0:kv_tile, kd0 : kd0 + d]
                v0 = v_flat[0:kv_tile, kd0 : kd0 + d]
                s0 = pl.matmul(qc, k0, b_trans=True, out_dtype=pl.FP32)
                p0 = pl.cast(pl.exp(pl.mul(s0, scale)), target_type=pl.FP16, mode="rint")
                acc = pl.matmul(p0, v0, out_dtype=pl.FP32)
                for j in pl.unroll(kv_tile, kv_span, kv_tile):
                    kj = k_flat[j : j + kv_tile, kd0 : kd0 + d]
                    vj = v_flat[j : j + kv_tile, kd0 : kd0 + d]
                    s = pl.matmul(qc, kj, b_trans=True, out_dtype=pl.FP32)
                    p = pl.exp(pl.mul(s, scale))
                    p16 = pl.cast(p, target_type=pl.FP16, mode="rint")
                    acc = pl.matmul_acc(acc, p16, vj)
                o_flat[r0 : r0 + row_tile, qd0 : qd0 + d] = pl.cast(
                    acc, target_type=pl.FP16, mode="rint"
                )
        return o_out

    return attn_tile


def read_allocation(work_dir: Path) -> dict:
    """Per-space high-water marks, from the compiler's own AllocateMemoryAddr dump."""
    from pypto.tools import memory_map as mm

    dumps = sorted(work_dir.glob("passes_dump/*_after_AllocateMemoryAddr.py"))
    if not dumps:
        return {}
    limits = mm.backend_limits("Ascend910B")
    worst: dict = {}
    for function in mm.parse_dump(dumps[0]):
        marks: dict[str, int] = {}
        for box in function.boxes:
            # Left/Right are compiler-managed L0A/L0B staging, not DSL budgets.
            if box.space in ("Left", "Right"):
                continue
            marks[box.space] = max(marks.get(box.space, 0), box.offset + box.size)
        for space, hwm in marks.items():
            limit = limits.get(space, 0)
            if space not in worst or hwm > worst[space]["used_kb"] * 1024:
                worst[space] = {
                    "used_kb": round(hwm / 1024, 1),
                    "limit_kb": round(limit / 1024, 1),
                    "pct": round(100.0 * hwm / limit, 1) if limit else 0.0,
                    "fn": function.name,
                }
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--row", type=int, default=64)
    ap.add_argument("--kv", type=int, default=128)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--platform", type=str, default="a2a3")
    args = ap.parse_args()

    from golden import TensorSpec, run

    d, row, kv = args.head_dim, args.row, args.kv
    slot = args.slot
    label = f"d={d} row={row} kv={kv} slot={slot or 'default'}"
    record = {"head_dim": d, "row": row, "kv": kv, "slot": slot}
    builds = Path("/root/workspace/pypto/pypto-lib/build_output")
    try:
        fn = build(d, row, kv, slot)
        run(
            fn=fn,
            specs=[
                TensorSpec("q", [T, HQ, d], torch.float16),
                TensorSpec("k", [T, HKV, d], torch.float16),
                TensorSpec("v", [T, HKV, d], torch.float16),
                TensorSpec("o_out", [T, HQ, d], torch.float16),
            ],
            compile_only=True,
            config=dict(platform=args.platform, dump_passes=True),
        )
    except Exception as error:  # a rejected tiling is a result, not a crash
        first = [ln for ln in str(error).strip().splitlines() if ln.strip()]
        record["status"] = "REJECTED"
        record["error"] = (first[0] if first else "")[:200]
        print(f"{label}: REJECTED -- {record['error']}")
        print("JSON " + json.dumps(record))
        return 0

    # The build directory is content-hashed, so a repeated configuration reuses
    # one rather than creating it. Take the newest dump instead of a new dir.
    dumps = sorted(
        builds.glob("_jit_attn_tile_*/passes_dump/*_after_AllocateMemoryAddr.py"),
        key=lambda p: p.stat().st_mtime,
    )
    record["status"] = "ok"
    record["work_dir"] = dumps[-1].parents[1].name if dumps else None
    record["spaces"] = read_allocation(dumps[-1].parents[1]) if dumps else {}
    spaces = " ".join(
        f"{s}={v['used_kb']:.0f}/{v['limit_kb']:.0f}KB({v['pct']:.0f}%)"
        for s, v in sorted(record["spaces"].items())
    )
    print(f"{label}: ok  {spaces or '(no report found)'}")
    print("JSON " + json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
