#!/usr/bin/env python3
"""Task 3: forward-at-scale sweep — N and P scaling, and the shape corners, vs expected_gla.

F2 (the N>2 loop-carry corruption) was fixed and spot-checked at 12/12, but the full sweep was
never run, so "the forward is correct at scale" has been an inference. This measures it.

Two axes, deliberately not a full cross product (that is ~75 configs and mostly redundant):

  N-axis      one representative shape, N = L//C in {2,4,8,16,32}, P in {1,2,4}
  shape-axis  N fixed, every reachable shape corner, P fixed at 2 (the real ring)

Each config is BUILT ONCE and dispatched `--repeats` times against DISTINCT seeds. Both matter:
`make_gla_inputs` seeds torch itself (default 42), so without varying the seed every repeat
replays one input point; and the pto-isa FIFO aliasing class of bug reproduced as rarely as
1 dispatch in 20, so single-shot verdicts are weak. A failure reports the seed that produced
it, so it can be replayed exactly.

Detection power: with R repeats, a defect that corrupts a fraction p of dispatches is missed
with probability (1-p)^R -- R=10 misses a 5% defect 60% of the time, R=20 misses it 36%. This
sweep reports its own power so a clean result is not overread.

Usage: python3 devtools/f_scale_sweep.py <device_csv> <platform> <backend> [--repeats N]
       backend: pypto | simpler
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs


def _impl(backend: str):
    if backend == "pypto":
        from gla.implementations.pypto.impl import PyPtoZeCo
        return PyPtoZeCo()
    from gla.implementations.simpler.impl import SimplerZeCo
    return SimplerZeCo()


# (C, dk, dv) corners. pypto is bounded by the 184 KB vector buffer (C,D <= 64) and by the
# min(dk,dv) >= C guard; simpler reaches 128.
SHAPES = {
    "pypto": [(16, 16, 16), (32, 32, 32), (32, 64, 32), (32, 32, 64), (64, 64, 64)],
    "simpler": [(16, 16, 16), (32, 32, 32), (32, 64, 32), (32, 32, 64), (64, 64, 64),
                (128, 128, 128)],
}
N_AXIS = [2, 4, 8, 16, 32]
N_AXIS_SHAPE = (32, 32, 32)   # representative: mid-size, square, reachable by both
SHAPE_AXIS_N = 4


def build_configs(backend: str, ps: list[int]) -> list[tuple]:
    """(P, L, C, dk, dv, axis) with no duplicates."""
    out, seen = [], set()

    def add(P, C, dk, dv, N, axis):
        key = (P, N * C, C, dk, dv)
        if key not in seen:
            seen.add(key)
            out.append((*key, axis))

    C, dk, dv = N_AXIS_SHAPE
    for N in N_AXIS:
        for P in ps:
            add(P, C, dk, dv, N, f"N={N}")
    for (C, dk, dv) in SHAPES[backend]:
        add(2 if 2 in ps else ps[0], C, dk, dv, SHAPE_AXIS_N, "shape")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("devices")
    ap.add_argument("platform")
    ap.add_argument("backend", choices=["pypto", "simpler"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=7000)
    args = ap.parse_args()

    devices = [int(x) for x in args.devices.split(",")]
    ps = [p for p in (1, 2, 4) if p <= len(devices)]
    R = args.repeats
    miss5 = 0.95 ** R

    configs = build_configs(args.backend, ps)
    print(f"=== {args.backend} scale sweep: {len(configs)} configs x {R} dispatches, "
          f"distinct seeds, devices={devices}")
    print(f"=== detection power: a 5%-rate defect is missed with p={miss5:.0%}; "
          f"a 25%-rate defect with p={0.75 ** R:.1%}\n", flush=True)

    bad, errored, results = [], [], []
    for (P, L, C, dk, dv, axis) in configs:
        tag = f"P={P} L={L:5d} C={C:3d} dk={dk:3d} dv={dv:3d} N={L // C:2d} [{axis}]"
        if P > len(devices):
            print(f"  SKIP  {tag}  (needs {P} devices)")
            continue
        impl = _impl(args.backend)
        t0 = time.time()
        worst, worst_seed = 0.0, args.seed0
        try:
            impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform=args.platform)
            for i in range(R):
                s = args.seed0 + i
                Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=s)
                exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                                   flatten_seq(A)).reshape(P, L, dv)
                err = (impl.forward(Q, K, V, A) - exp).abs().max().item()
                if err > worst:
                    worst, worst_seed = err, s
        except AssertionError as exc:
            print(f"  GUARD {tag}  {str(exc).splitlines()[0][:70]}")
            continue
        except Exception as exc:  # noqa: BLE001 - an error is a result here
            print(f"  ERROR {tag}  {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
            errored.append((tag, type(exc).__name__))
            continue
        finally:
            impl.close()

        ok = worst < 1e-2
        results.append((tag, worst, ok))
        if not ok:
            bad.append((tag, worst, worst_seed))
        print(f"  {'ok   ' if ok else 'WRONG'} {tag}  worst {worst:.3e}"
              f"{'' if ok else f'  seed={worst_seed}'}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== SUMMARY {args.backend}: {len(results) - len(bad)}/{len(results)} configs clean,"
          f" {len(bad)} wrong, {len(errored)} errored")
    for tag, worst, seed in bad:
        print(f"    WRONG {tag}  worst {worst:.3e}  replay with seed={seed}")
    for tag, kind in errored:
        print(f"    ERROR {tag}  {kind}")
    return 1 if (bad or errored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
