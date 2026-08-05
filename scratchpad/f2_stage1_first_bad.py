"""F2 Phase 2 — WHERE and HOW REPEATABLY does stage1's carry go wrong?

f2_stage1_only.py established that stage1 STANDALONE (one device, no distribution, no ring,
no stage2) is wrong at N=8 while N=2/4 are clean. The per-chunk state errors:

    N=8 : 9.5e-7 1.2e-6 9.5e-7 1.9e-6 9.5e-7 9.5e-7 4.99   2.51
    N=16: 9.5e-7 1.1e-6 7.81   3.75   1.54   0.73   7.07   3.18  1.26 0.52 6.88 2.74 1.29 0.57 5.71 2.38

so the state is exact until some chunk, one iteration is corrupted, the error decays with the
gate, and (at N=16) fresh corruptions re-inject every 4 chunks. Four is also the depth of the
cube<->vector slot pipe: both `*_slot_buffer`s are 4096 B at slot_size=1024.

This script compiles ONCE per N and dispatches R times, printing the corrupted-chunk set per
repeat. That separates the two candidate mechanisms:

  * identical bad-chunk set every repeat -> a static (deterministic) miscompile
  * bad-chunk set varies across repeats  -> a RACE, i.e. missing AIC/AIV synchronisation,
    which also explains the two discrete max_diff values seen in the fused sweep

Usage: python3 scratchpad/f2_stage1_first_bad.py <device> [C] [N_csv] [repeats]
"""
from __future__ import annotations

import sys

import torch

from f2_stage1_only import build_stage1_trace, reference_states

TOL = 1e-2


def run(N: int, C: int, device: int, repeats: int, platform: str = "a2a3"):
    from pypto import ir
    from pypto.runtime.runner import RunConfig

    dk = dv = C
    L = N * C
    _, K, V, A, ref = reference_states(C, N, dk, dv)
    tril = torch.tril(torch.ones(C, C))
    ones_cc = torch.ones(C, C)
    ones_cdv = torch.ones(C, dv)
    zero = torch.zeros(dk, dv)
    cfg = RunConfig(platform=platform, device_id=device)

    compiled = ir.compile(build_stage1_trace(L, C, dk, dv), platform=platform)

    out = []
    for _ in range(repeats):
        Sall = torch.zeros(N * dk, dv)
        compiled(A[0], K[0], V[0], tril, ones_cc, ones_cdv, zero, Sall, config=cfg)
        errs = [(Sall[n * dk:(n + 1) * dk] - ref[n]).abs().max().item() for n in range(N)]
        out.append(errs)
    return out


if __name__ == "__main__":
    device = int(sys.argv[1].split(",")[0])
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5, 6, 7, 8, 12, 16]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    print(f"=== stage1 standalone: corrupted chunks vs trip count (C=dk=dv={C}, {R} repeats) ===")
    for N in Ns:
        try:
            runs = run(N, C, device, R)
            sets = []
            for i, errs in enumerate(runs):
                bad = [n for n, e in enumerate(errs) if e > TOL]
                sets.append(tuple(bad))
                print(f"N={N:>3} run{i}: bad={bad if bad else 'CLEAN'}  max={max(errs):.3e}", flush=True)
                print("        " + " ".join(f"{e:.1e}" for e in errs), flush=True)
            stable = len(set(sets)) == 1
            print(f"N={N:>3} -> bad-chunk set {'STABLE' if stable else '*** VARIES (race) ***'}\n",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"N={N:>3}  ERROR: {str(e)[:180]}\n", flush=True)
