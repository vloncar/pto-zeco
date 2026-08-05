#!/usr/bin/env python3
"""Verify the simpler persistent-worker forward against the per-kernel forward + torch.

Gate for the amortized benchmark path. The persistent (multi-callable, held-worker)
runner was a silent-corruption trap on an older runtime, so it is checked
end-to-end — not just "did it run" — before any timing is quoted from it.

Runs, for each config:
  plain      — per-kernel worker cycling (today's forward)
  persistent — one held multi-callable worker per device
and compares both to expected_gla, and to each other, over several repeats
(the repeat matters: corruption previously appeared only on the 2nd+ dispatch).

Usage: python3 scratchpad/check_persistent_forward.py <dev_csv> [platform] [repeats]
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.simpler.impl import SimplerZeCo

CONFIGS = [(1, 128, 32, 32), (2, 128, 32, 32), (2, 256, 32, 32), (4, 128, 32, 32)]


def golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                        flatten_seq(A)).reshape(P, L, dv)


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    worst = 0.0
    failures = 0
    for (P, L, C, D) in CONFIGS:
        if P > len(devices):
            print(f"P={P} L={L} C={C} D={D}: SKIP (needs {P} devices)")
            continue
        torch.manual_seed(P * 100 + L)
        Q, K, V, A = make_gla_inputs(P, L, C, D)
        exp = golden(Q, K, V, A)

        impl = SimplerZeCo()
        impl.build(P, L, C, C, D, devices[:P], platform)
        try:
            impl._use_plain_runners()
            plain = impl.forward(Q, K, V, A)
            e_plain = (plain - exp).abs().max().item()

            impl._use_persistent_runners()
            errs = []
            for i in range(repeats):
                got = impl._forward_persistent(Q, K, V, A)
                errs.append(((got - exp).abs().max().item(),
                             (got - plain).abs().max().item()))
            for r in impl._runners:
                r.close()
        finally:
            impl.close()

        e_ref = max(e for e, _ in errs)
        e_vs_plain = max(d for _, d in errs)
        worst = max(worst, e_ref)
        ok = e_ref < 1e-2 and e_vs_plain < 1e-3
        failures += 0 if ok else 1
        print(f"P={P} L={L} C={C} D={D}: plain={e_plain:.3e}  "
              f"persistent(vs ref)={e_ref:.3e}  persistent(vs plain)={e_vs_plain:.3e}  "
              f"x{repeats}  {'OK' if ok else '*** MISMATCH ***'}")

    print()
    print(f"VERDICT: {'persistent forward is CORRECT' if failures == 0 else 'MISMATCH — do not use'}"
          f" (worst vs reference {worst:.3e}, {failures} bad config(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
