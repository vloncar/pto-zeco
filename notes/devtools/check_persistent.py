#!/usr/bin/env python3
"""Verify the simpler held-worker path against the per-kernel path + torch, BOTH directions.

Supersedes ``check_persistent_forward.py``, which only covered the forward (and called
``_forward_persistent``, now folded into ``forward()``).

For each config and each direction:
  plain      — one fresh worker per kernel dispatch (the reference path)
  persistent — one held multi-callable worker per device, dropped only around the
               boundary AllScan
and compares both to the torch golden, and to each other, over several repeats. The
repeat matters: the corruption this guards against (simpler ``a756969c``) appeared only
on the 2nd+ dispatch of a callable on a worker.

The backward is the direction that was never covered before, and the interesting one:
it dispatches five kernels per rank and one of them (``gate_cumsum``, again for the
reverse-cumsum) is a re-dispatch — exactly the pattern that used to corrupt dA.

Usage: python3 devtools/check_persistent.py <dev_csv> [platform] [repeats]
"""

from __future__ import annotations

import sys

import torch

from gla.common import (
    expected_gla,
    expected_gla_backward,
    flatten_seq,
    make_gla_inputs,
)
from gla.implementations.simpler.impl import SimplerZeCo

CONFIGS = [(1, 128, 32, 32), (2, 128, 32, 32), (2, 256, 32, 32), (4, 128, 32, 32)]

FWD_TOL_REF = 1e-2      # chunk math divides by within-chunk cumulative decay
BWD_TOL_REF = 1e-2      # relative, per-gradient (their magnitudes differ by orders)
TOL_VS_PLAIN = 1e-3     # held vs per-kernel: same kernels, so this should be ~0


def _fwd_golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                        flatten_seq(A)).reshape(P, L, dv)


def _bwd_golden(Q, K, V, A, dO):
    return expected_gla_backward(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                                 flatten_seq(A), flatten_seq(dO))


def _abs_err(got, exp) -> float:
    return (got - exp.reshape(got.shape)).abs().max().item()


def _rel_err(got_tuple, exp_tuple) -> float:
    """Worst per-gradient relative error (dQ/dK/dV/dA magnitudes differ hugely)."""
    worst = 0.0
    for g, e in zip(got_tuple, exp_tuple):
        e_r = e.reshape(g.shape)
        worst = max(worst, (g - e_r).abs().max().item() / max(e_r.abs().max().item(), 1e-12))
    return worst


def _tuple_delta(a, b) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def main() -> int:
    devices = [int(d) for d in sys.argv[1].split(",")]
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    print(f"devices={devices} platform={platform} repeats={repeats}\n")
    failures = 0

    for P, L, C, D in CONFIGS:
        if P > len(devices):
            print(f"P={P} L={L} C={C} D={D}: SKIP (needs {P} devices, have {len(devices)})")
            continue
        # Pass the seed THROUGH: make_gla_inputs seeds torch itself (default 42), so an
        # outer manual_seed is overwritten and every config replays one input point. That
        # is how the first run gave P=4/L=128 and P=2/L=256 byte-identical errors — same
        # seed, same P*L token count, just reshaped.
        seed = P * 1000 + L
        Q, K, V, A = make_gla_inputs(P, L, C, D, seed=seed)
        torch.manual_seed(seed + 1)
        dO = torch.randn(P, L, D)
        fwd_exp = _fwd_golden(Q, K, V, A)
        bwd_exp = _bwd_golden(Q, K, V, A, dO)

        impl = SimplerZeCo()
        impl.build(P, L, C, C, D, devices[:P], platform)
        try:
            impl._use_plain_runners()
            plain_f = impl.forward(Q, K, V, A)
            plain_b = impl.backward(Q, K, V, A, dO)

            impl._use_persistent_runners()
            f_ref, f_vs_plain, b_ref, b_vs_plain = [], [], [], []
            for _ in range(repeats):
                got_f = impl.forward(Q, K, V, A)
                f_ref.append(_abs_err(got_f, fwd_exp))
                f_vs_plain.append(_abs_err(got_f, plain_f))
                got_b = impl.backward(Q, K, V, A, dO)
                b_ref.append(_rel_err(got_b, bwd_exp))
                b_vs_plain.append(_tuple_delta(got_b, plain_b))
            impl._release_devices()
        finally:
            impl.close()

        ef, ep = max(f_ref), max(f_vs_plain)
        eb, bp = max(b_ref), max(b_vs_plain)
        ok_f = ef < FWD_TOL_REF and ep < TOL_VS_PLAIN
        ok_b = eb < BWD_TOL_REF and bp < TOL_VS_PLAIN
        failures += (0 if ok_f else 1) + (0 if ok_b else 1)

        print(f"P={P} L={L} C={C} D={D}  (plain fwd {_abs_err(plain_f, fwd_exp):.3e}, "
              f"plain bwd {_rel_err(plain_b, bwd_exp):.3e})")
        print(f"    forward  held: vs_ref={ef:.3e}  vs_plain={ep:.3e}  x{repeats}  "
              f"{'OK' if ok_f else '*** MISMATCH ***'}")
        print(f"    backward held: vs_ref={eb:.3e}  vs_plain={bp:.3e}  x{repeats}  "
              f"{'OK' if ok_b else '*** MISMATCH ***'}")

    print()
    print(f"VERDICT: {'held-worker path CORRECT both directions' if failures == 0 else 'MISMATCH'}"
          f" ({failures} bad direction/config combination(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
