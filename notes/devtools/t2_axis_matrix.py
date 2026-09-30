#!/usr/bin/env python3
"""Task 2: which axis actually triggers the P>=2 failure, and is it nondeterministic?

The known bad case is P=2, L=128, C=64, dk=32, dv=64 (P=1 clean at the same shape). Three
candidate rules fit that single point equally well:

  (a) dk != dv          -- any asymmetry between the two head dims
  (b) dk < C            -- which would make it F3.1c's `N < M` condition on the
                           `b = tril[C,C] @ la[C,dk]` matmul, i.e. the SAME defect
  (c) dk < dv           -- direction-sensitive asymmetry

Two configs separate them, both with dv >= C so the `dv >= C` guard stays out of the way:

  dk > dv, dk > C   (C=32, dk=64, dv=32) -- fails under (a), passes under (b) and (c)
  dk == C, dv > C   (C=32, dk=32, dv=64) -- fails under (a) and (c), passes under (b)

Each config is BUILT ONCE and then dispatched `REPEATS` times on the same inputs, so a
spread across repeats is dispatch-to-dispatch nondeterminism rather than build variation.
That is the property that most distinguishes this from F3.1c, which was bit-stable.

Usage: python3 devtools/t2_axis_matrix.py <device_csv> <platform> [repeats]
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

REPEATS = 3

# (P, L, C, dk, dv, note)
CONFIGS = [
    # --- the known point, and its P=1 control ---
    (2, 128, 64, 32, 64, "THE FAILING CASE"),
    (1, 128, 64, 32, 64, "same shape at P=1 -- known clean"),
    (4, 128, 64, 32, 64, "same shape at P=4 -- does it worsen?"),

    # --- controls: dk == dv, known good ---
    (2, 128, 64, 64, 64, "control dk == dv == C"),
    (2, 256, 64, 64, 64, "control dk == dv == C, N=4"),
    (2, 128, 32, 32, 32, "control dk == dv == C at C=32"),

    # --- the two discriminators, at C=32 so dv >= C holds both ways ---
    (2, 128, 32, 64, 32, "DISCRIMINATOR dk > dv, dk > C  -> (a) only"),
    (2, 128, 32, 32, 64, "DISCRIMINATOR dk == C, dv > C  -> (a),(c) not (b)"),

    # --- dk < C at a second size, to confirm the rule generalises ---
    (2, 128, 32, 16, 32, "dk < C at C=32 (analogue of the failing case)"),
    (2, 128, 64, 16, 64, "dk much smaller than C"),
]


def canary(devices, platform) -> bool:
    """Smallest known-good config first: a bad card fails every program at bring-up."""
    torch.manual_seed(0)
    Q, K, V, A = make_gla_inputs(1, 32, 16, 16)
    impl = PyPtoZeCo()
    try:
        impl.build(1, 32, 16, 16, 16, device_ids=devices[:1], platform=platform)
        impl.forward(Q, K, V, A)
        return True
    except Exception as exc:  # noqa: BLE001 - that is the signal
        print(f"CANARY FAILED on device {devices[0]}: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:160]}")
        return False
    finally:
        impl.close()


def structure(err: torch.Tensor, C: int) -> list[str]:
    """Where the error lives. Boundary-only implicates the ring; everywhere implicates compute."""
    P, L, dv = err.shape
    N = L // C
    out = []
    for p in range(P):
        e = err[p]
        per_chunk = [e[n * C:(n + 1) * C].max().item() for n in range(N)]
        out.append(f"      rank {p} per-chunk: " + " ".join(f"{v:.2e}" for v in per_chunk))
    return out


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2]
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else REPEATS

    if not canary(devices, platform):
        print("Re-submit on a different card before believing anything here.")
        return 2
    print(f"canary OK on device {devices[0]}\n")

    for (P, L, C, dk, dv, note) in CONFIGS:
        tag = f"P={P} L={L} C={C} dk={dk} dv={dv} (N={L // C})"
        print(f"\n=== {tag}\n    {note}", flush=True)
        if P > len(devices):
            print("    SKIP (not enough devices)")
            continue

        torch.manual_seed(1234)
        Q, K, V, A = make_gla_inputs(P, L, dk, dv)
        exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                           flatten_seq(A)).reshape(P, L, dv)

        impl = PyPtoZeCo()
        diffs = []
        try:
            impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform=platform)
            for _ in range(repeats):
                O = impl.forward(Q, K, V, A)
                diffs.append(((O - exp).abs(), (O - exp).abs().max().item()))
        except AssertionError as exc:  # the dv >= C guard
            print(f"    GUARDED: {str(exc).splitlines()[0][:120]}")
            continue
        except Exception as exc:  # noqa: BLE001 - a failure is a result here
            print(f"    ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:140]}")
            continue
        finally:
            impl.close()

        vals = [d for _, d in diffs]
        ok = all(v < 1e-2 for v in vals)
        spread = max(vals) - min(vals)
        print(f"    max diff per dispatch: " + " ".join(f"{v:.4e}" for v in vals))
        print(f"    verdict: {'OK' if ok else '*** WRONG ***'}"
              f"   spread {spread:.3e}"
              f"   {'(NONDETERMINISTIC)' if spread > 1e-6 else '(stable)'}")
        if not ok:
            for line in structure(diffs[0][0], C):
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
