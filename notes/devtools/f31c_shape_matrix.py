#!/usr/bin/env python3
"""F3.1c: localise the `C > D` HW failure by shape, and describe its error *structure*.

The known bad case is P=1, L=128, C=64, dk=dv=32 (max diff ~72) on a2a3 hardware only;
a2a3sim computes the same shape correctly. This sweep answers three questions that the
single failing config cannot:

  1. **dk or dv?** dk and dv have always been swept together, so "C > D" is ambiguous.
     (C=64, dk=64, dv=32) and (C=64, dk=32, dv=64) separate them.
  2. **Loop carry or within-chunk?** N = L//C = 1 has no loop-carried state at all.
     If N=1 already fails, the bug is inside a single chunk and the carry is innocent.
  3. **Ratio or absolute size?** C=32, dk=dv=16 is the same 2:1 shape at half the size.

Beyond pass/fail it prints *where* the error lives — per chunk, per row within a chunk,
per output column. A per-chunk ramp implicates the carry; a column pattern implicates
tile layout; a row pattern implicates the within-chunk cumulative decay.

Usage: python3 devtools/f31c_shape_matrix.py <device_csv> <platform>
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

# (P, L, C, dk, dv, note). P=1 throughout: the failure is in stage2 on a single device,
# so the ring and stage1 are not involved and every run is cheap.
CONFIGS = [
    (1, 128, 64, 32, 32, "KNOWN BAD baseline (N=2)"),
    (1, 64, 64, 32, 32, "N=1: no loop carry at all"),
    (1, 128, 64, 64, 32, "C == dk, dv < C   -> is dv the trigger?"),
    (1, 128, 64, 32, 64, "dk < C, dv == C   -> is dk the trigger?"),
    (1, 128, 64, 64, 64, "control: known good"),
    (1, 128, 32, 32, 32, "control: known good (pre-F3.1 ceiling)"),
    (1, 128, 32, 16, 16, "same 2:1 C>D ratio, half the size"),
]


def describe(err: torch.Tensor, C: int) -> list[str]:
    """err is [L, dv] absolute error for one rank. Report where it concentrates."""
    L, dv = err.shape
    N = L // C
    lines = []

    per_chunk = [err[n * C:(n + 1) * C].max().item() for n in range(N)]
    lines.append("    per chunk : " + " ".join(f"{v:.2e}" for v in per_chunk))

    # Row position *within* a chunk, maxed over chunks and columns.
    rows = err.reshape(N, C, dv).amax(dim=(0, 2))
    lines.append("    per row   : " + _sparkline(rows))

    cols = err.amax(dim=0)
    lines.append("    per col   : " + _sparkline(cols))

    bad = (err > 1e-2).nonzero()
    if bad.numel():
        lines.append(f"    first bad : token {bad[0, 0].item()} (chunk {bad[0, 0].item() // C}), "
                     f"col {bad[0, 1].item()};  {bad.shape[0]} of {L * dv} elements bad")
    return lines


def _sparkline(v: torch.Tensor) -> str:
    """Compact magnitude profile: one char per entry, '.' clean through '9' worst."""
    hi = v.max().item()
    if hi <= 0:
        return "." * v.numel()
    out = []
    for x in v.tolist():
        out.append("." if x < 1e-2 else str(min(9, int(9 * x / hi) + 1)))
    return "".join(out) + f"   (max {hi:.2e})"


def canary(devices, platform) -> bool:
    """Smallest known-good config, run first.

    Cards 1, 4 and 6 fail EVERY pypto program at device bring-up (`halMemCtl rc=42` in
    `init_aicore_register_addresses`) before any kernel math runs, and `npu-smi` reports
    them Health OK. Without this check a bad card produces a full matrix of identical
    "failures" that look like a code result — which is exactly how it presented the first
    time (see the F3.1d roadmap note). Abort instead.
    """
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
        print("The smallest known-good config cannot run, so this is the CARD, not the code.")
        print("Re-submit on a different card before believing any result from this box.")
        return False
    finally:
        impl.close()


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2]

    if not canary(devices, platform):
        return 2
    print(f"canary OK on device {devices[0]}\n")

    bad = 0
    for (P, L, C, dk, dv, note) in CONFIGS:
        tag = f"P={P} L={L} C={C} dk={dk} dv={dv} (N={L // C})"
        print(f"\n=== {tag}  --  {note}", flush=True)
        if P > len(devices):
            print("    SKIP (not enough devices)")
            continue

        torch.manual_seed(1234)
        Q, K, V, A = make_gla_inputs(P, L, dk, dv)
        exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                           flatten_seq(A)).reshape(P, L, dv)

        impl = PyPtoZeCo()
        try:
            impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform=platform)
            O = impl.forward(Q, K, V, A)
        except Exception as exc:  # noqa: BLE001 - a compile/run failure is a result here
            print(f"    ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:140]}")
            bad += 1
            continue
        finally:
            impl.close()

        err = (O - exp).abs()
        m = err.max().item()
        ok = m < 1e-2
        bad += 0 if ok else 1
        print(f"    max diff {m:.4e}   {'OK' if ok else '*** WRONG ***'}")
        if not ok:
            for line in describe(err[0], C):
                print(line)

    print(f"\n{len(CONFIGS) - bad}/{len(CONFIGS)} OK")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
