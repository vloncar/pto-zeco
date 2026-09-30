#!/usr/bin/env python3
"""Which card PAIRS on this box can actually stand up an HCCL comm domain?

`devtools/canary.py` is a SINGLE-device canary: it catches cards that fail at bring-up
(`halMemCtl rc=42`). It cannot catch the other failure mode, which is pair-scoped and only
appears once a program allocates a communication window:

    RuntimeError: alloc_domain(allocation_id=0) failed on N/N chips;
    control_alloc_domain failed on child: comm_alloc_domain_windows failed with code -1

That error cost three B4 runs read as code failures before a known-good comm program
(the AllScan forward) was run on the same cards and failed too. A P=1 canary would have
passed on those cards, so it would not have helped.

This runs the smallest real comm program — a 2-rank AllScan forward, one dispatch — over
every adjacent pair of the granted cards, and prints a map. Correctness is checked too, not
just "it didn't raise": a pair that allocates but computes garbage is worse than one that
fails loudly.

Pairs are only meaningful INSIDE one HCCS group (0-3 | 4-7); a straddling pair fails or
hangs for unrelated reasons, so those combinations are skipped rather than reported bad.

Usage: python3 devtools/comm_canary.py <device_csv> [platform]
"""

from __future__ import annotations

import itertools
import sys
import traceback


def _hccs_group(d: int) -> int:
    return 0 if d < 4 else 1


def try_pair(a: int, b: int, platform: str) -> tuple[bool, str]:
    """Build + dispatch a 2-rank AllScan forward on (a, b). Returns (ok, detail)."""
    from allscan.common import expected_allscan, make_inputs
    from allscan.implementations.pypto.impl import PyPtoAllscan

    dk, dv, K, P = 16, 16, 1, 2
    impl = PyPtoAllscan()
    try:
        impl.build(dk, dv, K, P, device_ids=[a, b], platform=platform)
        S, gammas, out = make_inputs(P, dk, dv)
        impl.run(S, gammas, out)
        err = (out - expected_allscan(S, gammas)).abs().max().item()
        return (err < 1e-3), f"max err {err:.2e}"
    except Exception as exc:  # noqa: BLE001 - the exception IS the measurement
        msg = " ".join(str(exc).split())
        for marker in ("comm_alloc_domain_windows", "comm_init", "halMemCtl"):
            if marker in msg:
                return False, marker
        return False, f"{type(exc).__name__}: {msg[:70]}"
    finally:
        try:
            impl.close()
        except Exception:  # noqa: BLE001, S110 - teardown noise must not mask the verdict
            traceback.print_exc(limit=1)


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    pairs = [(a, b) for a, b in itertools.combinations(devices, 2)
             if _hccs_group(a) == _hccs_group(b)]
    print(f"=== comm-domain canary over {devices} ({len(pairs)} in-group pairs)\n")
    good, bad = [], []
    for a, b in pairs:
        ok, detail = try_pair(a, b, platform)
        (good if ok else bad).append((a, b))
        print(f"  {'OK  ' if ok else 'BAD '} ({a},{b})  {detail}", flush=True)
    print(f"\n=== comm-capable pairs: {good or 'NONE'}")
    print(f"=== broken pairs:       {bad or 'none'}")
    if bad:
        suspects = {d for pair in bad for d in pair} - {d for pair in good for d in pair}
        print(f"=== cards in every broken pair and no good pair: {sorted(suspects) or 'none'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
