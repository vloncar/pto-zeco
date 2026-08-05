"""F2 gate: pypto fused GLA forward correctness across chunk counts N = L//C.

Historically deterministically WRONG for N > 2 (the distributed loop-carry
miscompile): P=1 broke at N>=8, P=2 at N>=4. Re-run after the PTOAS 0.50 update.

Usage: python3 f2_sweep.py <device_csv> [C]
"""
import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo


def golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)).reshape(P, L, dv)


def run(P, N, C, devices):
    dk = dv = C
    L = N * C
    torch.manual_seed(N * 100 + P)
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform="a2a3")
    try:
        O = impl.forward(Q, K, V, A)
    finally:
        impl.close()
    exp = golden(Q, K, V, A)
    return (O - exp).abs().max().item()


if __name__ == "__main__":
    devices = [int(x) for x in sys.argv[1].split(",")]
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    results = []
    for P in (1, 2):
        if len(devices) < P:
            continue
        for N in (2, 4, 8, 16):
            try:
                d = run(P, N, C, devices)
                ok = d < 1e-2
                results.append((P, N, d, ok, ""))
                print(f"P={P} N={N:2d} C={C}  max_diff={d:.3e}  {'PASS' if ok else '*** FAIL ***'}", flush=True)
            except Exception as e:  # noqa: BLE001 - sweep must continue past a bad config
                results.append((P, N, float("nan"), False, str(e)[:120]))
                print(f"P={P} N={N:2d} C={C}  ERROR: {str(e)[:120]}", flush=True)

    print("\n==== F2 SUMMARY ====")
    for P, N, d, ok, err in results:
        print(f"  P={P} N={N:2d}: {'PASS' if ok else 'FAIL'}  diff={d:.3e}  {err}")
    bad = [r for r in results if not r[3]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)
