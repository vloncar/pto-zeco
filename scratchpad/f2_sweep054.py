"""F2 gate sweep: pypto fused GLA forward vs expected_gla across P x N x C.

Usage: python3 scratchpad/f2_sweep054.py <device_csv> <C_csv> <P_csv> <N_csv> [repeat]
Example: python3 scratchpad/f2_sweep054.py $TASK_DEVICE 16,32 1,2,4 2,4,8,16 1
"""
import sys

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

TOL = 1e-2


def golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)).reshape(P, L, dv)


def run(P, N, C, devices, repeat):
    dk = dv = C
    L = N * C
    torch.manual_seed(N * 100 + P)
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    exp = golden(Q, K, V, A)
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=devices[:P], platform="a2a3")
    try:
        return [(impl.forward(Q, K, V, A) - exp).abs().max().item() for _ in range(repeat)]
    finally:
        impl.close()


if __name__ == "__main__":
    devices = [int(x) for x in sys.argv[1].split(",")]
    Cs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16]
    Ps = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2]
    Ns = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [2, 4, 8, 16]
    repeat = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    results = []
    for C in Cs:
        for P in Ps:
            if len(devices) < P:
                print(f"C={C} P={P}: SKIP (only {len(devices)} devices)", flush=True)
                continue
            for N in Ns:
                try:
                    ds = run(P, N, C, devices, repeat)
                    ok = all(d < TOL for d in ds)
                    results.append((C, P, N, max(ds), ok))
                    print(f"C={C:2d} P={P} N={N:2d}  max_diff={max(ds):.3e}  "
                          f"{'PASS' if ok else '*** FAIL ***'}"
                          + (f"   runs={['%.3e' % d for d in ds]}" if repeat > 1 else ""), flush=True)
                except Exception as e:  # noqa: BLE001
                    results.append((C, P, N, float("nan"), False))
                    print(f"C={C:2d} P={P} N={N:2d}  ERROR: {str(e)[:160]}", flush=True)

    bad = [r for r in results if not r[4]]
    print(f"\n==== {len(results) - len(bad)}/{len(results)} passed ====")
    for C, P, N, d, ok in bad:
        print(f"  FAIL C={C} P={P} N={N} diff={d:.3e}")
    sys.exit(1 if bad else 0)
