"""System test for the simpler-runtime ZeCO / GLA backend.

Runs the hand-written PTO-ISA kernels (gate_cumsum -> chunk_h(w=0) -> chunk_o) in
the **simpler runtime** (base env, no torch-npu) and checks the output against the
:func:`gla.common.expected_gla` golden. Realistic shapes: ``dk == dv == 128``,
``C == 128``.

``P == 1`` runs the compute on one device (no boundary). ``P > 1`` exchanges the
boundary state over the **real multi-device HCCL AllScan** (rank ``p`` computes on
``device_ids[p]``).

Runs on the **simulator** (``a2a3sim``) with no NPU by default, so it is CI-safe.
For hardware, HCCL must be preloaded::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so \
        python gla/tests/test_simpler_gla.py --platform a2a3 --devices 0,1

The pytest entry honours ``GLA_TEST_PLATFORM`` (default ``a2a3sim``) and
``GLA_TEST_DEVICES`` (default ``0`` -> P=1).
"""

import argparse
import os
import sys

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.simpler.impl import SimplerZeCo


def _golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)
    ).reshape(P, L, dv)


def run(device_ids: list[int], platform: str = "a2a3sim",
        L: int = 256, C: int = 128, D: int = 128, dv: int | None = None) -> bool:
    P = len(device_ids)
    dk = D
    dv = dv if dv is not None else D
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)

    impl = SimplerZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids, platform=platform)
    try:
        O = impl.forward(Q, K, V, A)
    finally:
        impl.close()

    exp = _golden(Q, K, V, A)
    max_abs = (O - exp).abs().max().item()
    max_rel = max_abs / (exp.abs().max().item() + 1e-6)
    per_rank = [((O[p] - exp[p]).abs().max() / (exp[p].abs().max() + 1e-6)).item() for p in range(P)]
    print(f"[simpler-gla] P={P} L={L} C={C} dk={dk} dv={dv} platform={platform}  "
          f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    for p in range(P):
        print(f"[simpler-gla]   rank {p}: max_rel = {per_rank[p]:.3e}")
    # fp32 kernels vs fp32 chunked golden: exact to ~1e-6.
    ok = max_rel < 2e-2
    print("[simpler-gla] RESULT:", "PASS" if ok else "FAIL")
    return ok


# Curated larger-shape correctness sweep (C,D in {16,32,64,128}; square + rectangular;
# larger L => more chunks N=L/C). Every config is checked against expected_gla.
SWEEP = [
    (16, 16, 256),    # N=16, small square
    (32, 32, 512),    # N=16
    (32, 64, 256),    # N=8,  rectangular C<D
    (64, 64, 512),    # N=8
    (64, 128, 512),   # N=8,  rectangular C<D (realistic)
    (128, 128, 512),  # N=4,  full square
    (128, 64, 256),   # N=2,  rectangular C>D
    (16, 128, 256),   # N=16, rectangular C<<D
]


def run_sweep(device_ids: list[int], platform: str = "a2a3") -> bool:
    """Run the curated shape sweep on the given devices; return True iff all pass."""
    results = []
    for C, D, L in SWEEP:
        try:
            ok = run(device_ids, platform=platform, L=L, C=C, D=D)
        except Exception as e:  # noqa: BLE001 - report + continue the sweep
            print(f"[simpler-gla-sweep] C={C} D={D} L={L} ERROR: {e}")
            ok = False
        results.append((C, D, L, ok))
    print("\n[simpler-gla-sweep] summary:")
    for C, D, L, ok in results:
        print(f"  C={C:<4} D={D:<4} L={L:<5} N={L // C:<3} -> {'PASS' if ok else 'FAIL'}")
    n_pass = sum(ok for *_, ok in results)
    print(f"[simpler-gla-sweep] {n_pass}/{len(results)} passed")
    return n_pass == len(results)


def test_simpler_zeco():
    """pytest entry: runs on the simulator by default (no NPU / no LD_PRELOAD)."""
    platform = os.environ.get("GLA_TEST_PLATFORM", "a2a3sim")
    devs = [int(x) for x in os.environ.get("GLA_TEST_DEVICES", "0").split(",") if x != ""]
    assert run(devs, platform=platform)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0", help="comma-separated device ids (P = count)")
    ap.add_argument("--platform", default="a2a3sim")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--D", type=int, default=128, help="key/query dim dk")
    ap.add_argument("--dv", type=int, default=None, help="value dim (default: = D)")
    ap.add_argument("--sweep", action="store_true", help="run the curated shape sweep")
    args = ap.parse_args()
    devs = [int(x) for x in args.devices.split(",") if x != ""]
    if args.sweep:
        sys.exit(0 if run_sweep(devs, args.platform) else 1)
    sys.exit(0 if run(devs, args.platform, args.L, args.C, args.D, args.dv) else 1)
