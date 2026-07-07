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
        L: int = 256, C: int = 128, D: int = 128) -> bool:
    P = len(device_ids)
    Q, K, V, A = make_gla_inputs(P, L, D, D)

    impl = SimplerZeCo()
    impl.build(P, L, C, D, D, device_ids=device_ids, platform=platform)
    try:
        O = impl.forward(Q, K, V, A)
    finally:
        impl.close()

    exp = _golden(Q, K, V, A)
    max_abs = (O - exp).abs().max().item()
    max_rel = max_abs / (exp.abs().max().item() + 1e-6)
    per_rank = [((O[p] - exp[p]).abs().max() / (exp[p].abs().max() + 1e-6)).item() for p in range(P)]
    print(f"[simpler-gla] P={P} L={L} C={C} D={D} platform={platform}  "
          f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    for p in range(P):
        print(f"[simpler-gla]   rank {p}: max_rel = {per_rank[p]:.3e}")
    # fp32 kernels vs fp32 chunked golden: exact to ~1e-6.
    ok = max_rel < 2e-2
    print("[simpler-gla] RESULT:", "PASS" if ok else "FAIL")
    return ok


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
    ap.add_argument("--D", type=int, default=128)
    args = ap.parse_args()
    devs = [int(x) for x in args.devices.split(",") if x != ""]
    sys.exit(0 if run(devs, args.platform, args.L, args.C, args.D) else 1)
