"""System test for the simpler-runtime ZeCO / GLA **backward** (B3).

Runs the hand-written PTO-ISA backward kernels (grad_o + grad_h) in the simpler
runtime and checks (dQ,dK,dV,dA) against :func:`gla.common.expected_gla_backward`.

``P == 1`` runs on one device with no boundary (reduces to the sequential GLA
backward), so it is CI-safe on the simulator.  ``P > 1`` exchanges the boundary
gradient over the **real multi-device HCCL AllScan backward** (reverse ring), so
it needs hardware + ``LD_PRELOAD=libhccl.so``::

    source /usr/local/Ascend/cann-9.0.0/set_env.sh
    LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so \
        python gla/tests/test_simpler_gla_backward.py --platform a2a3 --devices 6,7

The pytest entry honours ``GLA_TEST_PLATFORM`` (default ``a2a3sim``) and
``GLA_TEST_DEVICES`` (default ``0`` -> P=1).
"""

import argparse
import os
import sys

import torch

from gla.common import expected_gla_backward, flatten_seq, make_gla_inputs
from gla.implementations.simpler.impl import SimplerZeCo


def _golden(Q, K, V, A, dO):
    P, L, dk = Q.shape
    dv = V.shape[2]
    gQ, gK, gV, gA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO)
    )
    return (gQ.reshape(P, L, dk), gK.reshape(P, L, dk),
            gV.reshape(P, L, dv), gA.reshape(P, L, dk))


def run(device_ids, platform="a2a3sim", L=256, C=128, D=128, dv=None):
    P = len(device_ids)
    dk = D
    dv = dv if dv is not None else D
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    torch.manual_seed(1234)
    dO = torch.randn(P, L, dv, dtype=torch.float32)

    impl = SimplerZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids, platform=platform)
    try:
        dQ, dK, dV, dA = impl.backward(Q, K, V, A, dO)
    finally:
        impl.close()

    ref = _golden(Q, K, V, A, dO)
    names = ("dQ", "dK", "dV", "dA")
    worst = 0.0
    for nm, got, r in zip(names, (dQ, dK, dV, dA), ref):
        rel = ((got - r).abs().max() / (r.abs().max() + 1e-6)).item()
        worst = max(worst, rel)
        print(f"[simpler-gla-bwd] P={P} L={L} C={C} dk={dk} dv={dv} {nm}: max_rel = {rel:.3e}")
    ok = worst < 2e-2
    print("[simpler-gla-bwd] RESULT:", "PASS" if ok else "FAIL", f"(worst {worst:.3e})")
    return ok


def test_simpler_zeco_backward():
    """pytest entry: P=1 on the simulator by default (no NPU / no LD_PRELOAD)."""
    platform = os.environ.get("GLA_TEST_PLATFORM", "a2a3sim")
    devs = [int(x) for x in os.environ.get("GLA_TEST_DEVICES", "0").split(",") if x != ""]
    assert run(devs, platform=platform)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="0")
    ap.add_argument("--platform", default="a2a3sim")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--D", type=int, default=128, help="key/query dim dk")
    ap.add_argument("--dv", type=int, default=None)
    args = ap.parse_args()
    devs = [int(x) for x in args.devices.split(",") if x != ""]
    sys.exit(0 if run(devs, args.platform, args.L, args.C, args.D, args.dv) else 1)
