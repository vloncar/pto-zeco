"""System test for the direct PTO-runtime ("simpler") AllScan.

Compiles the AIV kernel + orchestration shim and runs the multi-chip DAG on the
target platform selected via ``--platform`` (a2a3 hardware or a2a3sim
simulator) across the devices given by ``--device``.
"""

import sys

import pytest
import torch

from allscan.common import expected_allscan, make_inputs
from allscan.implementations.simpler.impl import SimplerAllscan


@pytest.mark.parametrize("K", [1, 4])
@pytest.mark.parametrize("dk,dv", [(64, 64), (128, 128)])
def test_simpler_allscan(test_config, device_ids, dk, dv, K):
    if len(device_ids) < 2:
        pytest.skip(f"allscan needs at least 2 devices, got {device_ids}")

    P = len(device_ids)
    S_locals, gammas, outputs = make_inputs(P, dk, dv)

    impl = SimplerAllscan()
    impl.build(dk, dv, K, P, device_ids=device_ids, platform=test_config.platform)
    try:
        impl.run(S_locals, gammas, outputs)
    finally:
        impl.close()

    expected = expected_allscan(S_locals, gammas)
    assert torch.allclose(outputs, expected, atol=1e-3), (
        f"AllScan mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
