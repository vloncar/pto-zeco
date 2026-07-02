"""System test for the PyPTO DSL AllScan.

Compiles and runs on the target platform selected via ``--platform`` (a2a3
hardware or a2a3sim simulator) across the devices given by ``--device``.
"""

import sys

import pytest
import torch

from allscan.common import expected_allscan, make_inputs
from allscan.implementations.pypto.impl import PytoAllscan


@pytest.mark.parametrize("K", [1, 2, 4])
def test_pypto_allscan(test_config, device_ids, K):
    if len(device_ids) < 2:
        pytest.skip(f"allscan needs at least 2 devices, got {device_ids}")

    P = len(device_ids)
    dk, dv = 64, 64
    S_locals, gammas, outputs = make_inputs(P, dk, dv)

    impl = PytoAllscan()
    impl.build(dk, dv, K, P, device_ids=device_ids, platform=test_config.platform)
    try:
        impl.run(S_locals, gammas, outputs)
    finally:
        impl.close()

    expected = expected_allscan(S_locals, gammas)
    assert torch.allclose(outputs, expected, atol=1e-5), (
        f"AllScan mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
