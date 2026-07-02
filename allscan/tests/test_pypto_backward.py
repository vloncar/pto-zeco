"""System test for the PyPTO DSL AllScan backward pass.

Compiles and runs the reverse-ring backward program on the target platform
(``--platform``) across the devices given by ``--device``. Verifies
``(dS, dgamma)`` against the sequential golden.
"""

import sys

import pytest
import torch

from allscan.common import (
    expected_allscan,
    expected_allscan_backward,
    make_grad_inputs,
    make_inputs,
)
from allscan.implementations.pypto.impl import PytoAllscanBackward


@pytest.mark.parametrize("K", [1, 2, 4])
def test_pypto_allscan_backward(test_config, device_ids, K):
    if len(device_ids) < 2:
        pytest.skip(f"allscan needs at least 2 devices, got {device_ids}")

    P = len(device_ids)
    dk, dv = 64, 64
    S_locals, gammas, _ = make_inputs(P, dk, dv)
    g_out = make_grad_inputs(P, dk, dv)
    outs = expected_allscan(S_locals, gammas)

    dS = torch.zeros((P, dk, dv), dtype=torch.float32)
    dgamma = torch.zeros((P, dk, 1), dtype=torch.float32)

    impl = PytoAllscanBackward()
    impl.build(dk, dv, K, P, device_ids=device_ids, platform=test_config.platform)
    try:
        impl.run_backward(g_out, gammas, outs, dS, dgamma)
    finally:
        impl.close()

    exp_dS, exp_dgamma = expected_allscan_backward(gammas, outs, g_out)
    assert torch.allclose(dS, exp_dS, atol=1e-3), (
        f"dS mismatch: max diff = {(dS - exp_dS).abs().max().item()}"
    )
    assert torch.allclose(dgamma, exp_dgamma, atol=1e-3), (
        f"dgamma mismatch: max diff = {(dgamma - exp_dgamma).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
