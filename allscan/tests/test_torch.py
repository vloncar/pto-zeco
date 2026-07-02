"""Tests for the in-process torch reference AllScan.

Pure CPU — runs on any platform/device selection. Verifies the K-blocked ring
adapter matches the independently-written sequential reference.
"""

import sys

import pytest
import torch

from allscan.common import expected_allscan, make_inputs
from allscan.implementations.torch_ref import TorchAllscan


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("P", [2, 4])
def test_torch_matches_reference(P, K):
    dk, dv = 64, 64
    S_locals, gammas, outputs = make_inputs(P, dk, dv)

    impl = TorchAllscan()
    impl.build(dk, dv, K, P, device_ids=list(range(P)), platform="cpu")
    impl.run(S_locals, gammas, outputs)
    impl.close()

    expected = expected_allscan(S_locals, gammas)
    assert torch.allclose(outputs, expected, atol=1e-5), (
        f"torch AllScan mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
