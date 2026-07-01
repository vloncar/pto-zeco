"""Tests for the AllScan backward pass.

Two layers of verification, both pure CPU (run anywhere):

* ``test_backward_matches_autograd`` — validates the *closed-form* backward
  reference in :func:`expected_allscan_backward` against ``torch.autograd``
  differentiating the forward recurrence. This checks the math itself.

* ``test_torch_backward_matches_reference`` — validates the ``TorchAllscan``
  adapter's ``run_backward`` (the K-blocked reverse ring) against that same
  closed-form reference, the golden every device backend will verify against.
"""

import sys

import pytest
import torch

from common import (
    expected_allscan,
    expected_allscan_backward,
    make_grad_inputs,
    make_inputs,
)
from implementations.torch_ref import TorchAllscan


def _autograd_backward(S_locals, gammas, g_out):
    """Ground-truth grads via autograd through the forward recurrence."""
    P = S_locals.shape[0]
    S = S_locals.clone().requires_grad_(True)
    G = gammas.clone().requires_grad_(True)
    outs = [S[0]]
    for p in range(1, P):
        outs.append(S[p] + G[p] * outs[p - 1])
    out = torch.stack(outs)
    out.backward(g_out)
    return S.grad, G.grad


@pytest.mark.parametrize("P", [2, 4])
def test_backward_matches_autograd(P):
    dk, dv = 64, 64
    S_locals, gammas, _ = make_inputs(P, dk, dv)
    g_out = make_grad_inputs(P, dk, dv)
    outs = expected_allscan(S_locals, gammas)

    dS, dgamma = expected_allscan_backward(gammas, outs, g_out)
    ad_dS, ad_dgamma = _autograd_backward(S_locals, gammas, g_out)

    assert torch.allclose(dS, ad_dS, atol=1e-5), (
        f"dS vs autograd max diff = {(dS - ad_dS).abs().max().item()}"
    )
    assert torch.allclose(dgamma, ad_dgamma, atol=1e-5), (
        f"dgamma vs autograd max diff = {(dgamma - ad_dgamma).abs().max().item()}"
    )


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("P", [2, 4])
def test_torch_backward_matches_reference(P, K):
    dk, dv = 64, 64
    S_locals, gammas, outputs = make_inputs(P, dk, dv)
    g_out = make_grad_inputs(P, dk, dv)

    impl = TorchAllscan()
    impl.build(dk, dv, K, P, device_ids=list(range(P)), platform="cpu")
    impl.run(S_locals, gammas, outputs)  # forward -> retained outs

    dS = torch.zeros_like(g_out)
    dgamma = torch.zeros_like(gammas)
    impl.run_backward(g_out, gammas, outputs, dS, dgamma)
    impl.close()

    exp_dS, exp_dgamma = expected_allscan_backward(gammas, outputs, g_out)
    assert torch.allclose(dS, exp_dS, atol=1e-5), (
        f"dS mismatch: max diff = {(dS - exp_dS).abs().max().item()}"
    )
    assert torch.allclose(dgamma, exp_dgamma, atol=1e-5), (
        f"dgamma mismatch: max diff = {(dgamma - exp_dgamma).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
