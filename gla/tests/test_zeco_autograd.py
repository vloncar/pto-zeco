"""Tests for the differentiable ZeCO operator (:class:`gla.common.ZeCoModule`).

The autograd wrapper is backend-agnostic — it binds any :class:`ZeCoImpl`'s
forward + analytic backward into an autograd graph. These CPU tests exercise it
through the reference :class:`TorchZeCo` backend (no device), two ways:

  * :func:`torch.autograd.gradcheck` (finite differences, double precision) — the
    rigorous proof that the custom backward matches the forward's numeric Jacobian;
  * an end-to-end ``loss.backward()`` whose input grads match the analytic
    :func:`gla.common.expected_gla_backward` golden, across ranks / chunk sizes /
    ``dk != dv``.

The simpler-runtime backend runs the same wrapper through the real kernels in
``test_simpler_gla_backward.py --module`` (HW / sim); that path uses the analytic
golden (a finite-diff gradcheck would need ~O(inputs) device forwards).
"""

import pytest
import torch

from gla.common import (
    ZeCoModule,
    expected_gla_backward,
    flatten_seq,
    make_gla_inputs,
)
from gla.implementations.torch_ref import TorchZeCo


def _module(P, L, C, dk, dv):
    impl = TorchZeCo()
    impl.build(P, L, C, dk, dv, device_ids=list(range(P)), platform="cpu")
    return ZeCoModule(impl), impl


@pytest.mark.parametrize("P,L,C,dk,dv", [(1, 8, 4, 6, 5), (2, 8, 4, 5, 5), (2, 12, 4, 4, 6)])
def test_zeco_module_gradcheck(P, L, C, dk, dv):
    """Finite-difference gradcheck of the differentiable operator (double precision)."""
    torch.manual_seed(P * 10 + C)
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    Q, K, V, A = (t.double().requires_grad_(True) for t in (Q, K, V, A))
    mod, impl = _module(P, L, C, dk, dv)
    try:
        assert torch.autograd.gradcheck(
            lambda q, k, v, a: mod(q, k, v, a), (Q, K, V, A),
            eps=1e-6, atol=1e-5, rtol=1e-3,
        )
    finally:
        impl.close()


@pytest.mark.parametrize("P,L,C,dk,dv", [(1, 16, 8, 12, 12), (2, 16, 8, 16, 10), (4, 12, 4, 8, 14)])
def test_zeco_module_backward_matches_golden(P, L, C, dk, dv):
    """loss.backward() through the module == analytic golden on the full sequence."""
    torch.manual_seed(P * 7 + dk)
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    leaves = [t.clone().requires_grad_(True) for t in (Q, K, V, A)]
    W = torch.randn(P, L, dv)              # loss = (O * W).sum() -> dO = W

    mod, impl = _module(P, L, C, dk, dv)
    try:
        O = mod(*leaves)
        (O * W).sum().backward()
    finally:
        impl.close()

    gQ, gK, gV, gA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(W))
    ref = (gQ.reshape(P, L, dk), gK.reshape(P, L, dk),
           gV.reshape(P, L, dv), gA.reshape(P, L, dk))
    for name, leaf, r in zip(("dQ", "dK", "dV", "dA"), leaves, ref):
        assert torch.allclose(leaf.grad, r, atol=1e-4, rtol=1e-3), (
            f"{name} mismatch: max diff = {(leaf.grad - r).abs().max().item():.3e}")


def test_zeco_module_training_step_reduces_loss():
    """Smoke test: a gradient-descent step on the inputs lowers a regression loss,
    proving the operator is usable in a real optimisation loop."""
    torch.manual_seed(0)
    P, L, C, dk, dv = 2, 16, 8, 8, 8
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    target = torch.randn(P, L, dv)
    leaves = [t.clone().requires_grad_(True) for t in (Q, K, V)]  # optimise Q,K,V
    mod, impl = _module(P, L, C, dk, dv)
    try:
        def loss_of(qkv):
            O = mod(qkv[0], qkv[1], qkv[2], A)
            return (O - target).pow(2).mean()

        l0 = loss_of(leaves)
        l0.backward()
        with torch.no_grad():
            stepped = [t - 0.1 * t.grad for t in leaves]
        l1 = loss_of(stepped)
        assert l1.item() < l0.item(), f"loss did not decrease: {l0.item()} -> {l1.item()}"
    finally:
        impl.close()
