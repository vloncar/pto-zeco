"""Tests for the ZeCO / GLA operator backward pass (Section 2, B1 + B2).

Pure CPU — runs on any platform. Verifies, in dependency order:

  * :func:`expected_gla_backward` (the analytic sequential golden) matches
    ``torch.autograd`` on :func:`expected_gla` — the ground-truth oracle;
  * the in-process :class:`TorchZeCo` backward (which composes the AllScan
    backward for the boundary-state gradient, mirroring its forward) matches the
    golden across rank counts, chunk sizes, and dimensions;
  * the ``torch.distributed`` (gloo) reverse-ring backward produces the same
    per-rank gradients as the golden.
"""

import sys

import pytest
import torch

from gla.common import (
    expected_gla,
    expected_gla_backward,
    flatten_seq,
    gla_chunk_backward,
    make_gla_inputs,
)
from gla.implementations.torch_ref import TorchZeCo, run_distributed_zeco_backward


def _autograd_grads(Q, K, V, A, dO):
    """Ground-truth (dQ,dK,dV,dA) via autograd through :func:`expected_gla`."""
    leaves = [t.clone().requires_grad_(True) for t in (Q, K, V, A)]
    O = expected_gla(*leaves)
    O.backward(dO)
    return tuple(leaf.grad for leaf in leaves)


def test_expected_gla_backward_matches_autograd():
    """The analytic sequential backward equals autograd on the forward."""
    torch.manual_seed(0)
    T, dk, dv = 24, 12, 20
    Q, K, V, A = make_gla_inputs(1, T, dk, dv)
    Q, K, V, A = Q[0], K[0], V[0], A[0]
    dO = torch.randn(T, dv)

    dQ, dK, dV, dA = expected_gla_backward(Q, K, V, A, dO)
    gQ, gK, gV, gA = _autograd_grads(Q, K, V, A, dO)

    for name, got, ref in [("dQ", dQ, gQ), ("dK", dK, gK), ("dV", dV, gV), ("dA", dA, gA)]:
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), (
            f"{name} mismatch: max diff = {(got - ref).abs().max().item():.3e}"
        )


@pytest.mark.parametrize("L,C,dk,dv", [(16, 8, 16, 16), (32, 16, 32, 24), (48, 16, 24, 32)])
def test_gla_chunk_backward_p1_matches_golden(L, C, dk, dv):
    """The explicit chunk-parallel backward (B3 kernel blueprint) reduces to the
    sequential golden for a single device (zero boundary)."""
    torch.manual_seed(L + C + dk)
    Q, K, V, A = make_gla_inputs(1, L, dk, dv)
    Q, K, V, A = Q[0], K[0], V[0], A[0]
    dO = torch.randn(L, dv)
    z_kv = torch.zeros(dk, dv)
    z_k = torch.zeros(dk)
    dQ, dK, dV, dA, _ = gla_chunk_backward(Q, K, V, A, C, z_kv, dO, z_kv, z_k)
    gQ, gK, gV, gA = expected_gla_backward(Q, K, V, A, dO)
    for name, got, ref in [("dQ", dQ, gQ), ("dK", dK, gK), ("dV", dV, gV), ("dA", dA, gA)]:
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-3), (
            f"{name} mismatch: max diff = {(got - ref).abs().max().item():.3e}"
        )


def test_gla_chunk_backward_sp_matches_autograd():
    """With a non-zero boundary, gla_chunk_backward equals autograd through the
    composed stage-A scan + boundary fold + stage-C reconstruct (the SP local half)."""
    from gla.common import gla_chunk_scan, gla_reconstruct

    torch.manual_seed(7)
    L, C, dk, dv = 32, 16, 24, 20
    Q, K, V, A = make_gla_inputs(1, L, dk, dv)
    Q, K, V, A = Q[0], K[0], V[0], A[0]
    dO = torch.randn(L, dv)
    S_recv = torch.randn(dk, dv) * 0.1
    dS_total = torch.randn(dk, dv) * 0.1
    dg_total = torch.randn(dk) * 0.1

    # Autograd oracle: same composition SimplerZeCo.backward differentiates.
    leaves = [t.clone().requires_grad_(True) for t in (Q, K, V, A)]
    Sr = S_recv.clone().requires_grad_(True)
    S_prev, c_prev, O_intra, S_total, g_total = gla_chunk_scan(*leaves, C)
    O = gla_reconstruct(leaves[0], leaves[3], C, S_prev, c_prev, Sr, O_intra)
    gQ, gK, gV, gA = torch.autograd.grad(
        [O, S_total, g_total], leaves, [dO, dS_total, dg_total])

    dQ, dK, dV, dA, _ = gla_chunk_backward(Q, K, V, A, C, S_recv, dO, dS_total, dg_total)
    for name, got, ref in [("dQ", dQ, gQ), ("dK", dK, gK), ("dV", dV, gV), ("dA", dA, gA)]:
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-3), (
            f"{name} mismatch: max diff = {(got - ref).abs().max().item():.3e}"
        )


def _golden_grads(Q, K, V, A, dO):
    """Full-sequence golden (dQ,dK,dV,dA), rank-major ``[P, L, .]``."""
    P, L, dv = V.shape
    dk = Q.shape[2]
    dQ, dK, dV, dA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO)
    )
    return (dQ.reshape(P, L, dk), dK.reshape(P, L, dk),
            dV.reshape(P, L, dv), dA.reshape(P, L, dk))


@pytest.mark.parametrize("C", [4, 8])
@pytest.mark.parametrize("dk,dv", [(16, 16), (8, 24)])
@pytest.mark.parametrize("P", [1, 2, 4])
def test_torch_zeco_backward_matches_golden(P, dk, dv, C):
    """In-process ZeCO backward across P devices matches the sequential golden."""
    L = 16
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    torch.manual_seed(P * 100 + dk + C)
    dO = torch.randn(P, L, dv)

    impl = TorchZeCo()
    impl.build(P, L, C, dk, dv, device_ids=list(range(max(P, 1))), platform="cpu")
    impl.forward(Q, K, V, A)
    dQ, dK, dV, dA = impl.backward(Q, K, V, A, dO)
    impl.close()

    gQ, gK, gV, gA = _golden_grads(Q, K, V, A, dO)
    for name, got, ref in [("dQ", dQ, gQ), ("dK", dK, gK), ("dV", dV, gV), ("dA", dA, gA)]:
        assert torch.allclose(got, ref, atol=1e-4), (
            f"{name} mismatch (P={P},C={C},dk={dk},dv={dv}): "
            f"max diff = {(got - ref).abs().max().item():.3e}"
        )


@pytest.mark.parametrize("P", [2, 4])
def test_torch_distributed_zeco_backward(P):
    """The gloo per-rank reverse ring agrees with the golden (spawns P processes)."""
    assert run_distributed_zeco_backward(P=P, L=16, C=8, dk=16, dv=16) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
