"""Tests for the torch reference ZeCO / GLA operator (Phases 1 & 2).

Pure CPU — runs on any platform/device selection. Verifies that:

  * the chunk-wise scan reduces to the plain recurrent GLA on a single device
    (``P=1``), locking the chunk math;
  * the in-process :class:`TorchZeCo` (which composes the AllScan backend for the
    boundary hand-off) matches the sequential :func:`expected_gla` golden across
    rank counts, chunk sizes, and dimensions;
  * the ``torch.distributed`` (gloo) forward ring produces the same result as
    the golden.
"""

import sys

import pytest
import torch

from gla.common import (
    expected_gla,
    flatten_seq,
    gla_chunk_scan,
    gla_reconstruct,
    make_gla_inputs,
)
from gla.implementations.torch_ref import TorchZeCo, run_distributed_zeco


def _golden(Q, K, V, A):
    """Full-sequence recurrent GLA, reshaped back to rank-major ``[P, L, dv]``."""
    P, L, dv = V.shape
    return expected_gla(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)
    ).reshape(P, L, dv)


@pytest.mark.parametrize("C", [4, 8, 16])
def test_chunk_scan_reduces_to_recurrent(C):
    """Single device: chunk scan + reconstruct == plain recurrent GLA."""
    P, L, dk, dv = 1, 16, 16, 16
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    S_prev, c_prev, O_intra, _, _ = gla_chunk_scan(Q[0], K[0], V[0], A[0], C)
    S_recv = torch.zeros((dk, dv), dtype=torch.float32)  # no cross-device history
    O = gla_reconstruct(Q[0], A[0], C, S_prev, c_prev, S_recv, O_intra)
    expected = expected_gla(Q[0], K[0], V[0], A[0])
    assert torch.allclose(O, expected, atol=1e-4), (
        f"chunk scan mismatch: max diff = {(O - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("C", [4, 8])
@pytest.mark.parametrize("dk,dv", [(16, 16), (8, 24)])
@pytest.mark.parametrize("P", [1, 2, 4])
def test_torch_zeco_matches_golden(P, dk, dv, C):
    """In-process ZeCO across P devices matches the sequential golden."""
    L = 16
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)

    impl = TorchZeCo()
    impl.build(P, L, C, dk, dv, device_ids=list(range(max(P, 1))), platform="cpu")
    O = impl.forward(Q, K, V, A)
    impl.close()

    expected = _golden(Q, K, V, A)
    assert torch.allclose(O, expected, atol=1e-4), (
        f"ZeCO mismatch (P={P},C={C},dk={dk},dv={dv}): "
        f"max diff = {(O - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("P", [2, 4])
def test_torch_distributed_zeco(P):
    """The gloo per-rank ring agrees with the golden (spawns P processes)."""
    assert run_distributed_zeco(P=P, L=16, C=8, dk=16, dv=16) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
