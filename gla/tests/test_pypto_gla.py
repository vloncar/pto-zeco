"""System test for the PyPTO DSL ZeCO / GLA forward.

Compiles and runs on the target platform selected via ``--platform`` (a2a3
hardware or a2a3sim simulator) across the devices given by ``--device``.

:class:`PytoZeCo` uses the quadratic (whole-device-as-one-block) GLA form and
composes the PyPTO AllScan for the cross-device boundary hand-off. ``P=1``
exercises the compute alone (no exchange); ``P>=2`` exercises the full SP path.
Verified against the sequential :func:`gla.common.expected_gla` golden.
"""

import sys

import pytest
import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PytoZeCo


def _golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)
    ).reshape(P, L, dv)


@pytest.mark.parametrize("P", [1, 2])
def test_pypto_zeco(test_config, device_ids, P):
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")

    L, dk, dv = 32, 16, 16
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)

    impl = PytoZeCo()
    impl.build(P, L, L, dk, dv, device_ids=device_ids[:P], platform=test_config.platform)
    try:
        O = impl.forward(Q, K, V, A)
    finally:
        impl.close()

    expected = _golden(Q, K, V, A)
    # atol is looser than the torch backends: the quadratic form divides by the
    # device-global cumulative decay, so FP32 rounding is larger than the
    # recurrent reference's.
    assert torch.allclose(O, expected, atol=1e-2), (
        f"PyPTO ZeCO mismatch (P={P}): max diff = {(O - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
