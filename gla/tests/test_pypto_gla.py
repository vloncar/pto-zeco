"""System test for the PyPTO DSL ZeCO / GLA forward.

Compiles and runs on the target platform selected via ``--platform`` (a2a3
hardware or a2a3sim simulator) across the devices given by ``--device``.

:class:`PyPtoZeCo` uses the chunk-recurrent GLA form in a hybrid composition —
a fused distributed ``stage1 + AllScan-ring`` program, then a ``@pl.jit`` stage2
(see :mod:`gla.implementations.pypto.impl`). ``P=1`` exercises the compute alone
(no exchange, no distributed program); ``P>=2`` exercises the full SP path.
Verified against the sequential :func:`gla.common.expected_gla` golden.

Each parametrization runs in its own forked process (``@pytest.mark.forked``):
within one process a ``@pl.jit`` dispatch (P=1 stage2) followed by a
``DistributedWorker.prepare()`` (P=2 stage1+ring) on the same devices is the
unsupported jit-then-prepare coexistence and hangs — forking gives each case a
clean device state. Runs on both ``--platform a2a3sim`` and ``a2a3`` (the earlier
a2a3sim deadlock on the chunk kernels was fixed upstream).
"""

import sys

import pytest
import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo


def _golden(Q, K, V, A):
    P, L, dv = V.shape
    return expected_gla(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A)
    ).reshape(P, L, dv)


@pytest.mark.forked
@pytest.mark.parametrize("P", [1, 2, 4])
def test_pypto_zeco(test_config, device_ids, P):
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")

    L, C, dk, dv = 32, 16, 16, 16   # N = L // C = 2 chunks
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)

    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids[:P], platform=test_config.platform)
    try:
        O = impl.forward(Q, K, V, A)
    finally:
        impl.close()

    expected = _golden(Q, K, V, A)
    # atol looser than the torch backends: the on-device chunk math divides by the
    # within-chunk cumulative decay, so FP32 rounding is larger than the reference's.
    assert torch.allclose(O, expected, atol=1e-2), (
        f"PyPTO ZeCO mismatch (P={P}): max diff = {(O - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
