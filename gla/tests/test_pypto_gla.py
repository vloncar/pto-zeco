"""System test for the PyPTO DSL ZeCO / GLA forward.

Compiles and runs on the target platform selected via ``--platform`` (a2a3
hardware or a2a3sim simulator) across the devices given by ``--device``.

:class:`PyPtoZeCo` runs the entire forward as ONE fully-fused distributed program —
``stage1 + AllScan-ring + stage2``, all distributed InCore chip kernels, no ``@pl.jit``
(see :mod:`gla.implementations.pypto.impl`). ``P=1`` is the native single-rank path
(no boundary exchange); ``P>=2`` exercises the full SP path. Verified against the
sequential :func:`gla.common.expected_gla` golden.

Each parametrization runs in its own forked process (``@pytest.mark.forked``) so every
case starts from clean device state (a fresh ``DistributedWorker.prepare()``/``close()``
cycle per config). Runs on both ``--platform a2a3sim`` and ``a2a3`` — the fully-fused
forward (including ``stage2`` as a distributed chip kernel) is verified on a2a3 hardware
at P=1/2/4 after the upstream sim-scheduler and HW dist-chip fixes.
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


def _run_case(platform, device_ids, P, L, C, dk, dv):
    """Build + run one config, returning the max abs error against the golden."""
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids[:P], platform=platform)
    try:
        O = impl.forward(Q, K, V, A)
    finally:
        impl.close()
    return (O - _golden(Q, K, V, A)).abs().max().item()


@pytest.mark.forked
@pytest.mark.parametrize("P", [1, 2, 4])
def test_pypto_zeco(test_config, device_ids, P):
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")

    L, C, dk, dv = 32, 16, 16, 16   # N = L // C = 2 chunks
    # atol looser than the torch backends: the on-device chunk math divides by the
    # within-chunk cumulative decay, so FP32 rounding is larger than the reference's.
    err = _run_case(test_config.platform, device_ids, P, L, C, dk, dv)
    assert err < 1e-2, f"PyPTO ZeCO mismatch (P={P}): max diff = {err}"


# F3.1 sizes. The chunk kernels hold the whole per-chunk working set in the 184 KB vector
# buffer, so the reachable shapes are bounded, not arbitrary; these are the corners that
# bound the region (see the F3.1 note in :mod:`gla.implementations.pypto.fused_program`
# for the byte accounting and what C=128 / D=128 would need).
SIZES = [
    (128, 32, 32, 32),    # the pre-F3.1 ceiling — regression guard
    (128, 32, 64, 64),    # D=64 was already reachable (only C drives the [C,C] tiles)
    (128, 64, 32, 32),    # C=64, small head dim
    (256, 64, 64, 64),    # C=64, D=64: the mainstream GLA config, N=4 chunks
]


@pytest.mark.forked
@pytest.mark.parametrize("L,C,dk,dv", SIZES, ids=lambda v: str(v))
@pytest.mark.parametrize("P", [1, 2])
def test_pypto_zeco_sizes(test_config, device_ids, P, L, C, dk, dv):
    """Correctness across the shapes F3.1 unlocked (P=1 local, P=2 with the real ring)."""
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")
    err = _run_case(test_config.platform, device_ids, P, L, C, dk, dv)
    assert err < 1e-2, f"PyPTO ZeCO mismatch (P={P} L={L} C={C} D={dv}): max diff = {err}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
