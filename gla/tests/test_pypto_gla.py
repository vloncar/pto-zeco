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

import os
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


# `make_gla_inputs` seeds torch itself and defaults to seed=42, so a plain call gives every
# run the SAME input point. That matters here: the pto-isa FIFO aliasing this operator has
# tripped over is data- and timing-dependent, and reproduced as rarely as 1 dispatch in 20 —
# so a single dispatch on a single input is weak evidence of correctness. Each size case
# therefore dispatches REPEATS times against distinct seeds, and reports the seed that failed
# so it can be replayed. Kept small by default to bound CI time; raise it when hunting.
REPEATS = int(os.environ.get("ZECO_TEST_REPEATS", "3"))


def _run_case(platform, device_ids, P, L, C, dk, dv, seed=42, repeats=1):
    """Build once, dispatch `repeats` times on distinct seeds.

    Returns (worst_err, worst_seed) so a failure names the input that produced it.
    """
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids[:P], platform=platform)
    worst, worst_seed = 0.0, seed
    try:
        for i in range(repeats):
            s = seed + i
            Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=s)
            err = (impl.forward(Q, K, V, A) - _golden(Q, K, V, A)).abs().max().item()
            if err > worst:
                worst, worst_seed = err, s
    finally:
        impl.close()
    return worst, worst_seed


@pytest.mark.forked
@pytest.mark.parametrize("P", [1, 2, 4])
def test_pypto_zeco(test_config, device_ids, P):
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")

    L, C, dk, dv = 32, 16, 16, 16   # N = L // C = 2 chunks
    # atol looser than the torch backends: the on-device chunk math divides by the
    # within-chunk cumulative decay, so FP32 rounding is larger than the reference's.
    err, seed = _run_case(test_config.platform, device_ids, P, L, C, dk, dv, repeats=REPEATS)
    assert err < 1e-2, f"PyPTO ZeCO mismatch (P={P}, seed={seed}): max diff = {err}"


# F3.1 sizes. The chunk kernels hold the whole per-chunk working set in the 184 KB vector
# buffer, so the reachable shapes are bounded, not arbitrary; these are the corners that
# bound the region (see the F3.1 note in :mod:`gla.implementations.pypto.fused_program`
# for the byte accounting and what C=128 / D=128 would need).
SIZES = [
    (128, 32, 32, 32),    # the pre-F3.1 ceiling — regression guard
    (128, 32, 64, 64),    # D=64 was already reachable (only C drives the [C,C] tiles)
    (128, 32, 64, 32),    # dk != dv, both >= C — keeps asymmetric coverage in the suite
    (256, 64, 64, 64),    # C=64, D=64: the mainstream GLA config, N=4 chunks
]

# A head dim below C is silently wrong on a2a3 HARDWARE (correct on a2a3sim) — not an operator
# bug but a pto-isa one: a cross-core TPipe strides the consumer's local ring by the popped
# tile's own size instead of SLOT_SIZE, so a matmul's two operands alias in L1 whenever N < M
# (allscan/issues/pto-isa-fifo-local-slot-alias/, ROADMAP F3.1c + task 2; upstream issue #521).
# `build()` refuses these shapes rather than returning corrupt results, so they are kept here
# to assert that the guard fires. Turn them back into correctness cases once the upstream fix
# merges and the pto-isa pin moves past it.
#
# The two sides behave very differently, measured at 20 dispatches each:
#   dv < C  deterministic — 20/20 dispatches wrong.
#   dk < C  intermittent  — ~1/20, and some dk < C shapes stayed quiet across 20. At that rate
#           a clean run of 20 misses it 36% of the time, so quiet is NOT proof of safety and
#           the guard covers all of dk < C.
GUARDED_SIZES = [
    (128, 64, 32, 32),    # dv < C: C=64, dv=32
    (128, 32, 16, 16),    # dv < C: C=32, dv=16 — same defect at half the size
    (128, 64, 32, 64),    # dk < C: was a correctness case until task 2 showed it corrupts
    (128, 32, 16, 32),    # dk < C at C=32 — the second shape that reproduced
]


@pytest.mark.forked
@pytest.mark.parametrize("L,C,dk,dv", SIZES, ids=lambda v: str(v))
@pytest.mark.parametrize("P", [1, 2])
def test_pypto_zeco_sizes(test_config, device_ids, P, L, C, dk, dv):
    """Correctness across the shapes F3.1 unlocked (P=1 local, P=2 with the real ring)."""
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")
    err, seed = _run_case(test_config.platform, device_ids, P, L, C, dk, dv, repeats=REPEATS)
    assert err < 1e-2, (
        f"PyPTO ZeCO mismatch (P={P} L={L} C={C} dk={dk} dv={dv} seed={seed}): "
        f"max diff = {err}  [{REPEATS} dispatches; set ZECO_TEST_REPEATS to widen]"
    )


@pytest.mark.parametrize("L,C,dk,dv", GUARDED_SIZES, ids=lambda v: str(v))
def test_pypto_zeco_rejects_head_dim_below_C(test_config, device_ids, L, C, dk, dv):
    """A shape that the hardware computes incorrectly must be refused, not silently run.

    No device needed — the guard fires in build() before anything is dispatched.
    """
    impl = PyPtoZeCo()
    with pytest.raises(AssertionError, match="silently WRONG"):
        impl.build(1, L, C, dk, dv, device_ids=device_ids[:1] or [0],
                   platform=test_config.platform)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
