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
# 3 by default to bound CI time. Note this is thin for the `dk < C` shapes above, which
# corrupted only ~1 dispatch in 20 before the fix: 3 repeats would miss a regression there
# ~86% of the time. Raise ZECO_TEST_REPEATS when validating the carried pto-isa patch.
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
    (128, 32, 64, 32),    # dk != dv, both >= C
    (256, 64, 64, 64),    # C=64, D=64: the mainstream GLA config, N=4 chunks
    # Head dim 128 — unreachable until the chunk kernels blocked `dk` (task 5, 2026-08-21).
    # It needs BOTH levers: 4 blocks over the head dim AND the cube<->vector ring cut to depth
    # 1, which alone is 131072 B of the 188416 B budget. If either regresses this goes red
    # with "no blocking plan fits" rather than a wrong answer.
    (256, 64, 128, 64),   # dk=128
    # dv=128 needed a second thing: V staged in L1 rather than the vector buffer. Reaching it
    # by blocking alone was impossible -- three copies of the [dk,dv] state are 60% of the
    # budget and no amount of head-dim blocking touches them.
    (256, 64, 64, 128),   # dv=128
    # BOTH head dims at 128. Out of reach at any blocking while stage2 CARRIED its state:
    # three [dk,dv] copies are 60% of the budget and blocking the head dim does not touch a
    # loop carry. Reachable since stage2 rebuilds each chunk's state from stage1's snapshot
    # (A1, 2026-08-21) and holds only [BK,DV] at a time. If the snapshot path regresses this
    # goes red with "no blocking plan fits".
    (256, 64, 128, 128),  # dk = dv = 128
    # dv=256 is the first shape that needs the VALUE dim blocked as well: at one value block
    # the [C,dv] right-hand operand of `scores @ V` is 65536 B, the entire L0 buffer. The
    # chosen plan here should be 2 value blocks; if value blocking regresses, this goes red.
    (256, 64, 64, 256),   # dv=256 — forces the value split
    (192, 48, 48, 48),    # a multiple of 16 that is not a power of two
    # Both head dims below C. These were REFUSED by build() until 2026-08-13, because the
    # pto-isa FIFO local-slot aliasing (issue #521) made them silently corrupt: dv < C on
    # every dispatch, dk < C about 1 in 20. !1457 is merged and the fix is carried against
    # our pin, so they are correctness cases again — and they are the regression test for
    # that carried patch. If it is ever dropped, these are what should go red.
    (128, 64, 32, 32),    # dv < C  (and dk < C) — was the original F3.1c failure
    (128, 32, 16, 16),    # dv < C at half the size
    (128, 64, 32, 64),    # dk < C only — the intermittent side, needs the repeats below
    (128, 32, 16, 32),    # dk < C only, at C=32
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
