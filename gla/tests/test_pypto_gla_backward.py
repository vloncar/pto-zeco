"""System test for the PyPTO DSL ZeCO / GLA **backward** (B4).

:class:`PyPtoZeCo` runs the entire backward as ONE fully-fused distributed program —
``recompute + AllScan-ring + grad_o + reverse-ring + grad_h``, all distributed InCore chip
kernels, no host round-trip (see :mod:`gla.implementations.pypto.fused_backward_program`).
``P=1`` is the native single-rank path (no boundary, both rings gone); ``P>=2`` exercises
the full SP path, including the reverse ring whose message *is* the neighbour's adjoint.

Checked against :func:`gla.common.expected_gla_backward`, the analytic golden that
``gla/tests/test_gla_backward.py`` independently cross-checks against ``torch.autograd``.

Errors are compared **relatively**. Unlike the forward, the four gradients have very
different natural scales (``dA`` carries a ``1/a`` factor, so it runs an order of magnitude
above ``dQ``), and the on-device chunk math divides by the within-chunk cumulative decay,
so an absolute bound would be either vacuous for ``dA`` or unmeetable for ``dQ``.

Each parametrization runs in its own forked process (``@pytest.mark.forked``) so every case
starts from clean device state. Note this suite builds TWO ``DistributedWorker``s per config
— the forward's (from ``build``) and the backward's (lazily, on first ``backward``) — so it
is slower per case than the forward suite.
"""

import os
import sys

import pytest
import torch

from gla.common import ZeCoModule, expected_gla_backward, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

# See the forward suite: `make_gla_inputs` seeds torch itself, so without varying the seed
# every repeat replays ONE input point. Kept small by default to bound CI time.
REPEATS = int(os.environ.get("ZECO_TEST_REPEATS", "3"))


def _golden(Q, K, V, A, dO):
    P, L, dk = Q.shape
    dv = V.shape[2]
    gQ, gK, gV, gA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO))
    return (gQ.reshape(P, L, dk), gK.reshape(P, L, dk),
            gV.reshape(P, L, dv), gA.reshape(P, L, dk))


def _worst_rel(got, ref):
    """Max over the four gradients of ``max|got-ref| / max|ref|``, with the name."""
    worst, name = 0.0, ""
    for nm, g, r in zip(("dQ", "dK", "dV", "dA"), got, ref):
        rel = ((g - r).abs().max() / (r.abs().max() + 1e-6)).item()
        if rel > worst:
            worst, name = rel, nm
    return worst, name


def _run_case(platform, device_ids, P, L, C, dk, dv, seed=42, repeats=1):
    """Build once, dispatch `repeats` backwards on distinct seeds.

    Returns ``(worst_rel, which_grad, worst_seed)`` so a failure names both the gradient
    that drifted and the input that produced it.
    """
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids[:P], platform=platform)
    worst, name, worst_seed = 0.0, "", seed
    try:
        for i in range(repeats):
            s = seed + i
            Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=s)
            torch.manual_seed(s + 991)
            dO = torch.randn(P, L, dv, dtype=torch.float32)
            got = impl.backward(Q, K, V, A, dO)
            rel, nm = _worst_rel(got, _golden(Q, K, V, A, dO))
            if rel > worst:
                worst, name, worst_seed = rel, nm, s
    finally:
        impl.close()
    return worst, name, worst_seed


@pytest.mark.forked
@pytest.mark.parametrize("P", [1, 2, 4])
def test_pypto_zeco_backward(test_config, device_ids, P):
    """The small config across the rank counts: P=1 local, P=2/4 with the real rings."""
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")

    L, C, dk, dv = 32, 16, 16, 16   # N = L // C = 2 chunks
    err, nm, seed = _run_case(test_config.platform, device_ids, P, L, C, dk, dv,
                              repeats=REPEATS)
    assert err < 1e-3, (
        f"PyPTO ZeCO backward mismatch (P={P}, seed={seed}): worst rel err = {err:.3e} "
        f"on {nm}")


# Shapes. The backward's `grad_o` is the widest kernel in either direction (three [C,C]
# tiles plus ~12 [C,dk] and ~6 [dk,dv] tiles live at once), so it is expected to top out
# BELOW the forward's C=D=64 — these bound what actually fits, and a shape that stops
# compiling should be treated as the vector-buffer ceiling moving, not as a silent
# regression. Head dims below C are included deliberately: they are the shapes the carried
# pto-isa local-slot patch (MR !1457) protects, and the backward hits the `N < M` matmul
# predicate in more places than the forward does.
SIZES = [
    (128, 32, 32, 32),    # square, N=4 chunks
    (128, 32, 64, 32),    # dk != dv, both >= C
    (128, 32, 32, 64),    # the other rectangle
    (128, 16, 16, 16),    # small C, N=8 chunks — exercises the reverse walk
    (128, 32, 16, 32),    # dk < C — carried-patch regression guard
]


@pytest.mark.forked
@pytest.mark.parametrize("L,C,dk,dv", SIZES, ids=lambda v: str(v))
@pytest.mark.parametrize("P", [1, 2])
def test_pypto_zeco_backward_sizes(test_config, device_ids, P, L, C, dk, dv):
    """Correctness across shapes (P=1 local, P=2 with the real rings)."""
    if len(device_ids) < P:
        pytest.skip(f"need {P} devices, got {device_ids}")
    err, nm, seed = _run_case(test_config.platform, device_ids, P, L, C, dk, dv,
                              repeats=REPEATS)
    assert err < 1e-3, (
        f"PyPTO ZeCO backward mismatch (P={P} L={L} C={C} dk={dk} dv={dv} seed={seed}): "
        f"worst rel err = {err:.3e} on {nm}  "
        f"[{REPEATS} dispatches; set ZECO_TEST_REPEATS to widen]")


@pytest.mark.forked
def test_pypto_zeco_module_backward(test_config, device_ids):
    """``loss.backward()`` through :class:`gla.common.ZeCoModule` on the pypto kernels.

    The autograd wrapper is backend-agnostic, so with B4 landed pypto becomes a drop-in
    differentiable GLA operator rather than a forward-only one. This is the end-to-end
    claim: gradients arrive on the leaf tensors via the real fused device kernels, not via
    a traced forward. (``test_zeco_autograd.py`` does the rigorous finite-difference
    gradcheck on the CPU reference; a gradcheck here would need O(inputs) device forwards.)
    """
    P = 2 if len(device_ids) >= 2 else 1
    L, C, dk, dv = 64, 16, 16, 16
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids[:P], platform=test_config.platform)
    try:
        Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=11)
        leaves = [t.clone().requires_grad_(True) for t in (Q, K, V, A)]
        torch.manual_seed(23)
        dO = torch.randn(P, L, dv, dtype=torch.float32)
        ZeCoModule(impl)(*leaves).backward(dO)
        got = tuple(t.grad for t in leaves)
        assert all(g is not None for g in got), "autograd produced no gradient for some input"
        err, nm = _worst_rel(got, _golden(Q, K, V, A, dO))
        assert err < 1e-3, f"ZeCoModule backward worst rel err = {err:.3e} on {nm}"
    finally:
        impl.close()


@pytest.mark.forked
def test_pypto_zeco_backward_repeat_dispatch(test_config, device_ids):
    """Back-to-back backwards on ONE prepared worker stay correct.

    The forward has a race guard for exactly this (``test_pypto_allscan_back_to_back``):
    the reverse ring reuses its window and signal buffers on every dispatch, so a missing
    drain would show up only from the second dispatch on.
    """
    P = 2 if len(device_ids) >= 2 else 1
    L, C, dk, dv = 64, 16, 16, 16
    impl = PyPtoZeCo()
    impl.build(P, L, C, dk, dv, device_ids=device_ids[:P], platform=test_config.platform)
    try:
        Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=5)
        torch.manual_seed(77)
        dO = torch.randn(P, L, dv, dtype=torch.float32)
        ref = _golden(Q, K, V, A, dO)
        for i in range(6):
            err, nm = _worst_rel(impl.backward(Q, K, V, A, dO), ref)
            assert err < 1e-3, f"dispatch {i}: worst rel err = {err:.3e} on {nm}"
    finally:
        impl.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
