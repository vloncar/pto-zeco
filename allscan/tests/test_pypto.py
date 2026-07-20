"""System test for the PyPTO DSL AllScan.

Compiles and runs on the target platform selected via ``--platform`` (a2a3
hardware or a2a3sim simulator) across the devices given by ``--device``.
"""

import sys

import pytest
import torch

from allscan.common import expected_allscan, make_inputs
from allscan.implementations.pypto.impl import PyPtoAllscan


@pytest.mark.parametrize("K", [1, 2, 4])
def test_pypto_allscan(test_config, device_ids, K):
    if len(device_ids) < 2:
        pytest.skip(f"allscan needs at least 2 devices, got {device_ids}")

    P = len(device_ids)
    dk, dv = 64, 64
    S_locals, gammas, outputs = make_inputs(P, dk, dv)

    impl = PyPtoAllscan()
    impl.build(dk, dv, K, P, device_ids=device_ids, platform=test_config.platform)
    try:
        impl.run(S_locals, gammas, outputs)
    finally:
        impl.close()

    expected = expected_allscan(S_locals, gammas)
    assert torch.allclose(outputs, expected, atol=1e-5), (
        f"AllScan mismatch: max diff = {(outputs - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("K", [4])
def test_pypto_allscan_back_to_back(test_config, device_ids, K):
    """F1 race guard: many back-to-back dispatches must every one be correct.

    The cross-rank producer race (PTOAS #872 / #744) surfaced only under
    *repeated* dispatches — reproduced deterministically as back-to-back batched
    AllScan at 128^2 failing **262/640 (41%)** rings with no mitigation. With the
    ``TNOTIFY_IMPL`` DDR-fence stopgap in place this must be **0/N**. Fresh random
    inputs per dispatch so a stale (racy) cross-rank read produces a wrong output.
    """
    if len(device_ids) < 2:
        pytest.skip(f"allscan needs at least 2 devices, got {device_ids}")

    P = len(device_ids)
    dk = dv = 128          # 128^2 — the size at which the race reproduced
    n_dispatch = 32

    impl = PyPtoAllscan()
    impl.build(dk, dv, K, P, device_ids=device_ids, platform=test_config.platform)
    failures = []
    try:
        for i in range(n_dispatch):
            torch.manual_seed(1000 + i)
            S_locals = torch.rand((P, dk, dv), dtype=torch.float32)
            gammas = torch.rand((P, dk, 1), dtype=torch.float32)
            outputs = torch.zeros((P, dk, dv), dtype=torch.float32)
            impl.run(S_locals, gammas, outputs)
            max_diff = (outputs - expected_allscan(S_locals, gammas)).abs().max().item()
            if max_diff > 1e-4:
                failures.append((i, max_diff))
    finally:
        impl.close()

    assert not failures, (
        f"{len(failures)}/{n_dispatch} back-to-back dispatches wrong "
        f"(race regressed?): first few {failures[:5]}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
