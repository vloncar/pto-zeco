"""Smallest known-good pypto program, run before any HW sweep.

Cards 1, 4, 5 and 6 on this box fail EVERY pypto program at device bring-up
(`halMemCtl rc=42` in `init_aicore_register_addresses`) before a single kernel executes,
and both `npu-smi` (Health: OK) and the TaskQueue bad-card table fail to flag them. Without
a canary a bad card returns a full matrix of identical "failures" that read as a code
result — that has now cost two separate sweeps.

Exit code 99 is reserved for "the card is bad", so a submitting shell can retry elsewhere
instead of treating it as a real failure.
"""

from __future__ import annotations

BAD_CARD_EXIT = 99


def check(device: int, platform: str) -> bool:
    """True if the smallest known-good GLA config runs correctly on this device."""
    import torch

    from gla.common import expected_gla, flatten_seq, make_gla_inputs
    from gla.implementations.pypto.impl import PyPtoZeCo

    torch.manual_seed(0)
    Q, K, V, A = make_gla_inputs(1, 32, 16, 16)
    impl = PyPtoZeCo()
    try:
        impl.build(1, 32, 16, 16, 16, device_ids=[device], platform=platform)
        O = impl.forward(Q, K, V, A)
    except Exception as exc:  # noqa: BLE001 - that is the signal
        print(f"CANARY FAILED on device {device}: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:150]}")
        print("The smallest known-good config cannot run, so this is the CARD, not the code.")
        return False
    finally:
        impl.close()

    exp = expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                       flatten_seq(A)).reshape(1, 32, 16)
    err = (O - exp).abs().max().item()
    if err >= 1e-2:
        print(f"CANARY WRONG on device {device}: max diff {err:.3e} on a config that must pass.")
        return False
    print(f"canary OK on device {device} ({err:.2e})")
    return True
