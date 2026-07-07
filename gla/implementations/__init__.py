"""ZeCO / GLA implementation registry.

Each backend is imported defensively so a missing dependency drops only that
backend. ``REGISTRY`` is the ordered list of available implementations.
"""

from __future__ import annotations

import warnings

from gla.common import ZeCoImpl

REGISTRY: list[type[ZeCoImpl]] = []


def _try_register(import_path: str, class_name: str) -> None:
    import importlib

    try:
        module = importlib.import_module(import_path)
        REGISTRY.append(getattr(module, class_name))
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        warnings.warn(f"ZeCO backend {class_name} unavailable: {exc}", stacklevel=2)


_try_register("gla.implementations.torch_ref", "TorchZeCo")
_try_register("gla.implementations.pypto.impl", "PyPtoZeCo")
# Simpler-runtime backend: hand-written PTO-ISA kernels + real AllScan boundary.
# Base env (no torch-npu); needs LD_PRELOAD=libhccl.so for P>1 on HW.
_try_register("gla.implementations.simpler.impl", "SimplerZeCo")
