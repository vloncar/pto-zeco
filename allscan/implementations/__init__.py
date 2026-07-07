"""AllScan implementation registry.

Each backend is imported defensively so that a missing dependency (e.g. the PTO
runtime not being installed) drops only that backend instead of breaking the
whole suite. ``REGISTRY`` is the ordered list of available implementations.
"""

from __future__ import annotations

import warnings

from allscan.common import AllscanImpl

REGISTRY: list[type[AllscanImpl]] = []


def _try_register(import_path: str, class_name: str) -> None:
    import importlib

    try:
        module = importlib.import_module(import_path)
        REGISTRY.append(getattr(module, class_name))
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        warnings.warn(f"AllScan backend {class_name} unavailable: {exc}", stacklevel=2)


_try_register("allscan.implementations.torch_ref", "TorchAllscan")
_try_register("allscan.implementations.pypto.impl", "PyPtoAllscan")
_try_register("allscan.implementations.simpler.impl", "SimplerAllscan")
