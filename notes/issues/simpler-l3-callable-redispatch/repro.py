#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Can an L3 Worker serve several registered callables, and re-serve one of them?

Self-contained: uses only the runtime's own ``vector_example`` kernels, so it can be
dropped into ``tests/st/`` unchanged.

TWO callables are built from the SAME orchestration by permuting which AIV binary sits
at ``func_id`` 0 and 2.  ``kernel_add`` and ``kernel_mul`` take an identical argument
layout (args[0..2] tensors, trailing scalar ignored), so the permutation is signature-
safe, but the two callables compute *visibly different* functions:

    X  func0=add, func2=mul   ->  f = (a+b+1) * (a+b+2) + (a+b)
    Y  func0=mul, func2=add   ->  c = a*b ; f = ((c+1) + (c+2)) * c = (2c+3) * c

Distinct binaries at each func_id => distinct hashids, AND distinct goldens.  That
second half is the point: the runtime's own multi-callable coverage
(``tests/st/a2a3/tensormap_and_ringbuffer/dynamic_register``) dispatches two handles
whose kernels compute *the same* result, so it cannot observe a dispatch that silently
ran the wrong callable.  Here it can — every dispatch is checked against its own exact
golden, and on mismatch the observed value is also compared against the *other*
callable's golden, which distinguishes "ran the wrong binary" from "ran on stale args"
from "garbage".

Sequences exercised on ONE worker (all callables registered and all arg buffers
allocated before ``init()``, the supported ordering):

    single      X                 -- sanity: does a first dispatch work at all
    same2       X X               -- same callable twice
    multi2      X Y               -- second dispatch is a DIFFERENT callable
    redispatch  X Y X             -- callable re-dispatched after another ran  <-- the
                                     pattern a fused backward needs (gate_cumsum runs
                                     again for the reverse-cumsum after 3 other kernels)
    alternate   X Y X Y X Y       -- steady-state benchmark loop pattern

plus ``control``, which runs the ``redispatch`` sequence with a FRESH worker per
dispatch (one callable registered each time).  The control is the configuration this
project currently ships; it must pass, and if it does not, the failure is environmental
rather than a callable-staging bug.

Every dispatch uses different inputs, so a stale-buffer bug cannot masquerade as a pass.

Usage: python3 repro.py [device_id] [platform]
"""

from __future__ import annotations

import os
import sys

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable
from simpler.worker import Worker

from simpler_setup import TaskArgsBuilder, Tensor
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _build_l3_task_args

_RUNTIME = "tensormap_and_ringbuffer"
_SIZE = 128 * 128
_ORCH_SIG = [D.IN, D.IN, D.OUT]
_TOL = 1e-4


def _kernels_dir() -> str:
    """``examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels``.

    Located relative to the imported ``simpler`` package so the script works both
    inside the runtime tree and from this issue directory.
    """
    import simpler

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(simpler.__file__))))
    path = os.path.join(repo, "examples", "a2a3", "tensormap_and_ringbuffer",
                        "vector_example", "kernels")
    if not os.path.isdir(path):
        raise RuntimeError(f"vector_example kernels not found under {repo!r} (looked at {path})")
    return path


def build_callable(platform: str, *, swap_add_mul: bool) -> ChipCallable:
    """Compile the vector_example orchestration + 3 AIV kernels into a ChipCallable.

    ``swap_add_mul`` exchanges the binaries at func_id 0 and 2, which changes what the
    orchestration computes without changing any signature.
    """
    from simpler.task_interface import CoreCallable

    from simpler_setup.elf_parser import extract_text_section
    from simpler_setup.pto_isa import ensure_pto_isa_root

    kd = _kernels_dir()
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)

    orch_bytes = kc.compile_orchestration(
        runtime_name=_RUNTIME,
        source_path=os.path.join(kd, "orchestration", "example_orchestration.cpp"),
    )

    def _aiv(name: str) -> bytes:
        raw = kc.compile_incore(os.path.join(kd, "aiv", name), core_type="aiv",
                                pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs)
        return raw if platform.endswith("sim") else extract_text_section(raw)

    add = CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv("kernel_add.cpp"))
    add_scalar = CoreCallable.build(signature=[D.IN, D.OUT], binary=_aiv("kernel_add_scalar.cpp"))
    mul = CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv("kernel_mul.cpp"))

    first, third = (mul, add) if swap_add_mul else (add, mul)
    return ChipCallable.build(
        signature=_ORCH_SIG,
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, first), (1, add_scalar), (2, third)],
    )


def golden(kind: str, a: float, b: float) -> float:
    """Exact expected scalar for callable ``kind`` on constant inputs a, b."""
    if kind == "X":
        s = a + b
        return (s + 1.0) * (s + 2.0) + s
    c = a * b                      # kind == "Y": func0 is mul, func2 is add
    return ((c + 1.0) + (c + 2.0)) * c


def _make_args(a: float, b: float) -> TaskArgsBuilder:
    return TaskArgsBuilder(
        Tensor("a", torch.full((_SIZE,), a, dtype=torch.float32).share_memory_()),
        Tensor("b", torch.full((_SIZE,), b, dtype=torch.float32).share_memory_()),
        Tensor("f", torch.zeros(_SIZE, dtype=torch.float32).share_memory_()),
    )


# Distinct inputs per dispatch slot, chosen so X's and Y's goldens differ everywhere.
_INPUTS = [(2.0, 3.0), (5.0, 7.0), (1.5, 2.5), (4.0, 6.0), (3.0, 8.0), (2.5, 9.0)]

_SEQUENCES = {
    "single": ["X"],
    "same2": ["X", "X"],
    "multi2": ["X", "Y"],
    "redispatch": ["X", "Y", "X"],
    "alternate": ["X", "Y", "X", "Y", "X", "Y"],
}


def _classify(got: torch.Tensor, kind: str, a: float, b: float) -> tuple[bool, str]:
    """(ok, note) for one dispatch's output tensor."""
    want = golden(kind, a, b)
    other = golden("Y" if kind == "X" else "X", a, b)
    err = (got - want).abs().max().item()
    if err <= _TOL * max(1.0, abs(want)):
        return True, f"ok (err {err:.2e})"
    if (got - other).abs().max().item() <= _TOL * max(1.0, abs(other)):
        return False, f"WRONG: ran the OTHER callable (got {other:g}, want {want:g})"
    if not bool(torch.isfinite(got).all()):
        return False, f"WRONG: non-finite (want {want:g})"
    if float(got.abs().max()) == 0.0:
        return False, f"WRONG: output untouched/zero (want {want:g})"
    return False, f"WRONG: got {got.flatten()[0].item():g} .. want {want:g} (err {err:.2e})"


def run_persistent(platform: str, device: int, seq: list[str]) -> list[tuple[str, bool, str]]:
    """One worker, every callable in ``seq`` registered and every buffer staged pre-init."""
    callables = {k: build_callable(platform, swap_add_mul=(k == "Y")) for k in sorted(set(seq))}

    worker = Worker(level=3, device_ids=[device], num_sub_workers=0,
                    platform=platform, runtime=_RUNTIME)
    handles = {k: worker.register(cc) for k, cc in callables.items()}
    if len(handles) > 1:
        ids = {k: h.hashid for k, h in handles.items()}
        print(f"    handles: {ids}")
        if len(set(ids.values())) != len(ids):
            raise RuntimeError(f"callables did not get distinct hashids: {ids}")

    # One arg set per dispatch slot, ALL allocated before init() forks the chip child.
    argsets = [_make_args(*_INPUTS[i % len(_INPUTS)]) for i in range(len(seq))]
    chip_args = [_build_l3_task_args(a, _ORCH_SIG)[0] for a in argsets]

    worker.init()
    results: list[tuple[str, bool, str]] = []
    try:
        for i, kind in enumerate(seq):
            handle, ca = handles[kind], chip_args[i]
            worker.run(lambda o, _a, _c, _h=handle, _ca=ca: o.submit_next_level(
                _h, _ca, CallConfig(), worker=0))
            a, b = _INPUTS[i % len(_INPUTS)]
            ok, note = _classify(argsets[i].f, kind, a, b)
            results.append((f"{i}:{kind}", ok, note))
    finally:
        worker.close()
    return results


def run_fresh(platform: str, device: int, seq: list[str]) -> list[tuple[str, bool, str]]:
    """A brand-new single-callable worker per dispatch (the currently-shipped shape)."""
    results: list[tuple[str, bool, str]] = []
    for i, kind in enumerate(seq):
        cc = build_callable(platform, swap_add_mul=(kind == "Y"))
        worker = Worker(level=3, device_ids=[device], num_sub_workers=0,
                        platform=platform, runtime=_RUNTIME)
        handle = worker.register(cc)
        a, b = _INPUTS[i % len(_INPUTS)]
        argset = _make_args(a, b)
        ca, _ = _build_l3_task_args(argset, _ORCH_SIG)
        worker.init()
        try:
            worker.run(lambda o, _a, _c: o.submit_next_level(handle, ca, CallConfig(), worker=0))
        finally:
            worker.close()
        ok, note = _classify(argset.f, kind, a, b)
        results.append((f"{i}:{kind}", ok, note))
    return results


def main() -> int:
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"

    import simpler
    print(f"device={device} platform={platform}")
    print(f"simpler python tree: {os.path.dirname(os.path.abspath(simpler.__file__))}")
    print(f"kernels: {_kernels_dir()}\n")

    verdict: dict[str, str] = {}
    cases = [(n, s, run_persistent) for n, s in _SEQUENCES.items()]
    cases.append(("control", _SEQUENCES["redispatch"], run_fresh))

    for name, seq, runner in cases:
        shape = "fresh worker per dispatch" if runner is run_fresh else "one persistent worker"
        print(f"=== {name}: {' '.join(seq)}   ({shape})")
        try:
            results = runner(platform, device, seq)
        except Exception as exc:  # noqa: BLE001 -- a loud failure IS a result
            verdict[name] = f"RAISED {type(exc).__name__}"
            print(f"    RAISED {type(exc).__name__}: {str(exc)[:200]}\n")
            continue
        for label, ok, note in results:
            print(f"    dispatch {label}: {note}")
        bad = [label for label, ok, _ in results if not ok]
        verdict[name] = "PASS" if not bad else f"FAIL at dispatch {', '.join(bad)}"
        print(f"    -> {verdict[name]}\n")

    print("VERDICT")
    for name, v in verdict.items():
        print(f"  {name:<11}: {v}")
    return 0 if all(v == "PASS" for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
