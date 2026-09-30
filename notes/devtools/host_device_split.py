#!/usr/bin/env python3
"""Where does a ZeCO call actually spend its time — on the ordinary computer, or on the chips?

ROADMAP F6.6 asks whether the two backends put the same arithmetic in the same place.
They do not: ``simpler`` runs part of the GLA maths as host-side torch between device
dispatches, while ``pypto`` runs (nearly) all of it on device. Until the size of that
difference is known, no compute-vs-comm or kernel-vs-kernel split from ``gla/bench.py``
is defensible — a device-only figure flatters whichever backend offloads more.

This measures the size. It does **not** change either backend: it wraps their methods at
import time with accumulating timers, so the code under test is the shipped code.

Buckets, accumulated per timed call:

  simpler
    chip      ``_PersistentComputeRunner.run`` minus the lazy ``open()`` nested inside it
              — shared-memory staging + submit + wait for a compute dispatch
    open      standing a held compute worker back up after a boundary released it
    boundary  ``_boundary`` / ``_boundary_backward``: build + run + close of the AllScan
              worker (``_gammas`` is nested inside this, reported separately)
    release   ``_release_devices``: dropping the compute workers so the boundary can run
    host      everything left over = the leftover torch arithmetic F6.6 is about
      named:  ``_S_total``, ``_shift_snaps`` — the two pieces step 3 would port
      ``_gammas`` is host work in BOTH backends, so it is NOT a parity gap; it is broken
      out only so it can be excluded from the comparison.

  pypto
    chip      the prepared fused program's dispatch
    stage     ``_stage_inputs``: host gamma + the input copies into shared memory
      gamma:  the only GLA arithmetic pypto does off-device
    select    forward<->backward worker swap (must be 0 in steady state)
    host      everything left over (the output clones)

Usage: python3 devtools/host_device_split.py <dev_csv> <impl> <direction> [iters] [platform]
       impl:      simpler | pypto
       direction: forward | backward
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from collections import defaultdict

import torch

from gla.common import make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo
from gla.implementations.simpler import impl as simpler_mod
from gla.implementations.simpler.impl import SimplerZeCo

FWD_CONFIGS = [(2, 128, 32, 32), (4, 128, 32, 32), (2, 256, 32, 32),
               (4, 256, 32, 32), (2, 128, 32, 64), (4, 128, 32, 64)]
# pypto's backward exceeds the vector budget at D=64 (a real shape ceiling, ROADMAP task 5),
# so the backward comparison is the four D=32 configs.
BWD_CONFIGS = [(2, 128, 32, 32), (4, 128, 32, 32), (2, 256, 32, 32), (4, 256, 32, 32)]

ACC: defaultdict[str, float] = defaultdict(float)


def _timed(bucket, fn):
    """Wrap ``fn`` so its wall time accumulates into ``ACC[bucket]`` (ms)."""
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ACC[bucket] += (time.perf_counter() - t0) * 1e3
    return wrapper


# ---------------------------------------------------------------------------
# simpler: pure wrapping, no bodies copied
# ---------------------------------------------------------------------------

def patch_simpler():
    pr = simpler_mod._PersistentComputeRunner
    pr.run = _timed("run_total", pr.run)      # includes the lazy open() below
    pr.open = _timed("open", pr.open)
    simpler_mod._ComputeRunner.run = _timed("run_total", simpler_mod._ComputeRunner.run)

    s = SimplerZeCo
    s._boundary = _timed("boundary", s._boundary)
    s._boundary_backward = _timed("boundary", s._boundary_backward)
    s._release_devices = _timed("release", s._release_devices)
    s._gammas = _timed("gammas", s._gammas)   # nested inside boundary

    # Module-level functions: Python resolves the global at call time, so rebinding the
    # module attribute is enough — the call sites inside the methods pick it up.
    simpler_mod._S_total = _timed("S_total", simpler_mod._S_total)
    simpler_mod._shift_snaps = _timed("shift_snaps", simpler_mod._shift_snaps)


# ---------------------------------------------------------------------------
# pypto: one body is re-implemented (to split gamma from the copies) — guarded
# ---------------------------------------------------------------------------

class _RtProxy:
    """Times the prepared program's dispatch without touching pypto's code."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return self._inner(*a, **k)
        finally:
            ACC["chip"] += (time.perf_counter() - t0) * 1e3

    def close(self):
        return self._inner.close()


def _wrap_rt(obj):
    if obj._rt is not None and not isinstance(obj._rt, _RtProxy):
        obj._rt = _RtProxy(obj._rt)


def patch_pypto():
    # _stage_inputs is re-implemented so the host gamma can be separated from the plain
    # memcpys. That means a copied body, so assert the original still looks like this —
    # a silent drift here would misattribute pypto's only host arithmetic.
    src = inspect.getsource(PyPtoZeCo._stage_inputs)
    assert "A.prod(dim=1)" in src and src.count("copy_") == 5, (
        "pypto._stage_inputs has changed; update host_device_split.py to match:\n" + src)

    def stage(self, Q, K, V, A):
        t0 = time.perf_counter()
        gammas = A.prod(dim=1).reshape(self.P, self.dk, 1)
        t1 = time.perf_counter()
        self._h_Q.copy_(Q)
        self._h_K.copy_(K)
        self._h_V.copy_(V)
        self._h_A.copy_(A)
        self._h_g.copy_(gammas)
        t2 = time.perf_counter()
        ACC["gamma"] += (t1 - t0) * 1e3
        ACC["stage"] += (t2 - t0) * 1e3

    PyPtoZeCo._stage_inputs = stage

    orig_select = PyPtoZeCo._select

    def select(self, backward):
        t0 = time.perf_counter()
        orig_select(self, backward)
        ACC["select"] += (time.perf_counter() - t0) * 1e3
        _wrap_rt(self)

    PyPtoZeCo._select = select


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _inputs(P, L, C, D):
    Q, K, V, A = make_gla_inputs(P, L, D, D)     # make_gla_inputs seeds torch itself
    dO = torch.randn(P, L, D)
    return Q, K, V, A, dO


def _samples(call, iters, split_of):
    """Run ``call`` ``iters`` times, returning per-call (total_ms, bucket dict)."""
    out = []
    for _ in range(iters):
        ACC.clear()
        t0 = time.perf_counter()
        call()
        total = (time.perf_counter() - t0) * 1e3
        out.append((total, split_of(total, dict(ACC))))
    return out


def _simpler_split(total, acc):
    chip = acc.get("run_total", 0.0) - acc.get("open", 0.0)
    open_ms = acc.get("open", 0.0)
    boundary = acc.get("boundary", 0.0)
    release = acc.get("release", 0.0)
    host = total - acc.get("run_total", 0.0) - boundary - release
    return {
        "chip_ms": chip, "open_ms": open_ms, "boundary_ms": boundary,
        "release_ms": release, "host_ms": host,
        "named_S_total_ms": acc.get("S_total", 0.0),
        "named_shift_snaps_ms": acc.get("shift_snaps", 0.0),
        "gammas_in_boundary_ms": acc.get("gammas", 0.0),
    }


def _pypto_split(total, acc):
    stage = acc.get("stage", 0.0)
    chip = acc.get("chip", 0.0)
    select = acc.get("select", 0.0)
    return {
        "chip_ms": chip, "stage_ms": stage, "select_ms": select,
        "host_ms": total - chip - stage - select,
        "named_gamma_ms": acc.get("gamma", 0.0),
    }


def run_one(name, P, L, C, D, devices, platform, backward, iters):
    if name == "simpler":
        impl = SimplerZeCo()
        impl.build(P, L, C, D, D, devices[:P], platform)
        Q, K, V, A, dO = _inputs(P, L, C, D)

        def call():
            return impl.backward(Q, K, V, A, dO) if backward else impl.forward(Q, K, V, A)

        try:
            # Same gate bench.py uses: validates the held path against the per-kernel path
            # and leaves the held runners armed. Raises on a mismatch.
            impl._gate_persistent(call, (lambda g: tuple(g)) if backward else (lambda O: (O,)))
            call()                                   # warm, outside the timing
            rows = _samples(call, iters, _simpler_split)
        finally:
            impl._release_devices()
            impl.close()
    else:
        impl = PyPtoZeCo()
        impl.build(P, L, C, D, D, devices[:P], platform)
        _wrap_rt(impl)
        Q, K, V, A, dO = _inputs(P, L, C, D)

        def call():
            return impl.backward(Q, K, V, A, dO) if backward else impl.forward(Q, K, V, A)

        try:
            call()                                   # warm (also pays the direction swap)
            rows = _samples(call, iters, _pypto_split)
        finally:
            impl.close()
    return rows


def main() -> int:
    devices = [int(d) for d in sys.argv[1].split(",")]
    name = sys.argv[2]
    direction = sys.argv[3]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    platform = sys.argv[5] if len(sys.argv) > 5 else "a2a3"
    backward = direction == "backward"

    patch_simpler()
    patch_pypto()

    configs = [c for c in (BWD_CONFIGS if backward else FWD_CONFIGS) if c[0] <= len(devices)]
    print(f"impl={name} direction={direction} devices={devices} iters={iters}\n")

    out = []
    for (P, L, C, D) in configs:
        tag = f"P={P} L={L} C={C} D={D}"
        try:
            rows = run_one(name, P, L, C, D, devices, platform, backward, iters)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{tag}: FAILED {type(exc).__name__}: {exc}")
            continue
        totals = [t for t, _ in rows]
        mean_total = sum(totals) / len(totals)
        keys = rows[0][1].keys()
        mean = {k: sum(r[k] for _, r in rows) / len(rows) for k in keys}
        rec = {"impl": name, "direction": direction, "P": P, "L": L, "C": C, "D": D,
               "iters": iters, "total_ms": mean_total, **mean}
        out.append(rec)
        parts = "  ".join(f"{k[:-3]}={v:.2f}" for k, v in mean.items() if not k.startswith("named")
                          and not k.startswith("gammas"))
        print(f"{tag}: total={mean_total:.2f}ms  {parts}")
        named = {k: v for k, v in mean.items() if k.startswith("named") or k.startswith("gammas")}
        print("        " + "  ".join(f"{k}={v:.3f}" for k, v in named.items()))
        dest = os.environ.get("SPLIT_JSON")
        if dest:
            with open(dest, "w") as fh:
                json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
