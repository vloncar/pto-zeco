#!/usr/bin/env python3
"""Is a persistent L3 worker usable for amortized benchmarking on this runtime?

Two things must hold for the benchmark's steady-state path, and they are separate:

  A. same-callable reuse  — stand a worker up once, dispatch the SAME callable N times
  B. multi-callable reuse — one worker serving several DIFFERENT callables

`allscan/issues/simpler-second-callable-silent-corruption/` says B silently returned
wrong data on runtime a756969c (2026-07-22). The runtime has since moved twice
(-> 8cdb306c -> 9922afdb), so both are re-checked here before anything is built on them.

CRITICAL ordering constraint (see memory `simpler-backend-runtime-port`): every
`share_memory_()` buffer must be allocated BEFORE `worker.init()`, because init()
eagerly forks the chip child and a buffer created after the fork is invisible to it
(errno 107017 / `run failed with code -1`). A persistent worker therefore has to
pre-allocate its IO buffers once and reuse them in place — which is exactly what the
pypto backend already does. Both probes below follow that ordering.

Inputs differ per dispatch so a stale-buffer bug cannot masquerade as success.

Usage: python3 scratchpad/probe_persistent_worker.py <device_id> [platform] [n_repeat]
"""

from __future__ import annotations

import sys

import torch

from gla.implementations.simpler.impl import (
    _SPECS,
    RUNTIME,
    CHUNK_H_SPEC,
    GATE_CUMSUM_SPEC,
)

L, C, DK, DV = 128, 32, 32, 32
N = L // DK * 0 + L // C


def _ref_cumsum(tril: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(g)
    for s in range(0, g.shape[0], C):
        out[s:s + C] = tril @ g[s:s + C]
    return out


def _stage(named, sig):
    """Allocate one shm buffer per arg (must happen BEFORE worker.init())."""
    staged = []
    for i, (_lbl, t) in enumerate(named):
        dt = torch.int64 if t.dtype == torch.int64 else torch.float32
        shm = torch.zeros(t.numel(), dtype=dt).share_memory_()
        shm.copy_(t.reshape(-1).to(dt))
        staged.append((t, shm, sig[i]))
    return staged


def _run(worker, cid, staged, tags):
    from simpler.task_interface import CallConfig, TaskArgs
    from simpler_setup.torch_interop import make_tensor_arg

    def orch_fn(orch, _args, cfg):
        chip_args = TaskArgs()
        for (_t, shm, d) in staged:
            chip_args.add_tensor(make_tensor_arg(shm), tags[d])
        orch.submit_next_level(cid, chip_args, cfg, worker=0)

    worker.run(orch_fn, args=None, config=CallConfig())


def probe_same_callable(device, platform, n_repeat):
    """A: one worker, one callable, N dispatches with alternating inputs."""
    from simpler.task_interface import ArgDirection, TensorArgType
    from simpler.worker import Worker
    from simpler_setup.scene_test import _compile_chip_callable_from_spec

    D = ArgDirection
    tags = {D.IN: TensorArgType.INPUT, D.OUT: TensorArgType.OUTPUT_EXISTING,
            D.INOUT: TensorArgType.INOUT}
    sig = GATE_CUMSUM_SPEC["orchestration"]["signature"]

    tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
    config = torch.tensor([C, DK, DV, N], dtype=torch.int64)
    torch.manual_seed(0)
    inputs = [torch.randn(L, DK) * 0.1 - 0.2, torch.randn(L, DK) * 0.1 - 0.2]

    g_buf = torch.zeros(L, DK)
    g_cs = torch.zeros(L, DK)
    named = [("tril", tril), ("g", g_buf), ("g_cs", g_cs), ("config", config)]
    staged = _stage(named, sig)                      # BEFORE init()
    g_shm = staged[1][1]
    out_shm = staged[2][1]

    cc = _compile_chip_callable_from_spec(_SPECS["gate_cumsum"], platform, RUNTIME,
                                          f"gate_cumsum:{platform}:{device}")
    worker = Worker(level=3, device_ids=[device], num_sub_workers=0,
                    platform=platform, runtime=RUNTIME)
    cid = worker.register(cc)
    worker.init()

    worst = 0.0
    try:
        for i in range(n_repeat):
            g = inputs[i % 2]
            g_shm.copy_(g.reshape(-1))               # reuse the SAME buffer in place
            out_shm.zero_()
            _run(worker, cid, staged, tags)
            got = out_shm.reshape(L, DK).clone()
            err = (got - _ref_cumsum(tril, g)).abs().max().item()
            worst = max(worst, err)
            print(f"  A dispatch {i} (input g{i % 2 + 1}): max_abs_err = {err:.4e} "
                  f"{'OK' if err < 1e-3 else '*** WRONG ***'}")
    finally:
        worker.close()
    return worst


def probe_multi_callable(device, platform):
    """B: one worker, TWO different callables, buffers for both staged pre-init()."""
    from simpler.task_interface import ArgDirection, TensorArgType
    from simpler.worker import Worker
    from simpler_setup.scene_test import _compile_chip_callable_from_spec

    D = ArgDirection
    tags = {D.IN: TensorArgType.INPUT, D.OUT: TensorArgType.OUTPUT_EXISTING,
            D.INOUT: TensorArgType.INOUT}

    tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
    config = torch.tensor([C, DK, DV, N], dtype=torch.int64)
    torch.manual_seed(1)
    g = torch.randn(L, DK) * 0.1 - 0.2
    K = torch.randn(L, DK)
    V = torch.randn(L, DV)

    gc_sig = GATE_CUMSUM_SPEC["orchestration"]["signature"]
    ch_sig = CHUNK_H_SPEC["orchestration"]["signature"]
    g_cs = torch.zeros(L, DK)
    s_snap = torch.zeros(N, DK, DV)

    gc_staged = _stage([("tril", tril), ("g", g), ("g_cs", g_cs), ("config", config)], gc_sig)
    ch_staged = _stage([("k", K), ("v", V), ("g_cs", g_cs), ("s_snap", s_snap),
                        ("config", config)], ch_sig)   # ALL buffers before init()

    worker = Worker(level=3, device_ids=[device], num_sub_workers=0,
                    platform=platform, runtime=RUNTIME)
    cids = {}
    for name in ("gate_cumsum", "chunk_h"):
        cc = _compile_chip_callable_from_spec(_SPECS[name], platform, RUNTIME,
                                              f"{name}:{platform}:{device}")
        cids[name] = worker.register(cc)
    worker.init()

    try:
        _run(worker, cids["gate_cumsum"], gc_staged, tags)
        got_gcs = gc_staged[2][1].reshape(L, DK).clone()
        err1 = (got_gcs - _ref_cumsum(tril, g)).abs().max().item()
        print(f"  B dispatch 0 (gate_cumsum): max_abs_err = {err1:.4e} "
              f"{'OK' if err1 < 1e-3 else '*** WRONG ***'}")

        # 2nd dispatch, DIFFERENT callable — the case the issue says is unsafe.
        ch_staged[2][1].copy_(got_gcs.reshape(-1))
        _run(worker, cids["chunk_h"], ch_staged, tags)
        s = ch_staged[3][1].reshape(N, DK, DV).clone()
        finite = bool(torch.isfinite(s).all())
        nonzero = float(s.abs().max())
        print(f"  B dispatch 1 (chunk_h, DIFFERENT callable): finite={finite} "
              f"max|s_snap|={nonzero:.4e}")
        return err1, finite, nonzero
    finally:
        worker.close()


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    n_repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    print("A: same-callable reuse (one worker, N dispatches, buffers reused in place)")
    worst_a = probe_same_callable(device, platform, n_repeat)
    ok_a = worst_a < 1e-3
    print(f"  -> same-callable reuse {'CORRECT' if ok_a else 'BROKEN'} (worst {worst_a:.3e})\n")

    print("B: multi-callable reuse (one worker, two different callables)")
    try:
        err1, finite, mx = probe_multi_callable(device, platform)
        ok_b = err1 < 1e-3 and finite and mx > 0.0
        print(f"  -> multi-callable dispatch {'ran without error' if ok_b else 'SUSPECT'}\n")
    except Exception as exc:  # noqa: BLE001 — the failure mode IS the result
        ok_b = False
        print(f"  -> multi-callable dispatch FAILED LOUDLY: {type(exc).__name__}: "
              f"{str(exc)[:160]}\n")

    print("VERDICT")
    print(f"  A same-callable reuse : {'USABLE' if ok_a else 'NOT usable'}")
    print(f"  B multi-callable      : {'usable' if ok_b else 'NOT usable'}")
    if ok_a:
        print("  => amortized measure() can be built by looping kernel-outermost /"
              " iteration-innermost, which only ever repeats the SAME callable.")
    return 0 if ok_a else 1


if __name__ == "__main__":
    raise SystemExit(main())
