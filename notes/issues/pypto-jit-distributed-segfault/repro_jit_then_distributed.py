"""Repro: a @pl.jit dispatch followed by a distributed program (DistributedWorker
.prepare(), which FORKS chip workers) on the SAME devices SEGFAULTS, instead of
either coexisting or raising a clean error.

The distributed program here is the project's PyPTO AllScan collective (a ready-made
minimal distributed program); the bug is not AllScan-specific — any
DistributedWorker.prepare() on a device that a @pl.jit runtime already touched
in-process reproduces it.

  --mode allscan_only     : AllScan on devices D  -> PASS (control)
  --mode jit_then_allscan : trivial @pl.jit on D, then AllScan on D -> SEGFAULT

Run from the pto-zeco repo root (for the allscan import), on hardware:
  cd pto-zeco && LD_PRELOAD=.../libhccl.so \
    python .../repro_jit_then_distributed.py --mode allscan_only     --devices 4,5
  cd pto-zeco && LD_PRELOAD=.../libhccl.so \
    python .../repro_jit_then_distributed.py --mode jit_then_allscan --devices 4,5

Observed (a2a3 hardware): allscan_only -> "OK"; jit_then_allscan -> Fatal Python
error: Segmentation fault, in
  distributed_runner.py __init__ -> worker.py _chip_process_loop -> task_interface.py init
"""
import argparse
import importlib.util
import os
import tempfile

import torch

_JIT_SRC = '''
import pypto.language as pl

DK = 16


@pl.jit
def add1(a: pl.Tensor[[DK, DK], pl.FP32], c: pl.Out[pl.Tensor[[DK, DK], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        c[:, :] = pl.add(a, 1.0)
    return c
'''


def _load_jit():
    fd, path = tempfile.mkstemp(prefix="repro_jit_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(_JIT_SRC)
    spec = importlib.util.spec_from_file_location("repro_jit_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.add1


def run_trivial_jit(devices, platform):
    from pypto.runtime.runner import RunConfig
    add1 = _load_jit()
    DK = 16
    a = torch.full((DK, DK), 2.0, dtype=torch.float32)
    for d in devices:
        c = torch.zeros((DK, DK), dtype=torch.float32)
        add1(a, c, config=RunConfig(platform=platform, device_id=d))
        assert abs(c[0, 0].item() - 3.0) < 1e-3, c[0, 0].item()
    print(f"[jit] trivial @pl.jit ran on devices {devices}")


def run_allscan(devices, platform):
    from allscan.implementations.pypto.impl import PyPtoAllscan
    P = len(devices)
    dk = dv = 16
    S = torch.zeros((P, dk, dv), dtype=torch.float32)
    g = torch.ones((P, dk, 1), dtype=torch.float32)
    out = torch.zeros((P, dk, dv), dtype=torch.float32)
    a = PyPtoAllscan()
    a.build(dk, dv, 1, P, devices, platform)   # <-- DistributedWorker.prepare() forks chip workers
    a.run(S, g, out)
    a.close()
    print(f"[allscan] AllScan ran on devices {devices}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["allscan_only", "jit_then_allscan"], required=True)
    ap.add_argument("--devices", default="4,5")
    ap.add_argument("--platform", default="a2a3")
    args = ap.parse_args()
    devices = [int(x) for x in args.devices.split(",")]

    if args.mode == "jit_then_allscan":
        run_trivial_jit(devices, args.platform)
    run_allscan(devices, args.platform)
    print("OK")


if __name__ == "__main__":
    main()
