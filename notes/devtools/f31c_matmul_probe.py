#!/usr/bin/env python3
"""F3.1c: is a matmul whose OUTPUT tile is taller than wide (N < M) wrong on a2a3?

The GLA shape matrix says the fused forward is wrong exactly when ``dv < C`` and is
independent of ``dk``:

    C=64 dk=32 dv=32  WRONG      C=64 dk=32 dv=64  OK
    C=64 dk=64 dv=32  WRONG      C=64 dk=64 dv=64  OK
    C=32 dk=16 dv=16  WRONG      C=32 dk=32 dv=32  OK

Every ``[C, dv]`` tile in the output path (``o_intra``, ``o_inter``, ``o_n``, the store to
``O``) is *taller than wide* in precisely the failing configs, and square or wide in the
passing ones. This strips the GLA away and tests that claim on its own: one matmul
``[M, K] @ [K, N] -> [M, N]``, swept over N < M, N == M and N > M.

Two variants per shape, because the GLA matmul result does not go straight to memory — it
crosses the cube->vector pipe and is consumed by vector ops:

  * ``pure``  — store the matmul result directly (cube -> GM).
  * ``mixed`` — apply a vector op first, so the result crosses the C2V pipe like the real
    kernel's does.

If only ``mixed`` is wrong, the defect is in the cross-core transport of a tall tile; if
both are, it is the matmul or its L0C readout. Either way this is a far smaller reproducer
than the GLA operator, and one an upstream reader can run.

Usage: python3 devtools/f31c_matmul_probe.py <device> <platform>
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl

# (M, K, N, note) — N is the output width, M the output height.
CASES = [
    (64, 64, 64, "square  (GLA C=64 dv=64 -> passes)"),
    (64, 64, 32, "TALL    (GLA C=64 dv=32 -> fails)"),
    (64, 64, 16, "TALL x4"),
    (64, 32, 32, "TALL, smaller K"),
    (32, 32, 32, "square  (GLA C=32 dv=32 -> passes)"),
    (32, 32, 16, "TALL    (GLA C=32 dv=16 -> fails)"),
    (32, 32, 64, "WIDE    (GLA C=32 dv=64 -> passes)"),
    (64, 64, 128, "WIDE x2"),
    (128, 64, 64, "TALL x2, larger M"),
]


def build(M: int, K: int, N: int, mixed: bool):
    src = f'''
@pl.program
class MatmulProbe:
    @pl.function(type=pl.FunctionType.InCore)
    def mm(
        self,
        X: pl.Tensor[[{M}, {K}], pl.FP32],
        Y: pl.Tensor[[{K}, {N}], pl.FP32],
        Z: pl.Out[pl.Tensor[[{M}, {N}], pl.FP32]],
    ) -> pl.Tensor[[{M}, {N}], pl.FP32]:
        x = pl.load(X, [0, 0], [{M}, {K}])
        y = pl.load(Y, [0, 0], [{K}, {N}])
        z = pl.matmul(x, y, out_dtype=pl.FP32)
        {"zv = pl.mul(z, 2.0)" if mixed else "zv = z"}
        return pl.store(zv, [0, 0], Z)

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip(
        self,
        X: pl.Tensor[[{M}, {K}], pl.FP32],
        Y: pl.Tensor[[{K}, {N}], pl.FP32],
        Z: pl.Out[pl.Tensor[[{M}, {N}], pl.FP32]],
    ) -> pl.Tensor[[{M}, {N}], pl.FP32]:
        return self.mm(X, Y, Z)
'''
    # pypto's frontend PARSES source rather than tracing, so a class built by exec() fails
    # with "Cannot retrieve source code for class". pl.parse is the supported way in.
    return pl.parse(src)


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2]

    from canary import BAD_CARD_EXIT, check
    if platform != "a2a3sim" and not check(device, platform):
        return BAD_CARD_EXIT

    from pypto import ir
    from pypto.runtime.runner import RunConfig

    print(f"=== matmul [M,K] @ [K,N] on {platform} dev {device}")
    print(f"{'M':>4} {'K':>4} {'N':>4}  {'pure':<12} {'mixed':<12} note")
    print("-" * 66)
    bad = 0
    for (M, K, N, note) in CASES:
        torch.manual_seed(7)
        X = torch.randn(M, K, dtype=torch.float32)
        Y = torch.randn(K, N, dtype=torch.float32)
        ref = X @ Y
        scale = max(ref.abs().max().item(), 1e-30)

        cell = {}
        for mixed in (False, True):
            want = ref * 2.0 if mixed else ref
            try:
                compiled = ir.compile(build(M, K, N, mixed), platform=platform)
                Z = torch.zeros(M, N, dtype=torch.float32)
                compiled(X, Y, Z, config=RunConfig(platform=platform, device_id=device))
                rel = (Z - want).abs().max().item() / (scale * (2.0 if mixed else 1.0))
                cell[mixed] = f"{rel:.2e}{'' if rel < 1e-3 else ' WRONG'}"
                bad += 0 if rel < 1e-3 else 1
            except Exception as exc:  # noqa: BLE001 - a build/run failure is a result
                cell[mixed] = f"ERR {type(exc).__name__}"
                bad += 1
        print(f"{M:>4} {K:>4} {N:>4}  {cell[False]:<12} {cell[True]:<12} {note}")

    print(f"\n{2 * len(CASES) - bad}/{2 * len(CASES)} OK")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
