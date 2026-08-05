#!/usr/bin/env python3
"""F3.1 — isolate the new whole-chunk-decay path on hardware.

F3.1 replaced the two all-ones broadcast matmuls with

    gamma = exp(reshape(col_sum(log a), [DK, 1]))        # [DK, 1]
    S_new = row_expand_mul(S + (K/b)^T @ V, gamma)

Each piece is checked here on its own, smallest first, so a failure names the op rather
than the kernel:

  A. col_sum      -- colsum(la) as [1, DK]
  B. reshape      -- the same value reshaped to [DK, 1]
  C. gamma        -- exp of B (what the chunk kernels actually consume)
  D. row_expand   -- row_expand_mul(M, gamma) for an [DK, DV] M

E is the fallback route for the same [DK, 1] vector, avoiding ``reshape`` entirely:
row_sum over the transpose (``la^T`` is [DK, C], so its row sums are the column sums of
``la``). It costs one extra [DK, C] scratch tile, so it is only worth taking if reshape
turns out to be the broken link.

Usage: python3 scratchpad/f31_gamma_probe.py <device> [platform] [C] [DK] [DV]
"""

from __future__ import annotations

import sys

import pypto.language as pl
import torch


def build(C: int, DK: int, DV: int):
    @pl.program
    class GammaProbe:
        @pl.function(type=pl.FunctionType.InCore)
        def probe(
            self,
            A: pl.Tensor[[C, DK], pl.FP32],
            M: pl.Tensor[[DK, DV], pl.FP32],
            Ocs: pl.Out[pl.Tensor[[1, DK], pl.FP32]],
            Ors: pl.Out[pl.Tensor[[DK, 1], pl.FP32]],
            Ogm: pl.Out[pl.Tensor[[DK, 1], pl.FP32]],
            Oex: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
            Ort: pl.Out[pl.Tensor[[DK, 1], pl.FP32]],
        ) -> pl.Tensor[[DK, 1], pl.FP32]:
            a = pl.load(A, [0, 0], [C, DK])
            m = pl.load(M, [0, 0], [DK, DV])
            la = pl.log(a)
            cs = pl.tile.col_sum(la)                       # [1, DK]
            Ocs = pl.store(cs, [0, 0], Ocs)
            rs = pl.tile.reshape(cs, [DK, 1])              # [DK, 1]
            Ors = pl.store(rs, [0, 0], Ors)
            gamma = pl.exp(rs)
            Ogm = pl.store(gamma, [0, 0], Ogm)
            Oex = pl.store(pl.tile.row_expand_mul(m, gamma), [0, 0], Oex)
            # E: the reshape-free route to the same [DK, 1] vector.
            lat = pl.transpose(la, 0, 1)                   # [DK, C]
            tmp = pl.tile.create([DK, C], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            return pl.store(pl.exp(pl.tile.row_sum(lat, tmp)), [0, 0], Ort)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            A: pl.Tensor[[C, DK], pl.FP32],
            M: pl.Tensor[[DK, DV], pl.FP32],
            Ocs: pl.Out[pl.Tensor[[1, DK], pl.FP32]],
            Ors: pl.Out[pl.Tensor[[DK, 1], pl.FP32]],
            Ogm: pl.Out[pl.Tensor[[DK, 1], pl.FP32]],
            Oex: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
            Ort: pl.Out[pl.Tensor[[DK, 1], pl.FP32]],
        ) -> pl.Tensor[[DK, 1], pl.FP32]:
            return self.probe(A, M, Ocs, Ors, Ogm, Oex, Ort)

    return GammaProbe


def main() -> int:
    dev = int(sys.argv[1].split(",")[0])
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    C = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    DK = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    DV = int(sys.argv[5]) if len(sys.argv) > 5 else 16

    from pypto import ir
    from pypto.runtime.runner import RunConfig

    torch.manual_seed(0)
    A = torch.rand(C, DK, dtype=torch.float32) * 0.5 + 0.5   # decay factors in (0.5, 1.0]
    M = torch.randn(DK, DV, dtype=torch.float32)

    la = torch.log(A)
    exp_cs = la.sum(dim=0, keepdim=True)          # [1, DK]
    exp_rs = exp_cs.reshape(DK, 1)                # [DK, 1]
    exp_gm = torch.exp(exp_rs)
    exp_ex = M * exp_gm                            # row-wise broadcast

    compiled = ir.compile(build(C, DK, DV), platform=platform)
    Ocs = torch.zeros(1, DK)
    Ors = torch.zeros(DK, 1)
    Ogm = torch.zeros(DK, 1)
    Oex = torch.zeros(DK, DV)
    Ort = torch.zeros(DK, 1)
    compiled(A, M, Ocs, Ors, Ogm, Oex, Ort,
             config=RunConfig(platform=platform, device_id=dev))

    bad = 0
    for name, got, exp in (("A. col_sum      [1,DK]", Ocs, exp_cs),
                           ("B. reshape      [DK,1]", Ors, exp_rs),
                           ("C. gamma=exp    [DK,1]", Ogm, exp_gm),
                           ("D. row_expand_mul     ", Oex, exp_ex),
                           ("E. exp(row_sum(la^T)) ", Ort, exp_gm)):
        err = (got - exp).abs().max().item()
        ok = err < 1e-4
        bad += 0 if ok else 1
        print(f"{name}: max diff {err:.3e}  {'OK' if ok else '*** WRONG ***'}")
        if not ok:
            print(f"    got  {got.flatten()[:8].tolist()}")
            print(f"    want {exp.flatten()[:8].tolist()}")

    print(f"\nVERDICT: {'all four OK' if bad == 0 else f'{bad} stage(s) wrong'}"
          f"  (C={C} DK={DK} DV={DV} on {platform})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
