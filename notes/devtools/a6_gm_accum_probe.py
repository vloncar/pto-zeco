#!/usr/bin/env python3
"""A6: can a kernel ACCUMULATE into its own output tensor across an outer block loop?

The backward has a conflict the forward does not. Two of its outputs contract over
different axes -- ``dV`` sums over the head dim, ``dK``/``dA`` sum over the value dim --
and A6 blocks both. Whichever loop is outermost, the other output's partial sums have to
survive across it, and holding them all live is exactly the tile we are trying to split.

The cheap way out is to let GM hold the running sum: load the output block back, add, store.
This probe asks two things the DSL docs do not answer:

1. Is ``pl.load`` from the SSA handle a ``pl.store`` returned even expressible?
2. If so, is the read-after-write ordered correctly INSIDE one InCore kernel, or does the
   pipeline let iteration j+1's load race iteration j's store?

Question 2 cannot be answered by compiling -- it needs the device. Golden is a plain sum,
so a missed dependency shows up as a missing term, not as noise.

Usage: a6_gm_accum_probe.py <platform> [device] [NB] [C] [DV]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

C, DV, NB = 128, 128, 4


def build(nb: int, c: int, dv: int):
    @pl.program
    class GmAccum:
        @pl.function(type=pl.FunctionType.InCore)
        def acc(
            self,
            X: pl.Tensor[[nb * c, dv], pl.FP32],
            O: pl.Out[pl.Tensor[[c, dv], pl.FP32]],
        ) -> pl.Tensor[[c, dv], pl.FP32]:
            """O (zeroed by the caller) += X[j] for every block j, via GM round-trips."""
            out = O
            for j in pl.range(0, nb):
                blk = pl.load(X, [j * c, 0], [c, dv])
                cur = pl.load(out, [0, 0], [c, dv])
                out = pl.store(pl.add(cur, blk), [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(
            self,
            X: pl.Tensor[[nb * c, dv], pl.FP32],
            O: pl.Out[pl.Tensor[[c, dv], pl.FP32]],
        ) -> pl.Tensor[[c, dv], pl.FP32]:
            return self.acc(X, O)

    return GmAccum


def main():
    global C, DV, NB
    platform = sys.argv[1] if len(sys.argv) > 1 else "a2a3"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if len(sys.argv) > 3:
        NB = int(sys.argv[3])
    if len(sys.argv) > 5:
        C, DV = int(sys.argv[4]), int(sys.argv[5])

    torch.manual_seed(7)
    X = torch.randn(NB * C, DV)
    gold = X.reshape(NB, C, DV).sum(0)

    print(f"building NB={NB} C={C} DV={DV}", flush=True)
    try:
        compiled = ir.compile(build(NB, C, DV), platform=platform)
    except Exception as exc:  # noqa: BLE001 -- a refusal IS the answer to question 1
        print(f"NOT EXPRESSIBLE: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        return 2
    print("compiled OK", flush=True)
    O = torch.zeros(C, DV)
    compiled(X, O, config=RunConfig(platform=platform, device_id=device_id))
    e = (O - gold).abs().max().item()
    # A dropped dependency loses whole blocks, so also report how many blocks landed.
    ratio = (O.abs().sum() / gold.abs().sum()).item()
    print(f"max|O - sum| = {e:.3e}   sum-ratio = {ratio:.4f}")
    print("PASS" if e < 1e-3 else "FAIL")
    return 0 if e < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
