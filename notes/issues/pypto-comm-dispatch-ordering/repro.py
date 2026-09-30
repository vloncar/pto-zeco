#!/usr/bin/env python3
"""Minimal reproducer: a rank's `pld.system.wait` dispatch is scheduled BEFORE that same
rank's `remote_store`+`notify` dispatch, so two ranks deadlock.

Two ranks, trivial `t+t` compute, one comm ring in each direction. Per rank the program is:

    dispatch 1  compute
    dispatch 2  SEND    (remote_store + notify to the peer)
    dispatch 3  compute
    dispatch 4  WAIT    (pld.system.wait on the peer's notify)
    dispatch 5  compute

Note there is NO cyclic dependency: **both ranks send at dispatch 2 with nothing in front of
the send, and only then wait at dispatch 4.** Both notifies are therefore issued long before
either rank waits, and this program cannot deadlock for any ordering reason.

It deadlocks anyway, on both ranks, with

    PTO2_ERROR_SCHEDULER_TIMEOUT (finalize_native_run failed with code -100)
    sub_class=S1:running-stalled completed=0/1 running=1 ready=0 waiting=0 orch_done=1

and the error line reports `dispatch_id=2` -- the SECOND dispatch on the rank -- while a
healthy run reports `dispatch_id=5`.

Run:  LD_PRELOAD=<cann>/lib64/libhccl.so python3 repro.py 0,1
Expect: hangs, then raises with code -100. A fixed build completes in ~2s and prints
`fwd_ok=True rev_ok=True`.
"""

from __future__ import annotations

import sys
import time

import torch

import pypto.language as pl
import pypto.language.distributed as pld
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

D, P, S = 16, 2, 8


def build():
    @pl.program
    class TwoRing:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            t = pl.load(X, [0, 0], [D, D])
            return pl.store(pl.add(t, t), [0, 0], Out)

        @pl.function(type=pl.FunctionType.InCore)
        def send(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[S, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            t = pl.load(X, [0, 0], [D, D])
            Echo = pl.store(t, [0, 0], Echo)
            pld.tile.remote_store(t, target=dst, peer=peer, offsets=[0, 0])
            pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1,
                              op=pld.NotifyOp.AtomicAdd)
            return Echo

        @pl.function(type=pl.FunctionType.InCore)
        def recv(
            self,
            OutD: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[S, 1], pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            return pl.store(pl.load(dst, [0, 0], [D, D]), [0, 0], OutD)

        @pl.function(type=pl.FunctionType.Orchestration)
        def c_compute(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.compute(X, Out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def c_send(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[S, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.send(X, Echo, dst, signal, peer)

        @pl.function(type=pl.FunctionType.Orchestration)
        def c_recv(
            self,
            OutD: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[S, 1], pl.INT32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.recv(OutD, dst, signal)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            X: pl.Tensor[[P, D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[P, D, D], pl.FP32]],
            Rf: pl.Out[pl.Tensor[[P, D, D], pl.FP32]],
            Rb: pl.Out[pl.Tensor[[P, D, D], pl.FP32]],
        ):
            fdst_buf = pld.alloc_window_buffer(D * D * 4)
            fsig_buf = pld.alloc_window_buffer(S * 4)
            bdst_buf = pld.alloc_window_buffer(D * D * 4)
            bsig_buf = pld.alloc_window_buffer(S * 4)

            T0 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            T1 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            T2 = pl.create_tensor([P, D, D], dtype=pl.FP32)

            for r in pl.range(P):
                fdst = pld.window(fdst_buf, [D, D], dtype=pl.FP32)
                fsig = pld.window(fsig_buf, [S, 1], dtype=pl.INT32)
                bdst = pld.window(bdst_buf, [D, D], dtype=pl.FP32)
                bsig = pld.window(bsig_buf, [S, 1], dtype=pl.INT32)

                if r == 0:
                    a0 = self.c_compute(X[r], T0[r], device=r)
                    self.c_send(a0, Echo[r], fdst, fsig, r + 1, device=r)   # -> rank 1
                    a1 = self.c_compute(a0, T1[r], device=r)
                    self.c_recv(Rb[r], bdst, bsig, device=r)                # <- rank 1
                    self.c_compute(a1, T2[r], device=r)
                else:
                    b0 = self.c_compute(X[r], T0[r], device=r)
                    self.c_send(b0, Echo[r], bdst, bsig, r - 1, device=r)   # -> rank 0
                    b1 = self.c_compute(b0, T1[r], device=r)
                    self.c_recv(Rf[r], fdst, fsig, device=r)                # <- rank 0
                    self.c_compute(b1, T2[r], device=r)
            return Echo, Rf, Rb

    return TwoRing


def main() -> int:
    devices = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")][:P]
    torch.manual_seed(5)
    X = torch.randn(P, D, D).share_memory_()
    Echo = torch.zeros(P, D, D).share_memory_()
    Rf = torch.zeros(P, D, D).share_memory_()
    Rb = torch.zeros(P, D, D).share_memory_()

    compiled = ir.compile(build(), platform="a2a3",
                          distributed_config=DistributedConfig(device_ids=devices,
                                                               num_sub_workers=0))
    rt = compiled.prepare()
    try:
        t0 = time.time()
        rt(X, Echo, Rf, Rb)
        fwd_ok = (Rf[1] - X[0] * 2.0).abs().max().item() < 1e-6
        rev_ok = (Rb[0] - X[1] * 2.0).abs().max().item() < 1e-6
        print(f"COMPLETED in {time.time() - t0:.1f}s  fwd_ok={fwd_ok} rev_ok={rev_ok}")
    finally:
        rt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
