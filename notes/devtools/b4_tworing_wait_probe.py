#!/usr/bin/env python3
"""Do TWO opposite-direction comm rings in one program deadlock, with REAL waits?

Phase 1 bisect on the fused backward: the forward ring alone completes (2.4s), the reverse
ring alone completes (2.3s), both together stall with PTO2_ERROR_SCHEDULER_TIMEOUT
(`sub_class=S1:running-stalled completed=0/1 running=1 orch_done=1`). So the interaction is
the fault, not either ring.

This asks whether that is a property of TWO RINGS AS SUCH -- in which case it is a framework
bug with a small reproducer -- or of B4's particular kernels.

An earlier probe (`b4_tworing_probe`) reported "two rings pass, both transfers exact" and
that verdict was used to rule two rings out. It was WRONG: it replaced `pld.system.wait`
with a plain load, so it never exercised the waits and merely measured whether the timing
happened to work. Every wait here is a real `pld.system.wait`, which is the whole point.

Structure mirrors the fused backward at P=2, with trivial compute:

    rank 0:  compute -> SEND fwd->1 -> compute -> WAIT rev<-1 -> compute
    rank 1:  compute -> WAIT fwd<-0 -> compute -> SEND rev->0 -> compute

Variants:
    fwd    forward ring only  (rank 1 waits; rank 0 never waits)   -- expect completes
    rev    reverse ring only  (rank 0 waits; rank 1 never waits)   -- expect completes
    both   both rings, one window pair each                        -- the question

  both stalls, fwd+rev complete -> MINIMAL REPRODUCER of a two-ring deadlock; file it.
  all three complete            -> two rings as such are fine; the fault is in B4's kernels
                                   (blocked K loop, GM tensors written by earlier dispatches).

A stall here does NOT complete, by design -- it is a real deadlock, so the runner must bound
it with `timeout` and read a non-zero exit as the signal.

Usage: python3 devtools/b4_tworing_wait_probe.py <variant> <device_csv> [platform]
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


def build(do_fwd: int, do_rev: int, swap_r1: int = 0):
    """`do_fwd`/`do_rev`/`swap_r1` are closure constants, so the class body sees literals.

    `swap_r1` reorders rank 1 to SEND before it WAITS, so neither rank has a wait ahead of
    its own send. The program cannot deadlock in either order -- rank 0's forward send has
    no dependency and already precedes its own wait -- so if the default order hangs and
    this one does not, a rank blocked in a wait is failing to make its EARLIER send visible
    to the peer.
    """

    @pl.program
    class TwoRingWait:
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
            """A REAL wait, exactly as the ring uses it -- this is what the old probe
            replaced with a plain load, invalidating its verdict."""
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
                    if do_fwd == 1:
                        self.c_send(a0, Echo[r], fdst, fsig, r + 1, device=r)
                    a1 = self.c_compute(a0, T1[r], device=r)
                    if do_rev == 1:
                        self.c_recv(Rb[r], bdst, bsig, device=r)
                    self.c_compute(a1, T2[r], device=r)
                else:
                    b0 = self.c_compute(X[r], T0[r], device=r)
                    if swap_r1 == 1:
                        self.c_send(b0, Echo[r], bdst, bsig, r - 1, device=r)
                    if swap_r1 == 0:
                        if do_fwd == 1:
                            self.c_recv(Rf[r], fdst, fsig, device=r)
                    b1 = self.c_compute(b0, T1[r], device=r)
                    if swap_r1 == 1:
                        self.c_recv(Rf[r], fdst, fsig, device=r)
                    if swap_r1 == 0:
                        if do_rev == 1:
                            self.c_send(b1, Echo[r], bdst, bsig, r - 1, device=r)
                    self.c_compute(b1, T2[r], device=r)
            return Echo, Rf, Rb

    return TwoRingWait


def build_one(rev_only: int, pad: int = 0):
    """Single-ring control: TWO window buffers only.

    A separate builder is needed because `MaterializeCommDomainScopes` rejects any
    `alloc_window_buffer` that no chip_orch dispatch consumes, and an `if` around an alloc
    is real IR control flow (ConvertToSSA then rejects the cross-branch use). So the
    one-ring case cannot be expressed by gating the 4-buffer body -- it needs its own.
    """

    @pl.program
    class OneRingWait:
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
            dst_buf = pld.alloc_window_buffer(D * D * 4)
            sig_buf = pld.alloc_window_buffer(S * 4)

            T0 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            T1 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            T2 = pl.create_tensor([P, D, D], dtype=pl.FP32)
            T3 = pl.create_tensor([P, D, D], dtype=pl.FP32)

            for r in pl.range(P):
                dst = pld.window(dst_buf, [D, D], dtype=pl.FP32)
                sig = pld.window(sig_buf, [S, 1], dtype=pl.INT32)

                if r == 0:
                    a0 = self.c_compute(X[r], T0[r], device=r)
                    if rev_only == 0:
                        self.c_send(a0, Echo[r], dst, sig, r + 1, device=r)
                    a1 = self.c_compute(a0, T1[r], device=r)
                    if rev_only == 1:
                        self.c_recv(Rb[r], dst, sig, device=r)
                    a2 = self.c_compute(a1, T2[r], device=r)
                    if pad == 1:
                        self.c_compute(a2, T3[r], device=r)
                else:
                    b0 = self.c_compute(X[r], T0[r], device=r)
                    if rev_only == 0:
                        self.c_recv(Rf[r], dst, sig, device=r)
                    b1 = self.c_compute(b0, T1[r], device=r)
                    if rev_only == 1:
                        self.c_send(b1, Echo[r], dst, sig, r - 1, device=r)
                    b2 = self.c_compute(b1, T2[r], device=r)
                    if pad == 1:
                        self.c_compute(b2, T3[r], device=r)
            return Echo, Rf, Rb

    return OneRingWait



def build_dep(swap_r1: int):
    """THE MECHANISM TEST: identical to `build`, plus one artificial data dependency.

    Diagnosis (read out of the runtime's own contract, not guessed):
    pypto lowers each chip dispatch to `orch.submit_next_level`, whose deps come ONLY from
    tensor tags (`runtime/docs/orchestrator.md`). A comm window carries no edge, so program
    order in host_orch is discarded. The scheduler then routes a task to the per-worker FIFO
    as soon as it is READY -- and a `c_recv` dispatch takes only `Out` + the two windows, so
    NOTHING produces its inputs: fan-in 0, READY at submission. It therefore enters FIFO[r]
    ahead of the rank's own `c_send`, which is still PENDING behind the compute feeding it.
    One task per worker (`scheduler.md`: dispatch only when the worker is idle, strict head)
    => the spin-wait owns the core and the send never runs.

    This threads the send's own output tensor into the recv as an extra INPUT, creating a RAW
    edge send -> recv. The recv is no longer fan-in-free, so it cannot jump its rank's send.
    NOTHING ELSE CHANGES -- same kernels, same 4 buffers, same 5 dispatches, same waits.

      dep arms pass while `both`/`sendfirst` still deadlock -> mechanism CONFIRMED, and the
          fix is "give a rank's comm dispatches program-order edges"
      dep arms still deadlock -> the fan-in story is wrong; re-open the mechanism

    The dependency is payload-neutral: `recv` returns `dst + (dep - dep)`, so the transferred
    values are bit-identical to the undecorated arms and the existing checks still apply.
    """

    @pl.program
    class DepWait:
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
        def recv_dep(
            self,
            OutD: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[S, 1], pl.INT32],
            Dep: pl.Tensor[[D, D], pl.FP32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            """`Dep` exists only to create the scheduling edge. Subtracting it from itself
            keeps the stored payload exactly `dst`, so it cannot mask a delivery failure --
            and it keeps the tensor genuinely USED, so no pass can drop the argument and
            silently take the edge with it."""
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            d = pl.load(Dep, [0, 0], [D, D])
            zero = pl.sub(d, d)
            return pl.store(pl.add(pl.load(dst, [0, 0], [D, D]), zero), [0, 0], OutD)

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
        def c_recv_dep(
            self,
            OutD: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            dst: pld.DistributedTensor[[D, D], pl.FP32],
            signal: pld.DistributedTensor[[S, 1], pl.INT32],
            Dep: pl.Tensor[[D, D], pl.FP32],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.recv_dep(OutD, dst, signal, Dep)

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
                    # Capturing the send's Out is what creates the edge.
                    e0 = self.c_send(a0, Echo[r], fdst, fsig, r + 1, device=r)
                    a1 = self.c_compute(a0, T1[r], device=r)
                    self.c_recv_dep(Rb[r], bdst, bsig, e0, device=r)
                    self.c_compute(a1, T2[r], device=r)
                else:
                    b0 = self.c_compute(X[r], T0[r], device=r)
                    if swap_r1 == 1:
                        # rank 1 also sends before it waits -> its recv gets the same edge.
                        e1 = self.c_send(b0, Echo[r], bdst, bsig, r - 1, device=r)
                        b1 = self.c_compute(b0, T1[r], device=r)
                        self.c_recv_dep(Rf[r], fdst, fsig, e1, device=r)
                        self.c_compute(b1, T2[r], device=r)
                    else:
                        # rank 1's recv is already FIRST in its own program order, so it needs
                        # no ordering edge -- depend on b0 purely to keep the arg shape equal.
                        self.c_recv_dep(Rf[r], fdst, fsig, b0, device=r)
                        b1 = self.c_compute(b0, T1[r], device=r)
                        self.c_send(b1, Echo[r], bdst, bsig, r - 1, device=r)
                        self.c_compute(b1, T2[r], device=r)
            return Echo, Rf, Rb

    return DepWait


def build_samedir():
    """Control B: TWO window pairs, 5 dispatches -- but BOTH rings run 0 -> 1, so only
    rank 1 ever waits. Holds buffer count (4) and dispatch count (5) equal to `both`, and
    varies only whether BOTH ranks wait.

      completes -> the trigger is BOTH RANKS WAITING, not 4 buffers and not 5 dispatches
      deadlocks -> the trigger is the window/dispatch count, not the wait topology
    """

    @pl.program
    class SameDirWait:
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
                    self.c_send(a0, Echo[r], fdst, fsig, r + 1, device=r)
                    a1 = self.c_compute(a0, T1[r], device=r)
                    self.c_send(a1, Rb[r], bdst, bsig, r + 1, device=r)
                    self.c_compute(a1, T2[r], device=r)
                else:
                    b0 = self.c_compute(X[r], T0[r], device=r)
                    self.c_recv(Rf[r], fdst, fsig, device=r)
                    b1 = self.c_compute(b0, T1[r], device=r)
                    self.c_recv(Rb[r], bdst, bsig, device=r)
                    self.c_compute(b1, T2[r], device=r)
            return Echo, Rf, Rb

    return SameDirWait



def build_fusedcomm():
    """Control C: identical wait topology to `both` -- BOTH ranks wait, one ring each
    direction, 4 buffers -- but each rank's send and wait live in ONE kernel/dispatch
    instead of two.

    This is the shape that upstream programs actually use and that passes:
    `allscan_middle_step` does wait-then-send in a single InCore kernel, and
    `tests/st/distributed/test_l3_ep_dispatch_combine.py` puts every notify and wait of a
    rank inside one `chip_orch` dispatch.

      completes -> the trigger is the CROSS-DISPATCH split (a rank's wait sitting in a later
                   dispatch than its send), not the cyclic wait as such
      deadlocks -> the cyclic wait itself is fatal regardless of dispatch structure
    """

    @pl.program
    class FusedCommWait:
        @pl.function(type=pl.FunctionType.InCore)
        def send_then_wait(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            OutB: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            fdst: pld.DistributedTensor[[D, D], pl.FP32],
            fsig: pld.DistributedTensor[[S, 1], pl.INT32],
            bdst: pld.DistributedTensor[[D, D], pl.FP32],
            bsig: pld.DistributedTensor[[S, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tuple[pl.Tensor[[D, D], pl.FP32], pl.Tensor[[D, D], pl.FP32]]:
            t = pl.load(X, [0, 0], [D, D])
            Echo = pl.store(t, [0, 0], Echo)
            pld.tile.remote_store(t, target=fdst, peer=peer, offsets=[0, 0])
            pld.system.notify(target=fsig, peer=peer, offsets=[0, 0], value=1,
                              op=pld.NotifyOp.AtomicAdd)
            pld.system.wait(signal=bsig, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            OutB = pl.store(pl.load(bdst, [0, 0], [D, D]), [0, 0], OutB)
            return Echo, OutB

        @pl.function(type=pl.FunctionType.InCore)
        def wait_then_send(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            OutF: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            fdst: pld.DistributedTensor[[D, D], pl.FP32],
            fsig: pld.DistributedTensor[[S, 1], pl.INT32],
            bdst: pld.DistributedTensor[[D, D], pl.FP32],
            bsig: pld.DistributedTensor[[S, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tuple[pl.Tensor[[D, D], pl.FP32], pl.Tensor[[D, D], pl.FP32]]:
            pld.system.wait(signal=fsig, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
            OutF = pl.store(pl.load(fdst, [0, 0], [D, D]), [0, 0], OutF)
            t = pl.load(X, [0, 0], [D, D])
            Echo = pl.store(t, [0, 0], Echo)
            pld.tile.remote_store(t, target=bdst, peer=peer, offsets=[0, 0])
            pld.system.notify(target=bsig, peer=peer, offsets=[0, 0], value=1,
                              op=pld.NotifyOp.AtomicAdd)
            return Echo, OutF

        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            t = pl.load(X, [0, 0], [D, D])
            return pl.store(pl.add(t, t), [0, 0], Out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def c_compute(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Out: pl.Out[pl.Tensor[[D, D], pl.FP32]],
        ) -> pl.Tensor[[D, D], pl.FP32]:
            return self.compute(X, Out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def c_sw(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            OutB: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            fdst: pld.DistributedTensor[[D, D], pl.FP32],
            fsig: pld.DistributedTensor[[S, 1], pl.INT32],
            bdst: pld.DistributedTensor[[D, D], pl.FP32],
            bsig: pld.DistributedTensor[[S, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tuple[pl.Tensor[[D, D], pl.FP32], pl.Tensor[[D, D], pl.FP32]]:
            return self.send_then_wait(X, Echo, OutB, fdst, fsig, bdst, bsig, peer)

        @pl.function(type=pl.FunctionType.Orchestration)
        def c_ws(
            self,
            X: pl.Tensor[[D, D], pl.FP32],
            Echo: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            OutF: pl.Out[pl.Tensor[[D, D], pl.FP32]],
            fdst: pld.DistributedTensor[[D, D], pl.FP32],
            fsig: pld.DistributedTensor[[S, 1], pl.INT32],
            bdst: pld.DistributedTensor[[D, D], pl.FP32],
            bsig: pld.DistributedTensor[[S, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tuple[pl.Tensor[[D, D], pl.FP32], pl.Tensor[[D, D], pl.FP32]]:
            return self.wait_then_send(X, Echo, OutF, fdst, fsig, bdst, bsig, peer)

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

            for r in pl.range(P):
                fdst = pld.window(fdst_buf, [D, D], dtype=pl.FP32)
                fsig = pld.window(fsig_buf, [S, 1], dtype=pl.INT32)
                bdst = pld.window(bdst_buf, [D, D], dtype=pl.FP32)
                bsig = pld.window(bsig_buf, [S, 1], dtype=pl.INT32)

                if r == 0:
                    a0 = self.c_compute(X[r], T0[r], device=r)
                    self.c_sw(a0, Echo[r], Rb[r], fdst, fsig, bdst, bsig, r + 1, device=r)
                    self.c_compute(a0, T1[r], device=r)
                else:
                    b0 = self.c_compute(X[r], T0[r], device=r)
                    self.c_ws(b0, Echo[r], Rf[r], fdst, fsig, bdst, bsig, r - 1, device=r)
                    self.c_compute(b0, T1[r], device=r)
            return Echo, Rf, Rb

    return FusedCommWait


#                fwd, rev, swap_r1
_VARIANTS = {
    "fwd": (1, 0, 0),
    "rev": (0, 1, 0),
    "both": (1, 1, 0),
    "sendfirst": (1, 1, 1),
    "fwd5": (1, 0, 0),      # one ring, padded to 5 dispatches -- isolates dispatch count
    "samedir": (1, 1, 0),   # two rings, SAME direction -- isolates "both ranks wait"
    "fusedcomm": (1, 1, 0), # both ranks wait, but each rank's send+wait in ONE kernel
    # Mechanism test: `both`/`sendfirst` + a RAW edge send -> recv on the same rank, so the
    # recv is no longer fan-in-free and cannot be routed ahead of its rank's send.
    "bothdep": (1, 1, 0),
    "sendfirstdep": (1, 1, 1),
}


def select_program(variant: str):
    """Single source of truth for variant -> program.

    Kept here rather than in each caller: `b4_dump_variants.py` once carried its own copy,
    fell through to the wrong builder when new arms were added, and produced a dump of
    `sendfirst` labelled `sendfirstdep` -- which reads exactly like the compiler dropping the
    dependency argument. Anything that needs a variant's program calls this.
    """
    do_fwd, do_rev, swap_r1 = _VARIANTS[variant]
    if variant in ("bothdep", "sendfirstdep"):
        return build_dep(swap_r1)
    if variant == "fusedcomm":
        return build_fusedcomm()
    if variant == "samedir":
        return build_samedir()
    if variant == "fwd5":
        return build_one(0, pad=1)
    if do_fwd and do_rev:
        return build(do_fwd, do_rev, swap_r1)
    return build_one(1 if do_rev else 0)


def main() -> int:
    variant = sys.argv[1]
    if variant not in _VARIANTS:
        print(f"unknown variant {variant!r}; pick one of {sorted(_VARIANTS)}")
        return 2
    do_fwd, do_rev, swap_r1 = _VARIANTS[variant]
    devices = [int(x) for x in sys.argv[2].split(",")][:P]
    platform = sys.argv[3] if len(sys.argv) > 3 else "a2a3"

    torch.manual_seed(5)
    X = torch.randn(P, D, D).share_memory_()
    Echo = torch.zeros(P, D, D).share_memory_()
    Rf = torch.zeros(P, D, D).share_memory_()
    Rb = torch.zeros(P, D, D).share_memory_()

    both = do_fwd and do_rev
    print(f"--- two-ring WAIT probe, variant {variant} "
          f"(fwd={do_fwd} rev={do_rev} swap_r1={swap_r1}, {'4' if both else '2'} window buffers) on {devices}",
          flush=True)
    program = select_program(variant)
    compiled = ir.compile(program, platform=platform,
                          distributed_config=DistributedConfig(device_ids=devices,
                                                               num_sub_workers=0))
    rt = compiled.prepare()
    try:
        t0 = time.time()
        rt(X, Echo, Rf, Rb)
        dt = time.time() - t0
        # The second transfer's destination slot and expected value differ per variant, so
        # they are spelled out rather than derived. Getting this wrong once already made
        # three passing arms print `rev_payload_ok=False`, which reads as a delivery failure
        # when it was only the check looking at the wrong tensor.
        #
        #   variant     second transfer            lands in   expected
        #   rev/both    rank1 sends b1 -> rank0    Rb[0]      X[1]*4
        #   bothdep     same as both (the dep edge is payload-neutral)
        #   sendfirst   rank1 sends b0 -> rank0    Rb[0]      X[1]*2
        #   sendfirstdep same as sendfirst
        #   fusedcomm   rank1 sends b0 -> rank0    Rb[0]      X[1]*2
        #   samedir     rank0 sends a1 -> rank1    Rb[1]      X[0]*4   (both rings 0->1)
        #   fwd/fwd5    (no second transfer)       -           -
        ok_f = (Rf[1] - X[0] * 2.0).abs().max().item() < 1e-6 if do_fwd else None
        if variant in ("fwd", "fwd5"):
            ok_b = None
        elif variant == "samedir":
            ok_b = (Rb[1] - X[0] * 4.0).abs().max().item() < 1e-6
        elif variant in ("sendfirst", "sendfirstdep", "fusedcomm"):
            ok_b = (Rb[0] - X[1] * 2.0).abs().max().item() < 1e-6
        else:  # rev, both, bothdep
            ok_b = (Rb[0] - X[1] * 4.0).abs().max().item() < 1e-6
        print(f"{variant.upper()}: COMPLETED in {dt:.1f}s  fwd_payload_ok={ok_f} "
              f"rev_payload_ok={ok_b}", flush=True)
    finally:
        rt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
