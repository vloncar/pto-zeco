"""PyPTO DSL ZeCO / GLA forward + backward — benchmark/test adapter.

:class:`PyPtoZeCo` runs the **entire** ZeCO forward as ONE fully-fused distributed
``@pl.program`` (:mod:`.fused_program`), and the entire backward as a second one
(:mod:`.fused_backward_program`, B4). Per rank ``r`` (device ``r``) the single host
orchestrator runs three phases, all as distributed InCore chip kernels — no ``@pl.jit``,
no host round-trip:

  1. **stage1** — local end-of-slice state ``S_total[r]`` (chunk-recurrent scan from ``S=0``).
  2. **AllScan ring** — the exclusive-prefix boundary scan
     ``out[p] = S_local[p] + gamma[p]*out[p-1]``; rank ``r`` receives ``out[r-1]`` (its
     boundary ``S_recv[r]``, zero for rank 0) into a device-local window.
  3. **stage2** — ``O[r]`` = the same chunk recurrence initialised from ``S_recv[r]``.

``gamma`` (device total decay) is ``A.prod`` over tokens, computed host-side.

**Steady-state timing.** A ``DistributedWorker`` prepare/close (fork chip workers + HCCL
comm-domain setup) costs ~8–9 s and dominates a single ``forward``. To make the *operator*
comparison against ``simpler`` fair, :meth:`build` prepares the worker **once** and allocates
the shared-memory IO buffers reused in place; :meth:`forward`/:meth:`measure` then only copy
inputs and dispatch (mirrors :class:`allscan.implementations.pypto.impl.PyPtoAllscan`). The
one-time prepare is therefore paid at build, and :attr:`amortized_timing` is True.

``P == 1`` is a native path (single rank, no boundary, ``S_recv = 0``) built by a separate
factory (:func:`gla.implementations.pypto.fused_program._build_p1_forward_program`). It
usually still compiles to a distributed program (has ``prepare``); should a config ever
compile it non-distributed, :meth:`forward` falls back to the per-call
:func:`run_fused_forward` path and :meth:`measure` to the default per-call timing.

**One prepared program at a time.** :meth:`build` compiles both directions and allocates
every IO buffer, then prepares the forward; :meth:`backward` closes that worker and prepares
the backward's (and vice versa), so exactly one ``DistributedWorker`` — and exactly one HCCL
comm domain — exists at any moment. Two arrangements that look more natural were tried on
hardware at P=2 and **both fail**, in ways P=1 completely hides (a single rank needs no peer
comm, so every P=1 case passed while every P>1 case failed):

* two live workers — the second one's comm domain never comes up:
  ``_ensure_comm_base failed on 2/2 chips ... control_comm_init failed ... comm_init failed``;
* one worker hosting both programs via ``prepare(extra_compiled=[...])``, with or without
  ``persistent=True`` — prepare and the forward dispatch both succeed, then the first backward
  dispatch dies on device with ``ACL_ERROR_RT_AICPU_EXCEPTION`` /
  ``sched_error_code=100 SCHEDULER_TIMEOUT, sub_class=S1:running-stalled``, i.e. a ring wait
  that is never satisfied. Both generated orchestrators declare a comm domain named
  ``comm_d0`` (codegen numbers them per program from 0), so the two programs' windows are
  strong candidates for aliasing — but ``persistent=True``, which namespaces domains per
  program, does not rescue it, so the mechanism is not fully established.

The cost is one prepare/close (~8-9 s) per *direction switch*, not per call: a run that does
forward-only or backward-only work pays it once, and repeated calls in one direction stay
steady-state. Alternating forward/backward per step would pay it every step — worth fixing
before B5.4 quotes a fused training-step latency.

Runs on both a2a3sim and a2a3 hardware. Every distributed / HCCL run must set
``LD_PRELOAD=<cann>/lib64/libhccl.so`` or the rootinfo handshake hangs.
"""

from __future__ import annotations

import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gla.common import ZeCoImpl  # noqa: E402
from gla.implementations.pypto.fused_backward_program import (  # noqa: E402
    build_fused_backward_program,
)
from gla.implementations.pypto.fused_program import (  # noqa: E402
    build_fused_forward_program,
    compile_fused_forward,
    run_fused_forward,
)


class PyPtoZeCo(ZeCoImpl):
    """PyPTO ZeCO forward + backward: each as one fully-fused distributed program,
    dispatched on a reusable ``DistributedWorker`` (prepare once / dispatch many)."""

    name = "pypto"

    #: build() prepares the worker once, so measure() times only steady-state dispatch.
    amortized_timing = True

    #: The backward is amortized on the same terms: the first ``backward`` swaps the live
    #: worker to the backward program (paying one prepare, and one close of the forward's),
    #: and ``measure_backward`` warms that swap before sampling. So repeated backward calls
    #: time only the dispatch. Note it is the *direction switch* that is expensive here, not
    #: the direction itself — alternating forward/backward per iteration would pay it every
    #: time, which is why the benchmark measures one direction at a time.
    amortized_timing_backward = True

    def __init__(self) -> None:
        self._compiled = None
        self._bcompiled = None
        self._rt = None
        self._rt_backward = False

    def build(self, P, L, C, dk, dv, device_ids, platform):
        """Compile the fully-fused ZeCO forward and prepare the reusable worker.

        Args as in :meth:`gla.common.ZeCoImpl.build`. ``C`` is the real chunk size
        (``L`` divisible by ``C``; ``N = L // C``). Compiles once, then — for the
        distributed path — stands up the ``DistributedWorker`` and allocates the
        shared-memory IO buffers (reused in place by every ``forward``/``measure``).
        ``K=1`` is the ring pipeline depth.
        """
        assert L % C == 0, f"L={L} not divisible by C={C}"
        # NOTE: `min(dk, dv) >= C` was enforced here until 2026-08-13. A head dim below C
        # was silently corrupt on a2a3 — a pto-isa defect, not this operator: a cross-core
        # TPipe strided the consumer's local ring by the popped tile's own size instead of the
        # ring's SLOT_SIZE, so two differently-sized tiles held at once aliased in L1. For
        # [M,K] @ [K,N] the operands overlap iff N < M, which both head dims can reach
        # (dv via the [C, dv] output tiles, dk via `b = tril[C,C] @ la[C,dk]`).
        # Fixed upstream as pto-isa MR !1457 / issue #521, MERGED. Our pto-isa pin (83d01313,
        # set by runtime/pto_isa.pin in the simpler submodule) predates the merge, so the fix
        # is carried locally — see allscan/issues/pto-isa-fifo-local-slot-alias/
        # carried-on-pin-83d01313.patch. If that patch is ever lost, dk < C or dv < C silently
        # returns wrong results again; dv < C corrupts every dispatch, dk < C about 1 in 20.
        self.P, self.L, self.C, self.dk, self.dv = P, L, C, dk, dv
        self.platform = platform
        self.device_ids = list(device_ids[:P])

        # Release any worker from a previous config first (it holds forked chip
        # children that leak if not closed).
        self.close()

        # Host-side constant tiles for the chunk kernels: the within-chunk lower-triangular
        # ones matrix (it drives the cumprod matmul *and* is the causal mask — same values),
        # and a zero S-init / rank-0 boundary. Shared so they can be passed to the reusable
        # worker in place. (The two all-ones gamma-broadcast matrices are gone: the kernels
        # now reduce with col_sum + row_expand_mul — see fused_program's F3.1 note.)
        self._tril = torch.tril(torch.ones(C, C, dtype=torch.float32)).share_memory_()
        self._zero = torch.zeros(dk, dv, dtype=torch.float32).share_memory_()
        # Backward-only constants, cheap enough to always allocate: the upper-triangular
        # ones matrix (the per-chunk REVERSE cumulative sum of the gate gradient is
        # `triu @ dg_cs`), and the [dk,1] zero / one vectors that seed the decay carries.
        self._triu = torch.triu(torch.ones(C, C, dtype=torch.float32)).share_memory_()
        self._zerov = torch.zeros(dk, 1, dtype=torch.float32).share_memory_()
        self._onev = torch.ones(dk, 1, dtype=torch.float32).share_memory_()

        from pypto import ir
        from pypto.ir.distributed_compiled_program import DistributedConfig

        dist_cfg = DistributedConfig(device_ids=self.device_ids, num_sub_workers=0)
        # The chunk kernels block the head dim, and stage2 rebuilds each chunk's state from
        # stage1's per-chunk snapshot rather than carrying it. How many blocks, and how deep
        # the cube<->vector ring may be, depends on the shape. `compile_fused_forward` tries
        # the cheapest settings first and keeps the first that fits the vector buffer, so a
        # shape that already worked keeps its old choice and a bigger head dim gets a finer
        # one instead of failing outright.
        self._compiled, self.blocking = compile_fused_forward(
            L, C, dk, dv, 1, P, platform=platform, distributed_config=dist_cfg)
        # The BACKWARD is compiled lazily, on first use (see _ensure_backward_compiled).
        # It used to be compiled here, unconditionally, which coupled the two directions in
        # two unwanted ways: a forward-only run paid the backward's compile time, and — worse
        # — the backward's tighter shape ceiling *capped the forward*. `grad_o` is the widest
        # kernel in either direction, so at C=32/dk=dv=64 the backward overflows the vector
        # buffer by ~1 KB (189440 vs 188416) and `ir.compile` raises. With an eager compile
        # that killed the whole config, so a shape the forward reaches comfortably could not
        # be benchmarked or run at all. Only the *buffers* below must be eager (they have to
        # exist before prepare() forks the chip children); compiling does not.
        self._bcompiled = None
        self._bcompile_args = (L, C, dk, dv, P, platform, dist_cfg)

        # Prepare-once: for the distributed path, allocate the shared IO buffers BEFORE
        # prepare() forks the chip workers, and reuse them in place. (P=1 that compiles
        # non-distributed has no prepare -> forward()/measure() fall back to per-call.)
        self._rt = None
        if hasattr(self._compiled, "prepare"):
            self._h_Q = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._h_K = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._h_V = torch.zeros((P, L, dv), dtype=torch.float32).share_memory_()
            self._h_A = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._h_g = torch.zeros((P, dk, 1), dtype=torch.float32).share_memory_()
            self._h_O = torch.zeros((P, L, dv), dtype=torch.float32).share_memory_()
            self._b_dO = torch.zeros((P, L, dv), dtype=torch.float32).share_memory_()
            self._b_dQ = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._b_dK = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            self._b_dV = torch.zeros((P, L, dv), dtype=torch.float32).share_memory_()
            self._b_dA = torch.zeros((P, L, dk), dtype=torch.float32).share_memory_()
            # ONE worker hosting BOTH programs. Preparing the backward on a SECOND
            # DistributedWorker does not work: the two workers each try to stand up their
            # own HCCL comm domain over the same devices and the second one dies with
            # `_ensure_comm_base failed ... control_comm_init failed ... comm_init failed`.
            # (P=1 survives it — a single rank needs no peer comm — which is exactly why
            # the P=1 tests passed while every P>1 case failed.) `extra_compiled` is the
            # supported way to share one worker lifecycle across HOST programs; the
            # forward and backward then also share one comm domain. Both directions' IO
            # buffers must exist before this call, since prepare() forks the chip workers
            # and shared memory created afterwards is invisible to them.
            # Every buffer for BOTH directions is allocated above, before any prepare():
            # prepare() forks the chip workers and shared memory created afterwards is
            # invisible to them (errno 107017, INVALID_HANDLE). Only one program is
            # prepared at a time — see the module docstring for the two multi-program
            # arrangements that fail on hardware.
            self._rt_backward = False
            self._rt = self._compiled.prepare()

    def _stage_inputs(self, Q, K, V, A):
        """Copy the per-forward inputs (and host-side gamma) into the shared buffers."""
        gammas = A.prod(dim=1).reshape(self.P, self.dk, 1)
        self._h_Q.copy_(Q)
        self._h_K.copy_(K)
        self._h_V.copy_(V)
        self._h_A.copy_(A)
        self._h_g.copy_(gammas)

    def _select(self, backward: bool):
        """Make the worker for the requested direction the live one.

        A no-op when it already is, so repeated same-direction calls stay steady-state.
        Switching closes the current worker first: the two programs cannot be prepared at
        the same time (module docstring).
        """
        if self._rt is not None and self._rt_backward == backward:
            return
        if self._rt is not None:
            self._rt.close()
            self._rt = None
        if backward:
            self._ensure_backward_compiled()
        self._rt = (self._bcompiled if backward else self._compiled).prepare()
        self._rt_backward = backward

    def _ensure_backward_compiled(self):
        """Compile the fused backward on first use; a no-op afterwards.

        Kept out of :meth:`build` so a forward-only user neither pays its compile time nor
        inherits its narrower shape ceiling — see the note in :meth:`build`.
        """
        if self._bcompiled is not None:
            return
        from pypto import ir
        L, C, dk, dv, P, platform, dist_cfg = self._bcompile_args
        self._bcompiled = ir.compile(build_fused_backward_program(L, C, dk, dv, 1, P),
                                     platform=platform, distributed_config=dist_cfg)

    def _dispatch(self):
        """Run one fused-forward dispatch on the prepared worker (inputs already staged)."""
        self._select(backward=False)
        self._h_O.zero_()
        self._rt(self._h_Q, self._h_K, self._h_V, self._h_A, self._h_g,
                 self._tril, self._zero, self._h_O)

    def forward(self, Q, K, V, A):
        """ZeCO forward; args/return as in :meth:`gla.common.ZeCoImpl.forward`."""
        assert self._compiled is not None, "call build() first"
        if self._rt is not None:
            self._stage_inputs(Q, K, V, A)
            self._dispatch()
            return self._h_O.clone()
        # P=1 non-distributed fallback (rare): per-call compiled run.
        gammas = A.prod(dim=1).reshape(self.P, self.dk, 1)
        O = torch.zeros((self.P, self.L, self.dv), dtype=torch.float32)
        run_fused_forward(
            self._compiled, Q, K, V, A, gammas, self._tril, self._zero, O,
            platform=self.platform, device_ids=self.device_ids,
        )
        return O

    # ------------------------------------------------------------------
    # Backward (B4): the whole SP backward as a second fused program.
    # ------------------------------------------------------------------

    def backward(self, Q, K, V, A, dO):
        """ZeCO backward; args/return as in :meth:`gla.common.ZeCoImpl.backward`.

        Stateless by contract: the fused program recomputes the forward chunk scan on
        device (phase 1) rather than depending on a prior :meth:`forward` call, so no
        activations are held between the two.

        Shares the forward's staged input buffers — the backward reads the same
        ``Q,K,V,A,gammas`` — and adds only ``dO`` and the four gradient outputs.
        """
        assert self._compiled is not None, "call build() first"
        if self._rt is None:
            # The forward's rare non-distributed P=1 fallback has a single-kernel entry to
            # call directly; the backward is three chained chip dispatches, so there is no
            # equivalent and nothing to fall back to.
            raise NotImplementedError(
                "fused backward needs the distributed (prepare) path; this config compiled "
                f"the forward without one (P={self.P})")
        self._stage_inputs(Q, K, V, A)
        self._b_dO.copy_(dO)
        for t in (self._b_dQ, self._b_dK, self._b_dV, self._b_dA):
            t.zero_()
        self._select(backward=True)
        self._rt(self._h_Q, self._h_K, self._h_V, self._h_A, self._b_dO, self._h_g,
                 self._tril, self._triu, self._zero, self._zerov, self._onev,
                 self._b_dQ, self._b_dK, self._b_dV, self._b_dA)
        return (self._b_dQ.clone(), self._b_dK.clone(),
                self._b_dV.clone(), self._b_dA.clone())

    def measure_backward(self, Q, K, V, A, dO, n_iters):
        """Steady-state per-backward latency (build/prepare already paid, as in :meth:`measure`)."""
        self.backward(Q, K, V, A, dO)          # warm: first-dispatch costs outside the timing
        samples: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.backward(Q, K, V, A, dO)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def measure(self, Q, K, V, A, n_iters):
        """Steady-state per-forward latency: build/prepare is already paid, so time the
        repeated :meth:`forward` on the reusable worker. Falls back to the default
        per-call timing for a non-distributed P=1 config.

        Times the WHOLE call — host-side gamma, input staging, dispatch and the output
        copy — not the dispatch alone. It used to stage once *outside* the loop and time
        :meth:`_dispatch` by itself, while :meth:`measure_backward` timed a full
        ``backward()`` with staging *inside*. One backend, two stopwatch rules: the
        forward came out optimistic against its own backward (inflating the
        backward/forward ratio) and against simpler, which times a full call in both
        directions. Work-placement parity is ROADMAP F6.6; this is the measurement half
        of it, and it is a precondition for comparing the two backends at all.
        """
        if self._rt is None:
            return super().measure(Q, K, V, A, n_iters)
        self.forward(Q, K, V, A)          # warm: first-dispatch costs outside the timing
        samples: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.forward(Q, K, V, A)
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def close(self):
        """Release whichever direction's worker is live (it holds forked chip children
        that leak if not closed)."""
        if self._rt is not None:
            self._rt.close()
            self._rt = None
