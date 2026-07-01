#!/usr/bin/env python3
"""Direct PTO-runtime ("simpler") AllScan — benchmark/test adapter + CLI.

This is a hand-written PTO-runtime port of the PyPTO DSL program: one uniform
AIV kernel runs on every rank and selects its behaviour from ``rankId``:

    rank 0       : source — emit S_local, push block to rank 1, no wait.
    rank 1..P-2  : receive from prev, fuse (S_local + gamma (*) recv), push to next.
    rank P-1     : receive from prev, fuse, no push (chain terminates).

Each rank forwards its computed block straight into the next rank's recv slot in
the shared HCCL window (remote TSTORE) and signals it (TNOTIFY); the receiver
TWAITs before reading. Work is pipelined over K blocks of dk/K rows.

The kernels live in ``kernels/aiv/allscan_kernel.cpp`` and
``kernels/orchestration/allscan_orch.cpp``.

Run standalone::

    python implementations/simpler/impl.py -p a2a3sim -d 0-1
    python implementations/simpler/impl.py -p a2a3sim -d 0-3 --dk 128 --dv 128 --K 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import AllscanImpl, expected_allscan, make_inputs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(HERE, "kernels")
RUNTIME = "tensormap_and_ringbuffer"
DTYPE_NBYTES = 4  # float32


def build_chip_callable(platform: str, pto_isa_commit: str | None = None):
    """Compile the AIV allscan kernel + its C++ orchestration shim."""
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable
    from simpler_setup.elf_parser import extract_text_section
    from simpler_setup.kernel_compiler import KernelCompiler
    from simpler_setup.pto_isa import ensure_pto_isa_root

    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(commit=pto_isa_commit, clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)
    # src/common — for platform_comm/comm_context.h
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]

    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(KERNELS_DIR, "aiv/allscan_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=os.path.join(KERNELS_DIR, "orchestration/allscan_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        # S_local, gamma (IN), output (OUT), scratch (INOUT)
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="allscan_orchestration",
        config_name="allscan_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


class SimplerAllscan(AllscanImpl):
    """Direct PTO-runtime AllScan driven by a persistent L3 Worker.

    ``build`` compiles the kernels and stands up the Worker once; ``run`` copies
    the caller's inputs into the per-rank shared-memory tensors, executes the
    multi-chip DAG, and copies the results back. The comm domain is allocated
    inside the run (the window is zeroed at allocation), so the per-block signals
    are correct with a fixed epoch of 1.

    The runtime releases the comm domain at the end of every ``worker.run()``, so
    a plain ``run`` per iteration pays a full domain alloc/free + drain round-trip
    each time (the dominant per-call cost). :meth:`measure` therefore dispatches a
    whole batch of AllScans inside ONE ``worker.run()`` under a single domain,
    each iteration writing to a disjoint window slot, so that fixed overhead is
    paid once per batch and the reported timing reflects marginal kernel+comm cost.
    """

    name = "simpler"

    #: Number of AllScans dispatched per batched timing sample in measure().
    _MEASURE_BATCH = 16

    def __init__(self, pto_isa_commit: str | None = None) -> None:
        self.pto_isa_commit = pto_isa_commit
        self.worker = None

    def build(self, dk, dv, K, P, device_ids, platform):
        # Tear down any Worker from a previous config before standing up a new
        # one: build() is called once per benchmark config on a reused impl
        # object, and a forked L3 Worker that isn't closed leaks its chip child
        # processes (they busy-wait forever).
        self.close()

        if dk % K != 0:
            raise ValueError(f"dk ({dk}) must be divisible by K ({K})")
        if dk > 128 or dv > 128:
            raise ValueError(f"kernel tile capacity is 128x128, got dk={dk} dv={dv}")

        from simpler.task_interface import (
            CallConfig,
            CommBufferSpec,
            DataType,
            TaskArgs,
            TensorArgType,
        )
        # ``ContinuousTensor`` was folded into ``Tensor`` in the updated runtime
        # (identical ``make(data, shapes, dtype, child_memory=)`` signature).
        from simpler.task_interface import Tensor as ContinuousTensor
        from simpler.worker import Worker
        from simpler_setup.torch_interop import make_tensor_arg

        self.dk, self.dv, self.K, self.P = dk, dv, K, P
        self.device_ids = device_ids[:P]
        # Stash runtime symbols so both the single-shot and batched orch fns can
        # build args without re-importing.
        self._CallConfig = CallConfig
        self._CommBufferSpec = CommBufferSpec
        self._ContinuousTensor = ContinuousTensor
        self._DataType = DataType
        self._TaskArgs = TaskArgs
        self._TensorArgType = TensorArgType
        self._make_tensor_arg = make_tensor_arg

        # One scratch slot = recv region (dk*dv floats) + K int32 signal slots.
        self._slot_floats = dk * dv + K
        self._slot_nbytes = self._slot_floats * DTYPE_NBYTES

        chip_callable = build_chip_callable(platform, self.pto_isa_commit)
        self.worker = Worker(
            level=3,
            platform=platform,
            runtime=RUNTIME,
            device_ids=self.device_ids,
            num_sub_workers=0,
        )
        self._cid = self.worker.register(chip_callable)
        self.worker.init()

        # Per-rank shared-memory tensors (one private input/output per chip child).
        self.host_s = [torch.zeros((dk, dv), dtype=torch.float32).share_memory_() for _ in range(P)]
        self.host_g = [torch.zeros((dk, 1), dtype=torch.float32).share_memory_() for _ in range(P)]
        self.host_out = [torch.zeros((dk, dv), dtype=torch.float32).share_memory_() for _ in range(P)]

    def _submit_iter(self, orch, handle, cfg, slot_off_floats):
        """Submit one full P-rank AllScan, using the window slot at
        ``slot_off_floats`` (in floats) for every rank's recv+signal region."""
        Args = self._TaskArgs
        TT = self._TensorArgType
        mk = self._make_tensor_arg
        for i in range(self.P):
            domain = handle[i]
            chip_args = Args()
            chip_args.add_tensor(mk(self.host_s[i]), TT.INPUT)
            chip_args.add_tensor(mk(self.host_g[i]), TT.INPUT)
            chip_args.add_tensor(mk(self.host_out[i]), TT.OUTPUT_EXISTING)
            chip_args.add_tensor(
                self._ContinuousTensor.make(
                    data=domain.buffer_ptrs["scratch"] + slot_off_floats * DTYPE_NBYTES,
                    shapes=(self._slot_floats,),
                    dtype=self._DataType.FLOAT32,
                    child_memory=True,
                ),
                TT.INOUT,
            )
            chip_args.add_scalar(self.dk)
            chip_args.add_scalar(self.dv)
            chip_args.add_scalar(self.K)
            chip_args.add_scalar(domain.domain_size)
            chip_args.add_scalar(1)  # epoch — each slot is zeroed once at alloc, so always 1
            chip_args.add_scalar(domain.device_ctx)
            orch.submit_next_level(self._cid, chip_args, cfg, worker=i)

    def _domain(self, orch, name, n_slots):
        nbytes = n_slots * self._slot_nbytes
        window_size = max(((nbytes + 511) // 512) * 512, 4 * 1024)
        return orch.allocate_domain(
            name=name,
            workers=list(range(self.P)),
            window_size=window_size,
            buffers=[self._CommBufferSpec(
                name="scratch", dtype="float32", count=n_slots * self._slot_floats, nbytes=nbytes
            )],
        )

    def run(self, S_locals, gammas, outputs):
        assert self.worker is not None, "call build() first"
        for i in range(self.P):
            self.host_s[i].copy_(S_locals[i])
            self.host_g[i].copy_(gammas[i])
            self.host_out[i].zero_()

        def orch_fn(orch, _args, cfg):
            with self._domain(orch, "allscan", 1) as handle:
                self._submit_iter(orch, handle, cfg, 0)

        self.worker.run(orch_fn, args=None, config=self._CallConfig())
        for i in range(self.P):
            outputs[i].copy_(self.host_out[i])

    def run_batch(self, S_locals, gammas, n_iters: int) -> float:
        """Dispatch ``n_iters`` AllScans inside ONE worker.run() under a single
        comm domain, each iteration writing to a disjoint window slot. Returns
        the total wall time (seconds). This pays the comm-domain alloc/free and
        drain round-trip once for the whole batch instead of once per iteration,
        so ``total / n_iters`` reflects the marginal kernel+comm cost. The slots
        are disjoint, so iterations cannot race on each other's recv/signal.
        """
        assert self.worker is not None, "call build() first"
        for i in range(self.P):
            self.host_s[i].copy_(S_locals[i])
            self.host_g[i].copy_(gammas[i])

        def orch_fn(orch, _args, cfg):
            with self._domain(orch, "allscan_batch", n_iters) as handle:
                for it in range(n_iters):
                    self._submit_iter(orch, handle, cfg, it * self._slot_floats)

        t0 = time.perf_counter()
        self.worker.run(orch_fn, args=None, config=self._CallConfig())
        return time.perf_counter() - t0

    #: simpler amortizes the per-call comm-domain + drain overhead in measure().
    amortized_timing = True

    def measure(self, S_locals, gammas, outputs, n_iters):
        """Per-iteration samples with per-call orchestration overhead amortized.

        Each sample is one batched run of ``_MEASURE_BATCH`` AllScans divided by
        the batch size; ``n_iters`` such samples form the distribution.
        """
        batch = self._MEASURE_BATCH
        return [self.run_batch(S_locals, gammas, batch) / batch * 1e3 for _ in range(n_iters)]

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(x) for x in spec.split(",") if x != ""]
    if not (2 <= len(ids) <= 16):
        raise ValueError(f"allscan needs between 2 and 16 devices, got {len(ids)} ({ids})")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1' or '0-3'.")
    parser.add_argument("--dk", type=int, default=64, help="Key dimension (rows). Default 64.")
    parser.add_argument("--dv", type=int, default=64, help="Value dimension (cols). Default 64.")
    parser.add_argument("--K", type=int, default=1, help="Pipeline depth / number of blocks. Default 1.")
    parser.add_argument("--pto-isa-commit", default=None, help="Optional PTO ISA commit/tag to fetch.")
    cli = parser.parse_args()

    device_ids = _parse_device_range(cli.device)
    P = len(device_ids)
    print(f"[simpler] platform={cli.platform} devices={device_ids} P={P} dk={cli.dk} dv={cli.dv} K={cli.K}")

    S_locals, gammas, outputs = make_inputs(P, cli.dk, cli.dv)
    impl = SimplerAllscan(pto_isa_commit=cli.pto_isa_commit)
    impl.build(cli.dk, cli.dv, cli.K, P, device_ids, cli.platform)
    try:
        impl.run(S_locals, gammas, outputs)
    finally:
        impl.close()

    expected = expected_allscan(S_locals, gammas)
    max_diff = float((outputs - expected).abs().max())
    print(f"[simpler] max |out - expected| = {max_diff:.3e}")
    if max_diff > 1e-3:
        print("[simpler] golden check FAILED")
        return 1
    print("[simpler] all ranks matched golden ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
