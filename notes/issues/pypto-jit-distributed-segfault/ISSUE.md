# [Bug] Segfault when a distributed program (`DistributedWorker.prepare()`) is stood up on a device a `@pl.jit` runtime already used in-process

<!-- Draft for GitHub issue (template: bug_report.yml). Fill dropdowns as noted. -->

**Component:** Backend (runtime / DistributedWorker)
**NPU Kind:** Ascend 910B2 (a2a3)
**Host Platform:** Linux (aarch64)
**Git Commit ID:** `3d25d92f3e6f78873330da54863bfd6821cadec9` (branch `fence-onto-de48ab4`)

## Description

Within a single process, running a ``@pl.jit`` dispatch on device *D* and then standing
up a distributed program on device *D* (``ir.compile(..., distributed_config=...)`` →
``compiled.prepare()``, which **forks chip workers**) **segfaults** the process. Each
step works fine on its own. The crash is a hard ``Fatal Python error: Segmentation
fault`` — not a Python exception — so it cannot be caught or recovered.

This blocks composing a single-device ``@pl.jit`` compute phase with a distributed
collective phase on the same devices in one process (e.g. a sequence-parallel operator
that computes locally with ``@pl.jit``, then exchanges boundary state with a distributed
collective, then finishes locally).

Expected: the two runtimes either coexist, or ``prepare()`` raises a clean, catchable
error (e.g. "device *D* already has an active runtime"). A segfault is never acceptable.

## Steps to Reproduce

`repro_jit_then_distributed.py` (attached). The distributed program used is this
project's PyPTO AllScan collective as a ready-made minimal distributed program — the bug
is not AllScan-specific; any ``DistributedWorker.prepare()`` on a device a ``@pl.jit``
runtime already touched reproduces it.

```bash
cd pto-zeco
export PYTHONPATH=$PWD LD_PRELOAD=${CANN}/aarch64-linux/lib64/libhccl.so

# control — distributed program alone
python repro_jit_then_distributed.py --mode allscan_only     --devices 6,7   # -> PASS

# bug — trivial @pl.jit first, then the same distributed program on the same devices
python repro_jit_then_distributed.py --mode jit_then_allscan --devices 6,7   # -> SEGFAULT
```

The `@pl.jit` kernel is a trivial ``c = a + 1`` at ``pl.Level.CORE_GROUP``; the crash is
independent of what the jit kernel computes.

## Expected Behavior

`--mode jit_then_allscan` should print ``OK`` (both phases run), or ``prepare()`` should
raise a catchable exception. No segfault.

> Note: run each mode on a freshly-cleaned device environment (delete stale
> `/tmp/barrier_pto_multi_comm_*`, no stray `simpler`/`chip_process` workers). The
> canonical AllScan pytest (`allscan/tests/test_pypto.py --platform a2a3`) passing is a
> good health check that `--mode allscan_only` will pass. The primary evidence below is
> the real traceback from the ZeCO P=2 backend run, which is deterministic.

## Actual Behavior

`--mode allscan_only`: **PASS** (clean env). `--mode jit_then_allscan`: **segfault** with:

```
Fatal Python error: Segmentation fault

Current thread [...] (most recent call first):
  File ".../simpler/task_interface.py", line 1007 in init
  File ".../simpler/worker.py", line 1124 in _chip_process_loop
  File ".../simpler/worker.py", line 2968 in _start_hierarchical
  File ".../pypto/runtime/distributed_runner.py", line 999 in __init__
  File ".../pypto/ir/distributed_compiled_program.py", line 458 in prepare
  File ".../allscan/implementations/pypto/impl.py", line 90 in build   # PytoAllscan.build -> prepare()
  ...
```

The crash is in the forked chip worker's ``task_interface.init`` (``_chip_process_loop``),
i.e. the DistributedWorker's chip child fails to initialise on a device the in-process
``@pl.jit`` runtime already opened.

## Additional Context

- Discovered building a sequence-parallel GLA operator (ZeCO): per rank it runs a
  ``@pl.jit`` local compute (``stage1``), then a distributed AllScan for the boundary
  state, then another ``@pl.jit`` local compute (``stage2``). ``stage1`` (jit) succeeds;
  the subsequent AllScan ``prepare()`` segfaults. Sequencing the phases (jit fully
  returns before ``prepare()``) does not help — the jit runtime's per-run finalize does
  not reclaim the chip state in-process (``runner.py`` documents subprocess isolation as
  the robust pattern for a related SVM-registration case).
- Likely fix directions: (a) have the jit runtime fully release the device on finalize so
  a later ``prepare()`` can re-init it; or (b) detect an existing runtime on the target
  device in ``prepare()``/chip-worker ``init`` and raise instead of dereferencing into a
  segfault.
- Workarounds: run the jit phases in a separate process (subprocess isolation), or express
  the whole operator as one distributed program (no ``@pl.jit`` — the route we took).
