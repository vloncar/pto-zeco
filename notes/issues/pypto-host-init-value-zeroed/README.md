# `pl.create_tensor(init_value=<non-zero>)` is silently ignored in a HOST orchestrator

**Status:** CLOSED — **resolved by removing the kwarg**, not by fixing it. Issue
<https://github.com/hw-native-sys/pypto/issues/2505> closed 2026-08-27 as resolved by upstream
#2530; our PR <https://github.com/hw-native-sys/pypto/pull/2506> closed unmerged with it.
**Found:** 2026-08-21, while landing the snapshot/no-carry forward (A1).
**Diagnosed + fixed + filed:** 2026-08-24. **Superseded:** 2026-08-27.

## Outcome: `init_value` is gone, and that changes what we must do

Upstream #2530 bumped the runtime, which pulled in simpler #1975 removing
`TensorCreateInfo::set_initial_value` from all four runtime variants. That call was exactly what
the chip-level emitter lowered `init_value` to — so the *working* half of the asymmetry lost its
implementation. Rather than leave a kwarg that does nothing anywhere, upstream dropped
`init_value` from the `tensor.create` schema entirely. Both levels now raise:

```
create_tensor: init_value is no longer supported (got 0)...
```

The removal is at schema level (`.set_attr<double>("init_value")` is gone), so the parser, the
`.pto` deserializer and C++-built IR are all gated, not just the Python entry point.

The upstream-sanctioned replacement: **seed the buffer with a kernel that writes it, then order
every reader after that kernel with an explicit dependency** (`pl.submit(..., deps=[seed_tid])` /
`pl.at(..., deps=[seed_tid])`) — the pattern simpler #1922 established for the DeepSeek-V4 decode
buffers. `pl.full` is unaffected and remains the supported way to get a filled tensor.

### ACTION REQUIRED when the pypto pin moves

Our pin (`71020585`) still has the kwarg, so nothing breaks today. But **four live call sites will
raise the moment the pin moves past #2530**:

| file | line | call |
| --- | --- | --- |
| `gla/implementations/pypto/fused_program.py` | 644 | `pl.create_tensor([C, BV], dtype=pl.FP32, init_value=0)` |
| `gla/implementations/pypto/fused_program.py` | 1050 | `pl.create_tensor([C, BV], dtype=pl.FP32, init_value=0)` |
| `gla/implementations/pypto/fused_backward_program.py` | 648 | `pl.create_tensor([C, ZW], dtype=pl.FP32, init_value=0)` |
| `gla/implementations/pypto/fused_backward_program.py` | 1433 | `pl.create_tensor([C, ZW], dtype=pl.FP32, init_value=0)` |

All four are `init_value=0` seeding an accumulator, so they hit forward *and* backward. Note the
zero fill was never the broken case — it worked at both levels — so this is purely an API removal,
not a correctness fix for us.

### What stays true regardless

`DistributedCodegen::EmitTensorCreate` still allocates with a hard-coded `torch.zeros`. That is no
longer a silent contradiction of a documented promise, because the promise is gone — but it is
still the only shape that path can produce. And #2304 (the same emitter, `tensor.assemble` with no
handler at all) is untouched by #2530.

The diagnosis below is kept because the *mechanism* — two emitters for one op, only one
implementing a kwarg — is a live failure mode in this codebase, and because the debugging story is
the reusable part.

## What happens

`pl.create_tensor(shape, dtype, init_value=v)` documents itself as:

> If given, the runtime pre-fills the freshly allocated buffer with this scalar on the AICPU
> (before any kernel writes it). `init_value=0` zeroes the buffer and works for every dtype.
> Non-zero values work for integer and 32/64-bit float dtypes; non-zero fills of fp16/bf16 are
> rejected at codegen.

On a2a3 hardware, a non-zero fill is honoured when the tensor is created inside a **chip
orchestrator** and silently dropped when it is created inside the **host orchestrator**
(`level=pl.Level.HOST, role=pl.Role.Orchestrator`). `init_value=0` is honoured in both.

Measured (`probe.py`, a2a3, one device, tensor copied straight out by an InCore kernel):

| where created | dtype | `init_value` | observed | |
|---|---|---|---|---|
| HOST orch | FP32 | 0 | 0.0 | ok |
| HOST orch | FP32 | 1 | **0.0** | wrong |
| HOST orch | FP32 | 2 | **0.0** | wrong |
| HOST orch | FP32 | 7 | **0.0** | wrong |
| HOST orch | INT32 | 0 | 0 | ok |
| HOST orch | INT32 | 1 | **0** | wrong |
| HOST orch | INT32 | 5 | **0** | wrong |
| chip orch | FP32 | 0 | 0.0 | ok |
| chip orch | FP32 | 1 | 1.0 | ok |

So it is not a dtype restriction and not a value restriction — it is the orchestration level.

## Root cause

`DistributedCodegen::EmitTensorCreate` (`src/codegen/distributed/distributed_codegen.cpp`)
allocated the HOST-level buffer with a hard-coded `torch.zeros` and never looked at the
`init_value` kwarg the op carries:

```cpp
// share_memory_() is required for fork-based distributed runtime visibility.
emitter_.EmitLine("tensors[\"" + target + "\"] = torch.zeros(" + shape +
                  ", dtype=torch." + torch_dtype + ").share_memory_()");
```

The chip path is a *different* emitter — `REGISTER_ORCHESTRATION_OP(tensor_create)` in
`src/codegen/tensor_op_codegen.cpp` — and that one does read `init_value`, emitting
`TensorCreateInfo::set_initial_value(...)` for the AICPU to apply. Hence the split by
orchestration level: two emitters for one op, only one of which implemented the kwarg.

This is a pure codegen drop, visible without any hardware:

```python
program = passes.convert_to_ssa()(Input)          # host_orch creates a [64] FP32 buffer
code = codegen.DistributedCodegen().generate(program)
# init_value=None -> tensors["buf__ssa_v0"] = torch.zeros((64,), dtype=torch.float32)...
# init_value=0    -> tensors["buf__ssa_v0"] = torch.zeros((64,), dtype=torch.float32)...
# init_value=1    -> tensors["buf__ssa_v0"] = torch.zeros((64,), dtype=torch.float32)...   <-- dropped
# init_value=2.5  -> tensors["buf__ssa_v0"] = torch.zeros((64,), dtype=torch.float32)...   <-- dropped
# init_value=7    -> tensors["buf__ssa_v0"] = torch.zeros((64,), dtype=torch.int32)...     <-- dropped
# init_value=1 (BF16) -> torch.zeros((64,), dtype=torch.bfloat16)...  <-- dropped, AND silently
#                                                                     accepted where the chip
#                                                                     path hard-rejects it
```

The `pl.create_tensor` docstring's fp16/bf16 rejection never fired at HOST level either,
because nothing on that path inspected the kwarg at all.

## Fix

`EmitTensorCreate` now honours the kwarg: `torch.zeros` for absent/zero (so the common
path emits exactly what it did before, byte for byte), `torch.full(shape, <literal>,
dtype=...)` for a non-zero fill. A new `PythonFillLiteral` helper renders the value with
the same rules the chip path enforces — finite, whole-number for integer dtypes,
within +/-2^53 — so a call accepted at one orchestration level is accepted at the other.
Float literals use `max_digits10` precision so `2.5` does not become `2.500000`.

One deliberate asymmetry remains, now documented rather than silent: the chip path
rejects non-zero **fp16/bf16** fills because the orchestration translation unit has no
`half`/`bfloat16` type to pack them. At HOST level the buffer is a torch tensor, which has
no such limit, so the fill works there for every dtype. Refusing it would have been
gratuitous, but it is worth knowing when moving a `create_tensor` between levels.

Because the fill *is* the allocation at HOST level, it follows the allocation through the
pre-fork hoist into `_alloc_intermediates` for free — one emitter serves both.

Touched:

* `src/codegen/distributed/distributed_codegen.cpp` — the fix + `PythonFillLiteral`.
* `tests/ut/codegen/test_distributed_codegen.py` — 5 tests (non-zero fp32 fill, integer
  dtype literal, `init_value=0` still `torch.zeros`, fractional-into-integer rejected,
  fill survives alloc hoisting).
* `python/pypto/{language,ir}/op/tensor_ops.py` — docstrings, which claimed a uniform
  AICPU pre-fill that only one of the two levels performed.

Patch against upstream `71020585` (local B4 comm-ordering carries stripped):
`pypto-host-init-value.patch`, alongside this README.

## Verification

| | before | after |
|---|---|---|
| `probe.py` on a2a3, one device | 5 of 9 cases WRONG | **9 of 9 OK** |
| `tests/ut/codegen/` (857 tests) | passed | **passed** |

Post-fix probe output:

```
platform=a2a3
  HOST create_tensor FP32 init_value=0: got [0.0, 0.0] OK
  HOST create_tensor FP32 init_value=1: got [1.0, 1.0] OK
  HOST create_tensor FP32 init_value=2: got [2.0, 2.0] OK
  HOST create_tensor FP32 init_value=7: got [7.0, 7.0] OK
  HOST create_tensor INT32 init_value=0: got [0, 0] OK
  HOST create_tensor INT32 init_value=1: got [1, 1] OK
  HOST create_tensor INT32 init_value=5: got [5, 5] OK
  CHIP create_tensor FP32 init_value=0: got [0.0, 0.0] OK
  CHIP create_tensor FP32 init_value=1: got [1.0, 1.0] OK
```

### Rebased onto upstream main (`fef2831c`)

The fix is prepared in a **separate pypto instance** at `/root/pypto-pr`, branch
`fix/host-orch-create-tensor-init-value`, commit `65849c56` — cloned from upstream, built into
its own tree, and put ahead of the installed pypto by `PYTHONPATH` only. Nothing in
`/opt/pypto` or site-packages is written to, so the other session's environment is untouched.
`check_instance.py` asserts the right pypto is loaded before any measurement counts.

Main had moved **89 commits** since our pin and **all four touched files had changed**, so the
patch was re-derived against the new contents rather than replayed, with an assertion on every
anchor. Two things that a green local suite would not have caught:

* **The B4 comm-ordering fix has landed upstream** (`e2d0f78a`, "keep program order between a
  rank's comm dispatches"). `_alloc_intermediates` now takes `world_size=1` on main. Our new
  unit test asserted the *old* signature — it would have passed while testing a string that no
  longer appears. The local carry in `/opt/pypto` is redundant against current main.
* **The two docstring files had diverged elsewhere.** Copying them wholesale — which is how the
  first patch was built — would have silently reverted unrelated upstream work. They are
  targeted replacements now, and the guard that caught this is in the build script.

`EmitTensorCreate` itself is byte-identical on new main, so the defect is live there and the
fix applies as designed.

### Verification on upstream main

| check | result |
|---|---|
| `tests/ut/codegen/` (upstream main + fix) | **913 passed** |
| pre-commit, all hooks incl. 3 new on main | **passed**, no file rewritten |
| codegen matrix (fp32 / int32 / bf16, fills 0–7) | fills emitted correctly |
| GLA host orchestrators unchanged | **65 files, 445 `torch.zeros`, 0 `torch.full`** |

**Hardware could not be re-run on upstream main, and this is not a property of the fix.** Main's
`simpler` pin needs a compiled `_task_interface` newer than the one installed on this box;
rebuilding it would modify the shared installation. What was verified instead is the step that
actually carries the hardware result: for all 7 probe cases the **allocation line emitted by the
upstream-main build is identical to the line emitted by the build that passed 9/9 on hardware**.
The runtime executes that generated file, so the hardware result transfers. The rest of
`host_orch.py` does differ between the two builds — `make_tensor_arg` gained an `orch._worker`
argument upstream — which is unrelated upstream drift, and is the same drift that stops main
from running against the older installed runtime.

### The 5 GLA forward failures of 2026-08-24 are not this fix

A full forward+backward gate run against the patched build gave `5 failed, 23 passed` on the
forward and a clean `14 passed` on the backward. The forward failures are **not** attributable
to the change, and this was checked rather than assumed:

* The GLA programs pass only `init_value=0`, which the patched emitter routes down the
  **unchanged** `torch.zeros` branch.
* Of the 65 host orchestrators the run generated, **every one contains zero `torch.full(`** —
  so each is byte-identical to what stock upstream would emit. `verify_gla_unchanged.py` in
  `/root/pypto-pr` re-runs that scan over any `build_output/` tree.
* The run itself logged an AICPU exception (507018) and a forced device reset while a co-tenant
  container was compiling on the same box.

Worth re-running alone on a quiet box before drawing any conclusion about those 5 shapes.

### Known limitation, NOT introduced by this fix

In the **persistent** runner path `_alloc_intermediates` runs once at load and the
buffers are reused across dispatches (`base_tensor_frames`), so an intermediate a kernel
writes is not re-filled between runs. That was already true of the `init_value=0` zeroing
and is unchanged here — the fill has exactly the lifetime the zeroing always had. Worth
knowing before relying on `init_value` as a per-dispatch reset.

## Why it is worth reporting

The failure is invisible. `ir.compile` is clean, the device build is clean, the program runs,
and the only symptom is wrong numbers somewhere downstream. In our case a host-created ones
vector seeded a decay accumulator; rank 0 was exact to 6e-05 and rank 1 was off by the entire
boundary term, with the error shrinking chunk by chunk as the (wrongly zero) decay would have
shrunk it anyway. That reads like a distributed/boundary bug, not a memset bug, and it cost
most of a debugging session to localise.

Either the fill should work at host level, or `create_tensor` should reject a non-zero
`init_value` there the way it already rejects non-zero fp16/bf16 fills at codegen.

## Reproducer

`probe.py` in this directory. Run with the standard env:

```
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
python3 probe.py a2a3
```

## What we did instead (workaround, still in place)

The workaround below predates the fix and is still what `fused_program.py` ships, so the
forward does not depend on a locally-patched pypto. It can be simplified back to a plain
`pl.create_tensor(..., init_value=1.0)` once the fix is upstream.

`gla/implementations/pypto/fused_program.py` derives its ones column from an existing tile:

```python
a_seed = pl.load(A, [0, dof], [C, BK])
g0 = pl.add(pl.tile.reshape(pl.tile.col_sum(pl.mul(a_seed, 0.0)), [BK, 1]), 1.0)
```

Two more direct routes are closed, and are worth recording because they look obvious:

* `pl.tile.full([BK, 1], dtype=pl.FP32, value=1.0)` — rejected: `'pto.alloc_tile' op expects
  result row-major none_box tile row byte size (cols * sizeof(dtype)) to be 32-byte aligned,
  but got 4 bytes`. An **allocated** tile one fp32 column wide is illegal; a **derived** one
  (`reshape(col_sum(...), [BK, 1])`) is fine.
* `pl.load(zero, [dof, 0], [BK, 1])` out of a `[DK, DV]` tensor — rejected at device build:
  `TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ`. A width-1 slice of a wider
  ND tensor is a layout change; loading `[BK, 1]` out of a tensor that is itself `[·, 1]`
  (as the ring does with `gamma`) is fine.
