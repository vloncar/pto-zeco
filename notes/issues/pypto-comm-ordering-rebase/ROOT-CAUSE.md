# PR #2398 after rebase: the ordering token collapses into ONE dependency node

**Status:** root-caused 2026-08-20, fix designed, NOT implemented (cannot be tested locally).

## What happened

PR #2398 was **fully green** before the rebase (all 14 checks on `e6a09064`, including
`examples-tests`, `dist-system-tests`, `pypto-lib-model`). Rebased onto `c53e4f9f`, which
itself passes `examples-tests` on main, the PR now fails:

* `03_window_buffer.py` — **passes**
* `04_barrier.py` — **fails**, `PTO2 runtime failed with rc=-100`, `sched_error_code=100`

So the fix was correct against the older main and something main moved invalidates it.

## Root cause

The token is one tensor, sliced per rank. From the generated orchestration:

```python
tensors["__comm_d0_ord"] = torch.zeros((max(world_size, 1), 1), dtype=torch.int32).share_memory_()
...
_ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["__comm_d0_ord"][r__idx_v0, 0:1]),
                 TensorArgType.INOUT)
```

The newer simpler runtime memoizes host-tensor handles **by storage base**, and says so in
`Worker.make_tensor_arg`:

> "The handle is memoized by the tensor's storage base, so every ref over the same storage
> shares one canonical identity **and dependencies key on it**."
> "Memoized per storage base so every view of one storage shares an identity and **their
> dependencies key together**."

`[r, 0:1]` for every `r` is a view of ONE storage. So all ranks' tokens collapse into a
single dependency node: instead of the intended **per-rank** WAW chain, every rank's comm
dispatches serialise behind every other rank's.

That explains the pass/fail split exactly. `03_window_buffer` has no cross-rank wait, so a
global chain is merely over-ordered, not fatal. `04_barrier` requires all ranks to be in
flight simultaneously — a global chain makes rank 1 wait for rank 0's wait, which never
retires. Hence the scheduler error rather than a wrong result.

## Fix

Give each rank its **own storage**, so the runtime's per-storage memoization yields the
per-rank identity the design assumes:

```python
tensors["__comm_d0_ord"] = [torch.zeros((1,), dtype=torch.int32).share_memory_()
                            for _ in range(max(world_size, 1))]
...
_ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["__comm_d0_ord"][r__idx_v0]),
                 TensorArgType.INOUT)
```

A Python list indexed by the rank expression, one distinct allocation per element. Touches
`_alloc_intermediates` emission and both token emission sites (`distributed_codegen.cpp`,
`distributed_ops_codegen.cpp`).

## Why this was not caught before pushing

It cannot be reproduced on this box: current pypto main needs simpler `1f27a157`, which
needs pto-isa `f51c92f6`; we are pinned to simpler `3165cc89` + pto-isa `83d01313` with two
carried patches the `dk<C`/`dv<C` shapes depend on. Our older runtime does **not** memoize by
storage base, so the sliced token gives distinct identities here and the design works.

Two separate rebase breakages came from the same gap on the same day:

1. `make_tensor_arg` gained a `worker` first parameter — emitted call left one argument
   short. No textual conflict; C++ compiled; 185/185 codegen unit tests passed; nothing in
   `tests/ut` reaches the token emission. Fixed.
2. This one — a *semantic* change with no API signature to notice.

Neither is visible to a green local suite. See `../../pto-zeco/ROADMAP.md` environment note.

## Full CI result after the `make_tensor_arg` fix (head `1bc0ec16`)

| pass | fail |
|---|---|
| build, changes, toolchain, pre-commit, clang-tidy, CodeRabbit | **examples-tests** |
| unit-tests, codegen-tests | **pypto-lib-model** |
| system-tests, system-tests-direct, system-tests-a5sim, mixed-kernel-tests-a5 | **dist-system-tests** |

All three failures are the one root cause: the collapsed ordering token deadlocks anything
that needs ranks in flight simultaneously. Everything that does not is green, which is the
same signature as the `03` vs `04` split.

## Plan (chosen 2026-08-20): build the environment before touching the fix

Blind CI iteration costs ~15-20 min per attempt and cannot bisect. Instead, stand up a
**local** copy of what main pins, isolated from our own pinned tree:

* pto-isa `f51c92f6` fetched to `/tmp/pto-isa-new` (upstream is `hw-native-sys/pto-isa`).
* simpler `1f27a157` built from `/tmp/pypto-pr/runtime` into its own `.venv`
  (`--system-site-packages`, per the repo's venv-isolation rule), `PTO_ISA_ROOT` pointed at
  the fetched pto-isa. The runtime enforces `HEAD == pto_isa.pin`, and both are `f51c92f6`.
* `/opt/pto-isa` and `/opt/pypto/runtime` are **not** touched: pto-zeco stays on simpler
  `3165cc89` + pto-isa `83d01313` with its two carried patches.

Success criterion: reproduce the `04_barrier.py` failure locally. Only then is the per-rank
allocation fix worth writing, because only then can it be shown to work.

## Fix applied and verified locally (2026-08-20)

The token is now **one separate allocation per rank**, held in a list:

```python
tensors["__comm_d0_ord"] = [torch.zeros((1,), dtype=torch.int32).share_memory_()
                            for _ in range(max(world_size, 1))]
...
_ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["__comm_d0_ord"][r__idx_v0]),
                 TensorArgType.INOUT)
```

Each element is its own storage, so the runtime's per-storage memoization yields the
per-rank dependency identity the design assumes instead of fusing all ranks into one node.

**Control run, which is what makes this a diagnosis rather than a guess** — same isolated
environment, same example, `04_barrier.py -p a2a3sim -d 0,1`:

| tree | result |
|---|---|
| unmodified `origin/main` (`c53e4f9f`) | `OK` |
| this branch, before the fix | `barrier mismatch: got [[[0],[0]],[[1],[0]]] expected [[[0],[1]],[[1],[0]]]` |
| this branch, after the fix | `OK` |

Note the local symptom is a **wrong result** (rank 1 never observes rank 0's signal) while
CI showed `sched_error_code=100`. Same cause, different manifestation -- worth remembering
before treating a scheduler error and a wrong answer as unrelated bugs.

Full local validation (`devtools/validate_pypto_main.sh`, the stand-in for CI's
`examples-tests` + `dist-system-tests`):

* 12/12 example invocations pass -- all seven plain, plus `04 --use-builtin`,
  `05 --mode store`, `06 --mode get`, and `07` at 3 and 4 ranks.
* `tests/st/distributed`: 16 passed, 147 skipped (hardware-only), 1 deselected.

## Lesson

Three separate breakages came out of one rebase, and **none was visible to a green local
suite** on our pinned tree:

1. `make_tensor_arg` gained a parameter -- a signature change, no textual conflict.
2. Host-tensor handles became memoized by storage base -- a pure *semantic* change with no
   signature to notice at all.
3. (Same class, in the ST file: `Tensor` -> `TensorArg` and `_build_l3_task_args` gaining a
   required `worker` -- caught before pushing only because the sibling test was diffed
   between the two revisions first.)

The general rule: **when a dependency moves, diff a file that uses it the same way you do,
rather than relying on the compiler or the test suite to tell you what changed.** A rebase
reports textual conflicts, not invalidated assumptions.

## Does pto-zeco have the same pattern? No.

Checked, because it would surface the day our pin moves to a runtime that memoizes by
storage base. Both simpler-backend paths already allocate one **separate** buffer per rank:

* `allscan/implementations/simpler/impl.py:252-261` — `host_s`, `host_g`, `host_out`,
  `host_gout`, `host_outprev`, `host_dS`, `host_dgamma` are each
  `[torch.zeros(...).share_memory_() for _ in range(P)]`.
* `gla/implementations/simpler/impl.py:192, 271` — one `share_memory_()` allocation per
  dispatch argument.

No per-rank slice of a shared allocation anywhere, so the collapse this issue describes
cannot occur in our code. The per-rank-list shape the pypto fix adopts is the same one our
backend already uses.

## Round 2: the CI failure after the fix was the assertion, not the code

`dist-system-tests` came back `1 failed, 114 passed, 45 skipped`. The one failure was the
ST assertion added by this series:

```python
assert '_ord"] = torch.zeros(' in orch_src, "comm ordering token never allocated"
```

The fix changed the allocation to `[torch.zeros(...) for _ in range(...)]`, so the literal
no longer matched. The check was pinning the **spelling** rather than the property.

Replaced with one that pins the property the fix exists to establish:

```python
assert re.search(r'_ord"\] = \[torch\.zeros\(.*for _ in range\(', orch_src), (
    "comm ordering token is not a per-rank list of separate allocations")
```

Verified both directions against real generated output: passes as emitted, and fails when
the allocation is mutated back to rows of one tensor. So a future change that re-collapses
the token now fails this test instead of deadlocking a barrier.

Also confirmed here: **pyright passes once the correct runtime is installed.** The two
`"DataType" is unknown import symbol` errors that failed the hook all day were version skew,
which the main-branch comparison had already implied and the isolated environment now shows
directly.

Fourth instance of the day's theme, and the mildest: an assertion that matched a literal
string rather than the invariant behind it. **Assert the property, not the spelling.**
