# Upstream PR draft — hw-native-sys/pypto

`hw-native-sys/pypto` has no pull-request template; this follows the shape of the repo's own
commit/PR history (problem → cause → change → validation).

- base: `main`, head: `vloncar/pypto:fix/comm-dispatch-ordering`
- base commit: `71020585278b68f56c72c40d5570f07dbb20bc8b`
- files: `src/codegen/distributed/distributed_codegen.cpp`,
  `python/pypto/runtime/distributed_runner.py`,
  `tests/ut/codegen/test_distributed_codegen.py`

**Title:** `codegen(distributed): keep program order between a rank's comm dispatches`

---

## Problem

Fixes #<ISSUE>.

A chip dispatch becomes one `submit_next_level` DAG node whose dependencies are derived only
from tensor tags. A comm window carries no tag edge, so dispatches that interact solely through
`remote_store` / `notify` / `wait` are independent to the scheduler and the program order written
in `host_orch` is discarded. The scheduler routes a task to its per-worker FIFO as soon as it is
READY, and a dispatch whose only job is to wait usually has no producer at all — so it is routed
*ahead of that rank's own still-pending send*. One task at a time per worker, so the spin-wait
owns the core and the send never runs.

Deterministic, not a race. Any program where every rank waits and at least one rank's wait sits
in a different dispatch from its send deadlocks with
`SCHEDULER_TIMEOUT / S1:running-stalled ... waiting=0`.

## Change

Give each comm domain a per-rank ordering token and thread it through every comm dispatch as
`INOUT`, so a rank's comm dispatches form a WAW chain in program order:

```python
# _alloc_intermediates, pre-fork
tensors["__comm_d0_ord"] = torch.zeros((max(world_size, 1), 1), dtype=torch.int32).share_memory_()

# each dispatch carrying a DistributedTensor arg, appended as the LAST TENSOR
_ta_N.add_tensor(make_tensor_arg(tensors["__comm_d0_ord"][r, 0:1]), TensorArgType.INOUT)
```

Three details worth reviewing explicitly, each of which the first attempts got wrong:

1. **Appended as the last _tensor_, not the last arg.** `TaskArgs` rejects a tensor added after a
   scalar ("cannot add tensor after scalar"), and comm dispatches end in `add_scalar(device_ctx)`.
   `EmitCallToWorker` now buffers scalar emissions and flushes them after the token. Tensor and
   scalar args are indexed independently, so real args keep their indices.
2. **Allocated pre-fork**, in `_alloc_intermediates`. Allocating inside `host_orch` fails with
   `Failed to stage tensor N to device`, because `w.init()` has already forked the chip children
   and they cannot see a mapping created afterwards. `_alloc_intermediates` therefore gains a
   `world_size=1` parameter (defaulted, so an older caller still loads) — a domain's worker list
   may be `*range(world_size)`, so the token cannot be sized at codegen time.
3. **Only comm dispatches are affected.** The token is added when a call has a
   `DistributedTensorType` argument; compute dispatches are untouched.

### Why this costs nothing

A worker executes one task at a time (`runtime/docs/scheduler.md`: dispatch only when the target
worker is idle, strict FIFO head, no scan past it). The token therefore constrains **order**, not
concurrency — it removes a reordering freedom that was never observable except as this bug.

It is also host-side only: `aicpu_orchestration_config` validates
`actual_arg_count < expected_arg_count`, so a trailing extra arg is accepted and the generated
orchestration entry never indexes it. No chip `.cpp` regeneration and no `expected_arg_count`
change.

Dependency keys are exact-pointer (`TensorKey::operator==` compares `ptr` and `worker_id`, with
no range/overlap logic), so per-rank slices four bytes apart cannot alias into false cross-rank
edges.

## Alternative, if maintainers prefer it

The cleaner fix is an explicit task-to-task edge. One already exists a level down as
`L0TaskArgsWithDeps::add_dep`, and `runtime/docs/war-anti-dependency.md` recommends it as the way
to express an edge the tag-based dep-gen cannot infer. The host-level `TaskArgs` has no
equivalent — only `add_tensor` / `add_scalar` — so ordering has to be smuggled through a tensor.
**Exposing `add_dep` on the host submit API would make this one edge per pair with no synthetic
tensor at all.** That is a runtime-side change, so it is not attempted here; happy to redo this
on top of such an API if that is the preferred direction.

## Tests

| gate | result |
| --- | --- |
| `tests/ut/codegen/` + `tests/ut/ir/test_distributed_compiled_program.py` | 865 passed |
| reproducer from the issue (2 ranks, comm split across dispatches) | deadlock → **completes in ~2.2 s**, payloads correct |
| 2-rank AllScan forward suite incl. its P=4 back-to-back race guard | 4 passed |
| AllScan backward suite | 3 passed |
| fully-fused sequence-parallel backward, P=1 / P=2 / P=4 | 3 passed (deadlocked at every P>1 before) |

Regression arms that already passed and still pass: one ring; one ring padded to 5 dispatches;
two rings in the same direction; two rings with each rank's comm fused into one kernel.

Two golden assertions in `tests/ut/codegen/test_distributed_codegen.py` are updated for the
intentional `_alloc_intermediates` signature change. That is the only test edit.

Hardware: Ascend 910B (a2a3), Linux aarch64, ptoas 0.57.

## Docs

No user-facing API changes, so no doc updates are included. Note for maintainers: nothing in
`docs/en/user/distributed/` currently states that comm dispatches carry no implicit ordering —
if you would like that documented, say the word and I will add it to both the EN and ZH trees
(the `check-docs-en-zh-parity` hook requires both).
