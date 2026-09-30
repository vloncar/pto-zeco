# Distributed codegen drops program order between comm dispatches → deterministic P>1 deadlock

**TL;DR.** A chip dispatch becomes one `submit_next_level` DAG node whose dependencies are
derived *only from tensor tags*. A comm window carries no tag edge, so two dispatches that
interact solely through `remote_store`/`notify`/`wait` are independent to the scheduler and the
program order written in `host_orch` is discarded. Because the scheduler routes a task to its
per-worker FIFO as soon as it is READY, a dispatch whose only job is `pld.system.wait` — which
typically has **no producer at all** — is routed *ahead of the same rank's still-pending send*.
One task per worker, so the spin-wait owns the core and the send never runs.

This is deterministic, not a race.

## Symptom

Both ranks stall; the run dies with

```
finalize_native_run failed with code -100        # PTO2_ERROR_SCHEDULER_TIMEOUT
sub_class=S1:running-stalled (detail=1) completed=0/1 running=1 ready=0 waiting=0 orch_done=1
```

Note `waiting=0`: nothing was blocked on fan-in, which is the tell — the stalled task had no
unmet dependency, it was simply dispatched too early.

## Reproducer

`repro.py` in this directory. Two ranks, trivial `t+t` compute, one ring each direction:

```
per rank:  1 compute   2 SEND   3 compute   4 WAIT   5 compute
```

**There is no cyclic dependency.** Both ranks send at dispatch 2 with nothing in front of the
send, and only then wait at dispatch 4, so both notifies are issued long before either rank
waits. This program cannot deadlock for any ordering reason a user could reason about. It
deadlocks anyway, 100% of the time.

Corroboration from the error line itself: a healthy run reports `dispatch_id=5` (all five
dispatches ran); both stalled runs report `dispatch_id=2` — the *second* dispatch on the rank.
After the first compute, the next thing dispatched is the wait, not the send.

## Root cause

Three documented behaviours compose into the bug.

1. **Dependencies come only from tensor tags.** `runtime/docs/orchestrator.md` §7: *"tags inside
   TaskArgs drive deps"*, tracking RAW and WAW over a producer-keyed tensormap. A comm window is
   passed as `Tensor.make(..., child_memory=True)` and the peer-side effect of
   `remote_store`/`notify` is invisible to it, so **no edge is created between a rank's send and
   its own later wait**. Program order in `host_orch` is not preserved anywhere.

2. **The per-worker FIFO is ordered by readiness, not submission.**
   `runtime/docs/scheduler.md`: *"a task is routed only once all producers complete"* and *"both
   immediately-ready submissions and dependency-released consumers use the same routing
   operation"*. `directed-next-level-scheduling.md`: *"Submission records each live producer in
   the consumer's `fanin_count`"* — so zero producers means READY at submission.

3. **One task at a time per worker.** `scheduler.md`: dispatch only when the target worker is
   idle, strict FIFO head, *"no idle-worker search, rebinding, work stealing, or scan into
   another worker's queue"*.

Applied to the reproducer, rank 0's generated `host_orch` is:

| dispatch | tensor args | ready when |
| --- | --- | --- |
| `c_compute` | program inputs | **at submission** |
| `c_send` | `a0` ← `c_compute` | after compute |
| `c_compute` | `a0` ← `c_compute` | after compute |
| `c_recv` | `Rb[0]` (Out), `bdst`, `bsig` — **no producer** | **at submission** |
| `c_compute` | `a1` | later |

`FIFO[0] = [c_compute, c_recv]`. The worker runs the compute, then dispatches **`c_recv`, the
blocking wait**, ahead of `c_send`. `c_send` becomes ready and queues behind it, but the worker
is never idle again. Symmetric on rank 1. Deadlock.

The runtime states the violated contract outright
(`directed-next-level-scheduling.md`): *"A NEXT_LEVEL single does not wait for a
not-yet-dispatched peer at the same scheduler level. Work that requires concurrent placement
uses the group API."* pypto's codegen emits only singles.

## Why the usual explanations are wrong

A full factorial at P=2, each cell measured, with controls holding buffer count and dispatch
count fixed:

| arm | buffers | dispatches | every rank waits? | send+wait in one dispatch? | result |
| --- | --- | --- | --- | --- | --- |
| one ring | 2 | 4 | no | – | passes |
| one ring, padded | 2 | 5 | no | – | passes (kills "dispatch count") |
| two rings, SAME direction | 4 | 5 | no | – | passes (kills "buffer count") |
| two rings, comm fused per rank | 4 | 5 | **yes** | **yes** | **passes** |
| two rings, comm split | 4 | 5 | **yes** | **no** | **deadlock** |

So it is not "two windows", not the buffer count, not the dispatch count, and not the cyclic
wait as such. It is precisely *a rank's wait sitting in a different dispatch from its send*.

`pld.system.wait` documentation is also not the explanation. `docs/en/user/distributed/
04-debugging.md` says *"ensure every rank calls notify before any rank calls wait"* — the
reproducer **obeys** that rule and still deadlocks, while the fused-comm arm **violates** it
(wait-then-send inside one kernel) and passes.

## Fix

`fix.patch` in this directory. Give each comm domain a per-rank ordering token and thread it
through every comm dispatch as `INOUT`, so a rank's comm dispatches form a WAW chain in program
order:

```python
# _alloc_intermediates (pre-fork)
tensors["__comm_d0_ord"] = torch.zeros((max(world_size, 1), 1), dtype=torch.int32).share_memory_()

# every dispatch carrying a DistributedTensor arg, appended as the LAST TENSOR
_ta_N.add_tensor(make_tensor_arg(tensors["__comm_d0_ord"][r, 0:1]), TensorArgType.INOUT)
```

Four properties make this cheap and contained:

* **No parallelism is lost.** A worker executes one task at a time regardless, so the token
  constrains ORDER, never concurrency. Compute dispatches are untouched.
* **Host-side only — no chip regeneration.** `aicpu_orchestration_config` validates
  `actual_arg_count < expected_arg_count`, so a trailing extra arg is accepted and the generated
  orchestration entry simply never indexes it. `expected_arg_count` is unchanged.
* **Appended as the last TENSOR**, before the scalars — `TaskArgs` rejects a tensor added after
  a scalar, and tensor/scalar args are indexed independently so nothing shifts.
* **Allocated pre-fork** in `_alloc_intermediates`, which gains a `world_size=1` parameter
  (the domain's worker list may be `*range(world_size)`, so it cannot be sized at codegen time).
  Allocating it inside `host_orch` instead fails with `Failed to stage tensor N to device`,
  because the chip children are already forked and cannot see the new mapping.

Dependency keys are exact-pointer (`TensorKey::operator==` compares `ptr` and `worker_id`, no
range/overlap logic), so per-rank slices four bytes apart cannot alias into false cross-rank
edges.

### Alternative considered

The cleanest fix would be an explicit task-to-task edge, which already exists one level down as
`L0TaskArgsWithDeps::add_dep` and is the sanctioned way to express an edge the tag-based
dep-gen cannot infer (`runtime/docs/war-anti-dependency.md`). The host-level `TaskArgs` has no
equivalent — only `add_tensor`/`add_scalar` — so the ordering has to be smuggled through a
tensor. **Exposing `add_dep` on the host submit API would let this be one edge per pair with no
synthetic tensor at all**, and is probably the better long-term shape.

## Validation

Full detail in `VALIDATION.md`. Headline, on a2a3 with the comm canary green:

* `repro.py` and its sibling variant go from `SCHEDULER_TIMEOUT -100` to **completing in ~2.2s
  with correct payloads**, with no change to the program.
* No regression: `tests/ut/codegen/` + `test_distributed_compiled_program.py` **865 passed**;
  a 2-rank AllScan forward suite incl. its P=4 race guard **4 passed**; the backward **3 passed**.
* The workload that motivated this — a fully-fused ZeCO/GLA distributed backward that
  deadlocked at *every* `P>1` config — is now numerically correct at P=2 (`err < 1e-3`, 3
  seeds) with **no change to the operator itself**.

Before the codegen change the same conclusion was reached by hand: adding one artificial RAW
edge (the send's `Out` threaded into the recv as an extra `INPUT`, payload-neutral) turned the
deadlock into a 2.3s pass 4/4 while the unmodified controls stalled 2/2 — the mechanism
isolated with nothing else varied.

## Secondary request

Even with the ordering fixed, a program that violates this should not present as an opaque
500-second scheduler timeout. Either reject it at compile time — the codegen already validates
comm structure elsewhere (`materialize_comm_domain_scopes_pass.cpp:249` and `:261` reject a
non-unit-step `device=` loop and a `device=` Var that is not a `pl.range` induction variable) —
or document that comm dispatches carry no implicit ordering.
Nothing in `docs/en/user/distributed/` currently says program order is not preserved.
