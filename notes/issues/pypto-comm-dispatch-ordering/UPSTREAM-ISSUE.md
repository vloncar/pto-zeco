# Upstream issue draft — hw-native-sys/pypto

Filed with `.github/ISSUE_TEMPLATE/bug_report.yml`. Field values for the dropdowns:

| field | value |
| --- | --- |
| Component | `Codegen` |
| Git Commit ID | `71020585278b68f56c72c40d5570f07dbb20bc8b` |
| NPU Kind | `Ascend 910B` |
| Host Platform | `Linux (aarch64)` |

**Title:** `Distributed codegen drops program order between comm dispatches, so a wait is scheduled before its own rank's send (deterministic P>1 deadlock)`

---

## Description

A chip dispatch is lowered to one `orch.submit_next_level` call, and the runtime derives that
task's dependencies **only from the tensor tags in its `TaskArgs`**. A comm window is passed as
an opaque `Tensor.make(..., child_memory=True)`, and the peer-side effect of
`pld.tile.remote_store` / `pld.system.notify` / `pld.system.wait` is invisible to that
dependency analysis.

Consequently two dispatches on the same rank that interact *only* through a comm window have no
edge between them, and **the program order written in `host_orch` is not preserved anywhere**.
The scheduler routes a task to its per-worker FIFO as soon as it is READY, and a dispatch whose
only job is to wait usually has **no producer at all** — so it is routed *ahead of that rank's
own still-pending send*. Since a worker runs one task at a time, the spin-wait then owns the
core and the send never runs.

The result is a deterministic deadlock in any program where every rank waits and at least one
rank's wait lives in a different dispatch from its send.

## Steps to Reproduce

```markdown
1. Save the attached `repro.py` (two ranks, trivial `t + t` compute, one comm ring in each
   direction; only `pypto` and `torch` are imported).
2. `LD_PRELOAD=<cann>/aarch64-linux/lib64/libhccl.so python3 repro.py 0,1`
3. The run hangs, then raises `finalize_native_run failed with code -100`.
```

Per rank the program is:

```
dispatch 1  compute
dispatch 2  SEND    (remote_store + notify to the peer)
dispatch 3  compute
dispatch 4  WAIT    (pld.system.wait on the peer's notify)
dispatch 5  compute
```

**There is no cyclic dependency.** Both ranks send at dispatch 2 with nothing in front of the
send, and only then wait at dispatch 4, so both notifies are issued long before either rank
waits. No ordering a user can reason about makes this program deadlock.

## Expected Behavior

Dispatches issued in sequence from `host_orch` for a given rank should execute in that order —
or, if that is explicitly not guaranteed for comm dispatches, the program should be rejected at
compile time rather than hanging. `MaterializeCommDomainScopes` already rejects other
comm-structure mistakes with clear messages — `materialize_comm_domain_scopes_pass.cpp:249`
(*"device=r over a non-unit-step loop is not supported"*) and `:261` (*"device= Var is not the
induction variable of any enclosing pl.range loop"*) — so this class of error has precedent for
a compile-time diagnostic.

## Actual Behavior

Both ranks stall and the run fails with:

```
[ERROR] PTO2 scheduler timeout sub_class=S1:running-stalled (detail=1) \
        completed=0/1 running=1 ready=0 waiting=0 orch_done=1
RuntimeError: ... finalize_native_run failed with code -100 \
        (run_id=1 slot=0 generation=1 dispatch_id=2 run_epoch=2)
```

Two details pin the cause:

* `waiting=0` — nothing was blocked on fan-in. The stalled task had no unmet dependency; it was
  simply dispatched too early.
* `dispatch_id=2` — the stall is at the **second** dispatch on the rank. A healthy run reports
  `dispatch_id=5`. After the first compute, the next thing dispatched is the wait, not the send.

## Additional Context

### Why this happens, in the runtime's own terms

* `runtime/docs/orchestrator.md` §7: dependencies are RAW/WAW over a producer-keyed tensormap,
  driven by `TaskArgs` tags. Nothing else creates an edge.
* `runtime/docs/scheduler.md`: *"a task is routed only once all producers complete"*, and
  *"both immediately-ready submissions and dependency-released consumers use the same routing
  operation"* — so the FIFO is ordered by **readiness, not submission**.
* `runtime/docs/directed-next-level-scheduling.md`: *"Submission records each live producer in
  the consumer's `fanin_count`"* (zero producers ⇒ READY at submission), and dispatch happens
  only when the target worker is idle, strict FIFO head, with *"no idle-worker search,
  rebinding, work stealing, or scan into another worker's queue"*.

The same page states the contract this violates: *"A NEXT_LEVEL single does not wait for a
not-yet-dispatched peer at the same scheduler level. Work that requires concurrent placement
uses the group API."* The distributed codegen emits only singles
(`_submit_chip` → `submit_next_level`), never `submit_next_level_group`.

In the generated `host_orch.py` for the reproducer, rank 0's `c_recv` receives only its `Out`
tensor plus the two windows — no argument that any earlier dispatch produces — so its
`fanin_count` is 0 and it enters `FIFO[0]` immediately, behind only the first compute and
**ahead of `c_send`**, which is still pending on that compute.

### It is not the explanations one reaches for first

Full factorial at P=2, each cell measured, controls holding buffer count and dispatch count
fixed:

| arm | buffers | dispatches | every rank waits? | send+wait in one dispatch? | result |
| --- | --- | --- | --- | --- | --- |
| one ring | 2 | 4 | no | – | passes |
| one ring, padded to 5 dispatches | 2 | 5 | no | – | passes |
| two rings, SAME direction | 4 | 5 | no | – | passes |
| two rings, comm fused per rank | 4 | 5 | **yes** | **yes** | **passes** |
| two rings, comm split (`repro.py`) | 4 | 5 | **yes** | **no** | **deadlock** |

So it is not "two comm windows in one program", not the buffer count, not the dispatch count,
and not the cyclic wait as such — it is precisely *a rank's wait sitting in a different dispatch
from its send*.

It is also not the notify/wait ordering rule in `docs/en/user/distributed/04-debugging.md`
(*"ensure every rank calls notify before any rank calls wait"*): the reproducer **obeys** that
rule and deadlocks, while the fused-comm arm **violates** it (wait-then-send inside one kernel)
and passes, as does the AllScan ring's `allscan_middle_step`, which has the same wait-then-send
shape and is HW-validated at P=4.

### Direct confirmation

Adding **one** artificial RAW edge — the send's `Out` tensor threaded into the recv as an extra
`INPUT`, made payload-neutral by storing `dst + (dep - dep)` — with nothing else changed turns
the deadlock into a **2.3 s pass, 4/4 runs**, while the unmodified controls stall **2/2**. That
is the missing fan-in, isolated with no other variable moved.

### Impact

Any multi-phase distributed program where a rank's comm phases are split across dispatches. It
blocked a fully-fused sequence-parallel attention-style backward (recompute → forward ring →
gradient → reverse ring → gradient) at *every* `P>1` configuration, while the same operator was
correct at `P=1`. Programs that happen to keep all of a rank's comm inside a single `chip_orch`
dispatch — which is what the AllScan ring and
`tests/st/distributed/test_l3_ep_dispatch_combine.py` do — are unaffected, which is why this has
not been hit before.

A fix is attached as a separate PR.
