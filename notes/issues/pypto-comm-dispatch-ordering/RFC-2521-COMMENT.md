Thanks for writing this up — the L2 placement and the requested-vs-actual launch width split
both look right to us, and the HCCL policy table is genuinely useful.

Context for the feedback below: we build [ZeCO][zeco] — sequence-parallel gated linear
attention — on top of PyPTO, using the point-to-point rail (`pld.tile.remote_store`,
`pld.system.notify`, `pld.system.wait`) rather than the managed collectives. The ring
collective underneath it ("AllScan") and the fused operator are public at
https://github.com/vloncar/pto-zeco. It is a work in progress and parts of it are still
moving, but the forward and backward paths referenced below run on A2/A3 hardware today and
can be read and re-run. Everything we report here is measured on that code, not projected.

[zeco]: https://github.com/vloncar/pto-zeco

---

## 1. `_ord` — please scope the removal, and please re-check what it keys on

> The current `_ord` is a hidden per-rank, per-`CommDomain` dummy `InOut` tensor injected by
> L3 distributed codegen. It creates a WAW chain between ordinary Host builtin communication
> dispatches. [...] Remove the old `_ord` path when Host managed collectives are retired.

and, under Alternatives Considered:

> **Serialize every communication task through `_ord`** — This hides the true resource/
> dependency model and prevents safe independent communication from running concurrently.

`_ord` came from #2398 (merged `e2d0f78a`, 2026-08-21). It is not a convenience token for
managed collectives; it is the fix for a deterministic hang in user-written point-to-point
communication, and we would like to make sure the removal does not take that with it.

**What it actually keys on.** The description above suggests `_ord` is attached to Host
builtin communication dispatches. It is not. In
`src/codegen/distributed/distributed_codegen.cpp` (`EmitCallToWorker`), the flag that
triggers injection is set in the branch that binds a **window-buffer argument** — i.e. any
L3 dispatch that carries a `DistributedTensor`, whether or not a managed collective is
involved. Every dispatch in our ring is such a dispatch, and none of them is a managed
collective. So the set of affected programs is larger than the one the RFC describes, and
removing the injection wholesale would regress user-written SPMD, not just the legacy managed
path.

**Why it exists.** A chip dispatch becomes one `submit_next_level` DAG node whose dependencies
are derived *only from tensor tags*. A communication window carries no tag edge, so two
dispatches that interact solely through `remote_store` / `notify` / `wait` are independent to
the scheduler, and the program order written in the host orchestrator is discarded. The
scheduler routes a task to a per-worker FIFO as soon as it is READY, and a dispatch whose only
job is to wait typically has no producer at all — so it is admitted ahead of its own rank's
send. One task per worker, so the spin-wait owns the core and the send never runs. This is
deterministic, not a race; the tell is `waiting=0` in the stall dump. It cost us several
weeks before it was root-caused.

The relevance to this RFC is that **moving the collective to L2 does not by itself repair
this.** The RFC's position is:

> Normal compute -> collective -> compute ordering comes from real TensorMap edges [...]
> These parameter directions must survive L2 lowering.

That holds for a *managed* collective, whose payload tensors are real operands and therefore
do produce tag edges. It does not hold for the case `_ord` was written for, where the only
shared object is the window and the signal. Those carry no edge at either level. The RFC's
stated replacement for that case is:

> User-written SPMD communication continues to use explicit TaskId dependencies.

but that surface (`pl.submit`, `pl.spmd_submit`, `deps=[...]` in
`python/pypto/language/scope.py`) exists only *inside* a chip-level function. At HOST/L3,
where these dispatches live, there is no TaskId to take or pass. That absence is precisely why
a token was needed. If the plan is to remove `_ord`, we think the RFC needs to say what
replaces it for host-level dispatches that communicate — otherwise "explicit dependencies are
preferable" is true but not yet available.

Concretely, we would ask for one of:

1. scope the removal to *managed* collectives that have migrated to L2, and leave the
   window-argument injection in place for user-written communication; or
2. add a host-level explicit-ordering surface first (a returned/accepted token at L3), and
   retire `_ord` behind it.

On the "prevents safe independent communication from running concurrently" objection: that is
a fair characterisation, and we agree it is the right thing to fix eventually. In practice it
costs no parallelism today, because a worker runs one task at a time regardless. We would
rather trade that than reintroduce the hang, but a narrower edge (per-`CommDomain`, or derived
from the window identity) would be strictly better and we would be happy to help build it.

**One implementation trap worth recording**, because anyone rebuilding this at L2 will hit it.
Our first version allocated one tensor and passed each rank a slice of it. The runtime
memoizes host tensor handles **by storage base**, and dependencies key on that identity — so
every rank's slice collapsed into a *single* dependency node, turning an intended per-rank
chain into a global one. `03_window_buffer.py` still passed (no cross-rank wait; a global
chain is merely over-ordered), while `04_barrier.py` failed with `rc=-100`,
`sched_error_code=100`, because it needs all ranks in flight simultaneously. The fix is one
distinct allocation per rank, which is what landed
(`distributed_codegen.cpp:1464` on `c7ba9fb0`). A per-lane or per-rank control object at L2
will need the same care.

---

## 2. "Factor the sequence from an existing validated implementation" is underspecified — what is required depends on push vs pull

> The TPUT-to-notify remote-visibility sequence must be factored from an existing validated
> implementation, including its current barriers/fence (for example `pipe_barrier(PIPE_ALL)`
> and the existing DDR visibility operation). This RFC does not define a new memory-order
> primitive.

We agree with not defining a new primitive. The problem is that there is no single existing
implementation to factor from: the two built-in templates encode **different** disciplines,
and the difference is not arbitrary — it tracks the direction of the data movement.

On `c7ba9fb0`:

| Template | Sequence before the completion signal |
| --- | --- |
| `collectives/all_to_all_v/templates/kernel.cpp.in` | `pipe_barrier(PIPE_ALL)` (167) **then** `dsb(DSB_DDR)` (168), then `TNOTIFY` (173) |
| `collectives/allreduce/templates/kernel.cpp.in` | `pipe_barrier(PIPE_ALL)` only (88, 104, 165, 191); no `dsb` anywhere in the file |

That divergence is consistent, because the two kernels move data in opposite directions.
`all_to_all_v` **pushes**: the block writes into the peer's window with `TPUT` and then
signals, so the payload must be observable at DDR before the credit lands. Mesh `allreduce`
**pulls**: peers `TLOAD` through `CommRemotePtr`, and the local `TNOTIFY` publishes no
freshly-written payload at that point, so a pipe drain is sufficient there.

The RFC's own operator table mixes both families:

- push — `all_to_all_v`, `all_to_all`, `allgather`, `broadcast`
- pull — mesh `allreduce`, `reduce_scatter`

and Phase 0 nominates the **pull** kernel (`93d789bf` / #2160) as the implementation baseline.
Factoring a pull discipline into a push kernel is the specific mistake we would like the RFC
to rule out explicitly, because it does not fail loudly.

**How loudly it fails, measured.** This is the same hazard as PTOAS #744 / #872, which we
reported and which is now fixed upstream (PTOAS PR #873, merged 2026-07-10). From the
on-device data in those threads, for a chunked cross-rank push followed by a signal:

| Publish-before-signal sequence | Result |
| --- | --- |
| pipe drain only (`PIPE_MTE2/3`), i.e. the original | 30/30 fail |
| `dsb(DSB_DDR)` only, no full pipe drain | 10/10 fail |
| `pipe_barrier(PIPE_ALL)` **and** `dsb(DSB_DDR)` | 20/20 pass |

and for the `remote_store` variant specifically, the pipe-drain-only configuration measured
**~2.6% residual corruption**, going to 0% with the DDR release added. We had our own clean
result (0/640 at 128²) with the pipe barrier alone and initially believed the barrier was
sufficient; it was configuration luck. At a ~2.6% rate a short sweep reads clean.

So we would suggest the RFC's parenthetical be inverted: for push kernels the DDR release is
the load-bearing half, and the pipe barrier alone is the configuration that has been measured
to corrupt.

**And the compiled path cannot simply be copied into the templates.** For `pld.tile.remote_store`
the sequence is assembled across three layers: a peer-region cache operation emitted directly
by codegen as an acknowledged workaround, because the peer address is not expressible in the IR
(`src/backend/common/pto_ops_distributed.cpp`, which carries a live TODO to give it a
first-class representation); a `system.fence` inserted by the `InsertCommFence` pass; and
PTOAS's lowering of that fence from #873. The built-in collective kernels are hand-written
`.cpp.in` templates that go through none of those, which is exactly why the two above already
differ. Under this RFC that one open-coded sequence gets multiplied across five more
collectives at `B` lanes each, so it would be worth either (a) stating the required sequence
per family normatively in the RFC, or (b) giving the templates a shared helper so there is one
place to be right.

---

## 3. Independent measurement supporting the motivation

> The task setup, parameter transfer, wake-up, and completion round trips add milliseconds in
> observed workloads and can dominate the collective kernel itself.

We can corroborate this from a different operator. Benchmarking our ring collective on
Ascend 910B2, hand-written PTO-ISA kernels versus PyPTO, at 4 ranks:

- full per-call: **67-200 ms**
- the same work with the domain setup/teardown amortised over 16 exchanges under one domain:
  **flat 32-40 ms**
- at 4 ranks / 128×128 the hand-written and generated versions are **level at ~32 ms**

So the alarming number was almost entirely per-dispatch communication-domain lifecycle plus
drain overhead, and the underlying data movement was competitive all along. That is the same
conclusion as #2069's ~124 ms, reached from a different direction, and it supports both this
RFC and #2069 being worth doing.

One caveat we would flag for your benchmark harness: only one set of worker processes may be
prepared per device set. Two prepared workers fork chip children on the same devices and their
communicators collide (`HcclCommInitRootInfo failed: 7`). It looks like a code failure and
is not one.

Also relevant to Phase 0: our forward/backward benchmarking was **setup-dominated** (~9 s of
prepare/close around the measured region) at small payloads. If the AllToAllV sweep includes
latency-oriented sizes, the harness will need to exclude that region explicitly or the small-`L`
points will be measuring the harness.

---

## 4. Signal reuse across invocations — worth an explicit test, and there is a DSL gap behind it

The testing plan lists:

> repeated signal reuse across many invocations, including interleaved task completion

We would push for this to be treated as a first-class requirement rather than one line in a
list, because the DSL currently makes the *alternative* hard too. Our collective does not
decrement (it is single-shot per window), so to batch 16 independent exchanges we had to
**generate** a host orchestrator with 16 explicitly-named disjoint buffer pairs. Two DSL
limitations forced that: `alloc_window_buffer` names must be globally unique and are
LHS-injected, and `pld.window` takes no offset, so disjoint windows cannot be carved out of
one buffer in a loop.

The RFC's self-clearing credit protocol is the right answer and would remove that need
entirely. We mention it because it means "reuse the signal" is not just a nice property — for
anyone on the point-to-point rail today it is the *only* affordable option, and the workaround
is code generation.

---

## 5. The correctness matrix needs repetition counts attached

The case list is good — skew, zero-length lanes, ragged tails, self-peer, dtype coverage. What
it does not state is how many times each case runs, and in our experience every defect in this
area has been intermittent rather than deterministic.

Two calibration points from our own hardware runs:

- a defect that fires at **5%** survives a 10-run clean sweep **60%** of the time, and a 3-run
  sweep **86%** of the time
- one of our corruption bugs reproduced at about **1 in 20**; a clean run of 20 therefore
  misses it **36%** of the time

We now quote the repetition count next to every result for exactly this reason. Without it,
"quiet shapes" get certified that were never actually tested to the required power — and a
`B`-lane protocol with per-lane credits has considerably more interleavings to get wrong than
a single-block one. It would be worth the acceptance criteria naming a minimum `R` per case,
scaled to the failure rate the criteria intend to exclude.

---

## 6. The EP8/EP16 matrix assumes a healthy box

Minor, but it cost us three misread results. Device pairs on our machine periodically stop
being able to open a communication domain — `comm_alloc_domain_windows` fails with code `-1` —
and at its worst we were down to one usable pair on the whole box. It clears with a reset and
comes back later, so it is a passing condition rather than a property of particular cards. What
makes it expensive is that **a single-rank check passes throughout** and `npu-smi` reports
Health OK, so it presents as an ordinary failure of the code under test. We now front every
multi-rank job with a probe that actually opens a domain before believing a failure belongs to
us. If your EP8/EP16 sweep runs across a shared pool, the same guard is probably worth having.

---

Happy to help on any of this — in particular we can supply the reproducer and stall-dump
signature behind #2398 if that is useful while deciding the scope of the `_ord` removal, and we
would be glad to review the push-family publish-before-signal sequence once there is a template
to look at.
