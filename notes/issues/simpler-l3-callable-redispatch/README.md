# L3 callable re-dispatch on a held Worker — RESOLVED upstream, workaround removed

**Status:** closed as **fixed upstream, nothing to file**. Verified on HW 2026-08-17.
**Component:** simpler runtime (L3 `Worker` / chip-child callable staging).
**Platform:** a2a3 (910B2), simpler `3165cc89`, pto-isa `83d01313`, CANN 9.0.0.
**Supersedes:** the archived `simpler-second-callable-silent-corruption` write-up
(`.archive/issues-resolved-2026-08-12.tar.gz`), which described the same mechanism as
open on simpler `a756969c`.

## What was believed

An L3 chip child binds to the **first callable it actually runs**; dispatching a
different registered callable on that same worker afterwards silently returns wrong
data. Measured on `a756969c` (2026-07-22) through the GLA backward: dQ/dK/dV correct
(~4e-7) but **dA at `max_rel = 1.0`**. dA is the only output fed by the backward's
*second* `gate_cumsum` dispatch (the reverse-cumsum, after `chunk_h`, `grad_o`,
`grad_h`), which is what pinned the cause.

Precision matters about what was still believed by 2026-08-17, because the tree held two
contradictory statements:

* **ROADMAP F6.2** had already shipped a persistent *multi-callable* forward on runtime
  `9922afdb`, "verified bitwise identical to the safe path" — three different callables
  on one held worker, working.
* **`_ComputeRunner`'s docstring** still asserted that any second dispatch of a
  different callable silently returns wrong data, and told the reader not to optimise it.

So the live belief was narrower than the docstring: the forward was known fine, and what
remained un-retested was the **backward's re-dispatch** — the same callable dispatched
again after others have run. That is why `_persistent_dispatches()` listed only the
forward's three kernels and the backward kept paying a fresh worker per dispatch: ~5P
worker fork/init/close cycles per call plus two HCCL builds, which dominated its
measured latency.

## What is actually true now

`repro.py` tests the mechanism directly against the runtime, with no GLA code involved.
It builds **two** callables from the runtime's own `vector_example` orchestration by
permuting which AIV binary sits at `func_id` 0 and 2 — `kernel_add` and `kernel_mul`
take an identical argument layout, so the permutation is signature-safe but the two
callables compute visibly different functions:

| | func0 | func2 | f |
|---|---|---|---|
| **X** | add | mul | `(a+b+1)*(a+b+2) + (a+b)` |
| **Y** | mul | add | `c = a*b` ; `((c+1)+(c+2)) * c` |

Distinct binaries per `func_id` ⇒ distinct hashids **and** distinct goldens. That second
half is the point: the runtime's own multi-callable coverage
(`tests/st/a2a3/tensormap_and_ringbuffer/dynamic_register`) dispatches two handles whose
kernels compute *the same* result, so it structurally cannot observe a dispatch that ran
the wrong callable. It is also marked `platforms(["a2a3sim"])` — sim only, no HW
coverage. The earlier local probe (`devtools/probe_persistent_worker.py`) had the same
blind spot from the other side: its multi-callable case asserted only
`finite and nonzero`, which is not a correctness check when the failure mode is silent
wrong data.

Result on one held worker, every callable registered and every buffer allocated
pre-`init()`, different inputs per dispatch, each output checked against its own exact
golden:

```
single      X              -> PASS   (err 0.00e+00)
same2       X X            -> PASS   (err 0.00e+00 both)
multi2      X Y            -> PASS   (err 0.00e+00 both)
redispatch  X Y X          -> PASS   (err 0.00e+00 all three)   <- the backward's pattern
alternate   X Y X Y X Y    -> PASS   (err 0.00e+00 all six)
control     X Y X (fresh worker each)  -> PASS
```

Not "close enough" — **bit-identical**, on all six sequences. Multi-callable reuse and
re-dispatch are correct on our pin.

## Where it was fixed

Not bisected (each point costs a build plus an HW run), so this is inference from the
124 commits between `a756969c` and `3165cc89`. The two that describe the mechanism:

* **`39f5cdd9` Refactor: isolate hierarchical worker run state (#1459)** — "**Namespace
  TensorMap entries by run** and protect concurrent access". A TensorMap not namespaced
  per run is precisely how a second dispatch would collide with the first dispatch's
  entries and return the wrong data.
* **`da75d350` Refactor: isolate resources per worker run (#1466)** — "the cleanup at a
  run's fence swept Worker-level registries and could free a resource another run still
  owned".

Both landed as *refactors*, which is why nothing in a changelog announced a
silent-corruption fix and why the constraint stayed in our docstrings long after it
stopped being true.

## Consequences (applied)

`gla/implementations/simpler/impl.py`:

1. `_persistent_dispatches()` now lists **every** kernel either direction runs
   (`grad_o` and `grad_h` added). The reverse-cumsum needs no slot of its own — it is a
   second `gate_cumsum` dispatch with identical shapes, so it reuses that slot.
2. `_forward_persistent()` **deleted**. It differed from `forward()` only by dropping
   the compute workers before the boundary AllScan; that is now `_release_devices()`,
   called from both `forward()` and `backward()`. One code path per direction, serving
   both runner shapes — the per-kernel runner's `close()` is a no-op.
3. `measure_backward()` added, so the backward is timed in steady state like the
   forward. This is what B5.4 was missing.
4. The silent fallback is **gone**. `_gate_persistent()` still checks the held path
   against the per-kernel path before timing either direction, but **raises** on a
   mismatch instead of quietly reverting and leaving `amortized_timing` False. A silent
   fallback is how B5.4 came to report non-amortized numbers under an `SS=Y` heading.

The one real constraint that remains: **a device hosts one worker at a time**, so a held
compute worker still has to be dropped around the boundary AllScan. Removing that needs
comm and compute to co-reside in one worker — a separate, larger runtime question.

## GLA-level verification

`devtools/check_persistent.py`, held path vs per-kernel path vs torch golden, x2 repeats
per direction. Two runs; **vs_plain is 0.00e+00 in every cell of both**, i.e. holding a
worker across dispatches is bit-identical to cycling one per dispatch.

Run A — cards 0-3, canary green on all six pairs (shared input point, see caveat):

| config | fwd vs ref | fwd vs plain | bwd vs ref | bwd vs plain |
|---|---|---|---|---|
| P=1 L=128 | 2.29e-05 | **0.00e+00** | 2.34e-07 | **0.00e+00** |
| P=2 L=128 | 1.91e-05 | **0.00e+00** | 2.60e-07 | **0.00e+00** |
| P=2 L=256 | 1.72e-05 | **0.00e+00** | 3.00e-07 | **0.00e+00** |
| P=4 L=128 | 1.72e-05 | **0.00e+00** | 3.00e-07 | **0.00e+00** |

Run B — cards 0,1, per-config seeds (P=4 skipped, only two cards were grantable):

| config | fwd vs ref | fwd vs plain | bwd vs ref | bwd vs plain |
|---|---|---|---|---|
| P=1 L=128 | 1.53e-05 | **0.00e+00** | 2.35e-07 | **0.00e+00** |
| P=2 L=128 | 2.29e-05 | **0.00e+00** | 4.09e-07 | **0.00e+00** |
| P=2 L=256 | 2.29e-05 | **0.00e+00** | 3.07e-07 | **0.00e+00** |

The backward rows are the ones that matter: they are the case that used to give dA
`max_rel = 1.0`. `P>=2` additionally covers `_release_devices()` — dropping and reopening
the held workers around both boundary phases — which P=1 skips entirely. P=4 is covered by
run A only.

Two caveats found while running this, both fixed in code rather than footnoted:

* In run A, P=4/L=128 and P=2/L=256 reported byte-identical errors. Cause:
  `make_gla_inputs` **seeds torch itself** (default 42), so an outer `manual_seed` is
  overwritten, and with the same seed and the same `P*L` token count those two configs are
  literally the same input point reshaped. The seed is now passed through per config,
  which is what run B's distinct numbers show.
* The canary is a **gate**, not advisory — and I had demoted it to advisory on the
  incorrect belief that `comm_alloc_domain_windows` was pypto-only. The simpler boundary
  reaches it too, via `orch.allocate_domain` → `Worker._allocate_domain` →
  `_dispatch_control_domain`. That cost a run: on VIS=2,3 the canary said "comm-capable
  pairs: NONE" and the job died 8 minutes later inside `_boundary_on` with exactly that
  error. Restored as a hard gate; it then correctly aborted a 4,5,6,7 grant in 2 minutes.

## Files

* `repro.py` — the reproducer. Self-contained on the runtime's own example kernels, so
  it can be dropped into `tests/st/` unchanged if upstream wants the coverage.
* `devtools/simpler_redispatch_run.sh` — HW runner (single card, no HCCL).
* `devtools/check_persistent.py` — the GLA-level check, both directions, held vs
  per-kernel vs torch golden.

## Lesson

The bug report was right when written and stale when used. It sat in a docstring that
said "do not optimise this without re-checking", and for three weeks nobody re-checked —
while the dependency moved 124 commits. **Re-test a dependency bug against the current
pin before designing around it**, and give the probe an oracle strong enough to see the
failure mode it is looking for: the previous probe would have passed on corrupt data.

## Upstream contribution: `test_l3_callable_isolation.py`

`repro.py` is packaged as a system test for `hw-native-sys/simpler`, to land at
`tests/st/a2a3/tensormap_and_ringbuffer/test_l3_callable_isolation.py`.

**What it adds that the existing coverage cannot reach.** The two callables in
`dynamic_register/test_dynamic_register.py` differ only by an unused child entry at
`func_id` 99, so they compute the same value — its own docstring says "Both identities
execute equivalent kernels and must produce numerically identical outputs." A dispatch that
ran the wrong one produces exactly the value that test asserts. Here X and Y compute
different functions (47 vs 90 on the first input pair), the test asserts they differ for
every input pair used, and a wrong-callable result is reported as such rather than as an
ordinary numeric mismatch. It also runs on **hardware**; all five `dynamic_register` cases
are `a2a3sim`-only, and upstream has since marked four of the five `@pytest.mark.manual`,
which is `exclude` by default — so most of that coverage does not run automatically either.

**Verified**: 6/6 on `a2a3sim` and 6/6 on `a2a3` hardware (cards 0-3, single device, no
collectives), plus a no-hardware case that asserts the two callables still disagree and that
each of the four failure messages fires.

**Which API version.** The runs above were on our pin `3165cc89`. The file here targets
**simpler main**, which is 175 commits ahead and changed two things the test uses:

| our pin `3165cc89` | simpler main |
|---|---|
| `from simpler_setup import Tensor` | `TensorArg` |
| `_build_l3_task_args(args, sig)` | `_build_l3_task_args(args, sig, worker)` |

Nothing else the test touches moved — `KernelCompiler`, `CoreCallable.build`,
`ChipCallable.build`, `extract_text_section` and `ensure_pto_isa_root` are unchanged, which
is why the same file serves both with those two edits. Retargeting was done by diffing the
sibling `test_dynamic_register.py` between the two revisions rather than by guessing.

`test_l3_callable_isolation.pin-3165cc89.py` is the exact file that produced the hardware
results, kept so the claim above is checkable rather than asserted:

```
diff <(sed 's/TensorArg/Tensor/g; s/, _ORCH_SIG, worker)/, _ORCH_SIG)/g' \
        test_l3_callable_isolation.py) test_l3_callable_isolation.pin-3165cc89.py
```

is empty — the two files differ by those two API edits and nothing else.

**The main-targeted form is therefore not locally runnable**: this box is pinned to
`3165cc89` for the pto-zeco work and cannot import current main (see the environment note in
`../../pto-zeco/ROADMAP.md`). The hardware evidence is from the pin-API form of the same
file; main's form is CI-verified. That gap is exactly what bit pypto PR #2398 on the same
day — a rebase left an emitted call one argument short because upstream had changed a helper
signature, with no textual conflict and a green unit suite. Same failure mode, caught here
before pushing because the sibling-test diff was checked first.

## Verified against main, including that it can fail

Run in the isolated main environment (`devtools/pypto_main_env.sh`), against the simpler
revision pypto main pins (`1f27a157`), platform `a2a3sim`.

**1. It runs.** 6/6 device cases pass — `single`, `same_twice`, `two_callables`,
`redispatch_after_other`, `alternating`, and the fresh-worker baseline — plus the
no-hardware `test_the_two_callables_are_distinguishable`.

**2. It is the right "main".** `1f27a157` is main-era but not the tip (`aac09a61`), so the
gap was checked rather than assumed. Every function the test calls is identical across it:
`_build_l3_task_args`, `compile_incore`, `compile_orchestration`,
`get_orchestration_include_dirs`, and the `TensorArg` NamedTuple. The only change to the
sibling `test_dynamic_register.py` over that span is adding `@pytest.mark.manual`. So the
result carries to the tip for everything this test touches.

**3. It detects the defect it exists for.** Fault injected end to end: on the third dispatch
of `X Y X` — the re-dispatch, i.e. the case that historically returned silently wrong data —
the OTHER callable's handle is submitted while the check still expects the named one. The
test fails, and names the culprit rather than reporting a bare numeric mismatch:

```
Failed: dispatch 2:X named callable X but produced callable Y's result (39.375); expected 34
```

The same injection through `dynamic_register/test_dynamic_register.py` would pass, because
its two callables compute the same value — which is the coverage gap this file closes.

## Review finding on the PR, and what it changed

An automated reviewer raised one thing: both hardware cases start their `try` *after*
`worker.init()`, so a `register()` or `init()` that raises skips `worker.close()`.

It is right, and the runtime says so in its own words. `Worker.init` "raises after a
bounded rollback that reaps the children it forked **best-effort** (a child wedged in
native code past the deadline may be left behind)"; a worker in that state answers
`init()` with "Worker startup failed; close this Worker and create a new one"; and
`Worker.close` is documented as the call that "retries journaled teardown debt ... keeps
each resource until its native free succeeds and preserves the child pid/mailbox pair
until `waitpid` proves the child is gone", with "Put the call in a `finally` — a worker
that is never closed keeps its device held."

That failure mode is one we have actually been bitten by: a card left held turns one
failed case into every later case on that card failing.

Both cases now open the `try` immediately after `Worker(...)`, with `register()`,
`_build_l3_task_args()` and `init()` inside it.

**Checked that the new `close()` cannot make things worse.** A `finally` that raises would
hide the real error, which is worse than the leak it fixes. Measured on `a2a3sim`:

| worker state when `close()` is called | result |
|---|---|
| constructed, never `init()`ed (NEW) | returns cleanly |
| `init()` fault-injected to fail after the chip child was forked (FAILED) | returns cleanly |

The fault injection breaks `Worker._start_hierarchical`, which runs after the children are
forked, so the worker really has spent resources when it raises. Child accounting through
`/proc/<pid>/task/*/children` shows the chip child gone and only torch's own
`multiprocessing` helper left, in both the failed-init and the successful paths — so in
this particular fault `init`'s own rollback did the reaping and `close()` had nothing left
to do. That does not make the guard pointless: the rollback is documented as best-effort,
and `close()` is the only thing that retries what it could not finish.

Sibling tests in the same directory (`dynamic_register/test_dynamic_register.py`, all five
cases) have the same narrow shape. Left alone — a test-only PR should not rewrite its
neighbours — but worth a follow-up.

**Re-run after the change**, all 7 cases, both platforms:

| | `a2a3sim` | `a2a3` hardware, device 0 |
|---|---|---|
| `single` | PASS 31.8s | PASS 23.3s |
| `same_twice` | PASS 31.5s | PASS 24.2s |
| `two_callables` | PASS 30.1s | PASS 22.3s |
| `redispatch_after_other` | PASS 30.2s | PASS 24.5s |
| `alternating` | PASS 31.2s | PASS 23.9s |
| fresh-worker baseline | PASS 31.2s | PASS 26.4s |
| `the_two_callables_are_distinguishable` (no device) | PASS 2.7s | — |

`pre-commit run --files <the test>`: all hooks pass, nothing rewritten.
