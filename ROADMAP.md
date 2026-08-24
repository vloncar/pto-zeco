# ZeCO Roadmap

**Status: 2026-08-12.** ZeCO = sequence-parallel single-head Gated Linear Attention (GLA)
built on the AllScan collective. Backends: `torch` (reference), `torch.distributed`
(reference SP decomposition), `simpler` (hand-written PTO-ISA kernels), `pypto`
(fully-fused distributed `@pl.program`).

The question this work exists to answer: **for a real sequence-parallel operator, how do
hand-written kernels compare with generated ones** — in correctness, in reachable problem
size, and in steady-state cost.

## Where we are

| Piece | torch | torch-dist | simpler | pypto |
|---|---|---|---|---|
| **Forward** (GLA compute + AllScan boundary) | ✅ | ✅ | ✅ HW P=1/2/4 | ✅ HW **`C ≤ 128`**, head dims **to 1024** |
| **AllScan-collective backward** (building block) | ✅ | ✅ | ✅ HW | ✅ HW |
| **ZeCO/GLA operator backward** (dQ,dK,dV,dA) | ✅ | ✅ | ✅ HW P=1/2 | ✅ HW **P=1/2/4** — `C ≤ 32`, `D ≤ 64` (no blocking yet — Task A6) |

**B4 is correct on HW at P=1, P=2 and P=4.** The pypto backward is one fully-fused
distributed program, the same shape as the forward. The `P>1` hang that blocked it for weeks
was a **pypto codegen bug, not an operator bug** — the operator needed no change — and is
root-caused and fixed below.

**Correction (2026-08-14): the "silently discarded cross-rank write" this section previously
claimed does not exist.** That conclusion came from a family of probes that replaced
`pld.system.wait` with a plain load so runs would complete instead of hanging — which deleted
the only construct providing cross-rank ordering, leaving the receiver free to read before the
sender's write landed. A self-controlled re-read test (same program, receiver reads the same
window three times with compute chains between) settles it: **read1 6/36, read2 36/36,
read3 36/36, never-arrived 0/36.** Payload and signal always arrive, always together. The
sender, the peer address translation, the store path and delivery itself are all *proven
correct*; the probes were measuring their own missing synchronisation. Those probes are
removed (archived in `../.archive/b4-waitless-probes-invalid-2026-08-14.tar.gz`).

**ROOT CAUSE FOUND AND FIXED (2026-08-14) — it is a pypto codegen bug, not signal accounting.**
A chip dispatch becomes one `submit_next_level` DAG node whose dependencies come *only from
tensor tags*. A comm window carries no tag edge, so dispatches that interact solely through
`remote_store`/`notify`/`wait` are independent to the scheduler and **the program order written
in `host_orch` is discarded**. The scheduler routes a task to its per-worker FIFO as soon as it
is READY, and a dispatch whose only job is to wait usually has *no producer at all* — so it is
dispatched ahead of that rank's own send. One task per worker, so the spin-wait owns the core
and the send never runs. Deterministic, not a race; `waiting=0` in the stall dump is the tell.

In B4 exactly: rank 0's `chip_bwd_terminal` (the reverse-ring wait) takes only `zerov`,
`dStot[r]`, `dgam[r]` and the two windows — no local producer — so it is dispatched ahead of
`chip_orch_first`, the forward send rank 1 is parked on. That also explains why *both* bisect
arms passed: stubbing either ring deletes one of the two blocking waits.

Fix: give each comm domain a per-rank ordering token, threaded through every comm dispatch as
`INOUT`, so a rank's comm dispatches form a WAW chain in program order. It costs no parallelism
(a worker runs one task at a time anyway) and is host-side only. Issue + reproducer + patch in
`../allscan/issues/pypto-comm-dispatch-ordering/`. See task 4.

It is also *narrower* than the forward in reachable shapes (`C ≤ 32` against the forward's
`C ≤ 64`): its widest kernel holds about twice the working set. That is a measured ceiling,
not an estimate, and task 5's job to lift.

Both correctness gates that dominated this roadmap for months are **closed**: the cross-rank
producer race and the `N = L//C > 2` loop-carry corruption. Neither was ever a bug in this
operator — both were toolchain defects, and both fixes are now upstream.

**Two carried patches — a rebuild from stock silently loses either one:**

1. **pto-isa FIFO local-slot fix** (MR !1457, merged upstream; our pin predates it — see task 1).
   Without it `dv < C` corrupts every dispatch and `dk < C` about 1 in 20, *silently*. With it
   all head-dim ratios are correct and the shape guard is gone.
2. **pypto comm-dispatch ordering fix** — filed upstream 2026-08-17 as **pypto issue #2397 /
   PR #2398**, carried locally until it merges (see task 4 and
   `../allscan/issues/pypto-comm-dispatch-ordering/`). Without it every `P>1` B4 config
   deadlocks. It spans a C++ codegen change **and** `python/pypto/runtime/distributed_runner.py`,
   so reverting only the `.so` leaves a mismatched pair that raises `TypeError`; revert or
   re-apply both halves together.

   **Rebased onto current main 2026-08-20** (it was 38 commits behind). The rebase broke it
   twice, in ways no textual conflict and no green local suite could show — see
   `../allscan/issues/pypto-comm-ordering-rebase/ROOT-CAUSE.md`:
   (a) `make_tensor_arg` gained a `worker` first parameter, leaving both token emissions one
   argument short; (b) host-tensor handles are now memoized **by storage base**, so the token
   — rows of one tensor sliced per rank — fused every rank into a single dependency node and
   a barrier could not progress. Fix: one separate allocation per rank, in a list.
   Confirmed by a control run (unmodified main `OK`, branch before the fix wrong, branch
   after `OK`) and by 12/12 examples + 10 098 unit tests + the distributed STs locally, all
   in the isolated main environment.

Re-validated on HW 2026-08-12, **on the updated stack** (pypto main / ptoas 0.57 / simpler
`3165cc89`):

| suite | result |
|---|---|
| forward — `test_pypto_gla`, `test_simpler_gla`, `test_pypto`, `test_simpler` | **21 passed, 1 failed** |
| backward — `test_simpler_gla_backward`, `test_gla_backward`, `test_zeco_autograd`, `test_*_backward` | **41 passed, 0 failed** |
| AllScan race guard `test_pypto_allscan_back_to_back[4]` | **passed** (F1 still fixed on 0.57) |

The single forward failure is `[2-128-64-32-64]`, unchanged by the update and now understood
as the `dk < C` case of task 2. Do **not** read a per-P pattern into a single pytest run: the
failure is intermittent, and dedicated repeats show P=1, P=2 and P=4 all affected.

## Environment

Updated 2026-08-12 (was pypto `f621eca4` / ptoas 0.54 / simpler `9922afdb`).

| | |
|---|---|
| pypto | `71020585` (main) |
| ptoas | **0.57** — venv at `/opt/ptoas-venv`, exposed as `/opt/ptoas-bin/bin/ptoas` |
| pto-isa | pin `83d01313` (`/opt/pto-isa`), **+2 local patches** — the DIR_BOTH V2C ring offset (`69a81f3b`) and the consumer local-slot stride (MR !1457) |
| simpler runtime | `3165cc89` (pypto submodule `runtime/`) |

pypto main still pins pto-isa at `83d01313`, which predates `69a81f3b`, so **the DIR_BOTH
patch has to stay carried** — it is not dropped by moving pypto forward. The pypto-side F2
patch *was* dropped: it is upstream as #2271. The pto-isa pin is set by
`runtime/pto_isa.pin` in the simpler submodule, not by pypto directly, so it only moves when
the runtime bumps it.

**ptoas layout gotcha (new at 0.57).** 0.57 installs as a Python venv, and the container's
`PATH` puts `/opt/ptoas-bin/bin` *ahead of* `/usr/local/python3.12.13/bin`. Installing the
venv directly at `/opt/ptoas-bin` therefore shadows the system `python3`/`pip` — which
silently installed pypto **into the ptoas venv** on the first attempt. The venv now lives at
`/opt/ptoas-venv` and `/opt/ptoas-bin/bin` holds a single symlink to `ptoas`, so nothing can
shadow the interpreter. Keep it that way; `PTOAS_ROOT=/opt/ptoas-bin` only needs
`$PTOAS_ROOT/ptoas` or `$PTOAS_ROOT/bin/ptoas` to be executable. Also note the release's
bundle tarball requires CPython **3.11** and we run 3.12 — use the **cp312 wheel**.

Run notes: `LD_PRELOAD=<cann>/lib64/libhccl.so` or HCCL hangs at rootinfo; `PYTHONPATH` must
*prepend* `pto-zeco` (`set_env.sh` resets it); delete stale `/tmp/barrier_pto_multi_comm_*`;
keep multi-card sets inside one HCCS group (0-3 | 4-7). Backup of the previous environment,
with restore instructions: `/root/env-backup-2026-08-12/RESTORE.md`.

---

# What's done

## Forward

- **F1 — cross-rank ring race.** Reproduced deterministically (262/640 rings wrong at 128²),
  root-caused to a missing producer drain before `TNOTIFY`, and fixed at the framework level.
  Now fixed **upstream** (pypto #2183 phi, #2168 `InsertCommFence`, ptoas 0.54's combined
  `PIPE_ALL` fence lowering); no local workaround remains.
- **F2 — the `N > 2` chunk failure.** A bidirectional cube↔vector `TPipe` indexed **both**
  rings from the same GM base, so the cube's matmul results and the vector's operands
  overwrote each other. Fix needed two halves, both **merged upstream**: pto-isa `69a81f3b`
  (per-direction entry offset) and pypto **#2271** (size the GM workspace for both rings).
  Fused forward went 5/12 → **12/12 at C=16 and C=32**.
- **F3.1 — pypto shape ceiling `C=32` → `C=64, D=64`,** by deleting scaffolding rather than
  tiling. `mask` was bit-identical to `tril`; `ones_cc`/`ones_cdv` existed only to broadcast
  the whole-chunk decay by matmul (`col_sum` + `row_expand_mul` does it directly); and the
  state update factors so `gamma` comes out front. Net **5 fewer live tiles, 2 fewer matmuls,
  1 fewer transpose per chunk**, ~35% smaller vector footprint. `C=D=64` sits at **96%** of
  the 184 KB budget — the last shape reachable without real blocking.
- **F3.1b (part) — head dim 128 on HW, either side.** Two changes, both hardware-validated
  (regression 22 passed / 1 skipped at P=1 and P=2). **Head-dim blocking** (`bf9a3db`): the
  chunk kernels walk `dk` in blocks and sum the two head-contracting matmuls' partials in the
  **vector** unit, since fp32 cube accumulation is broken on a2a3; block count and cross-core
  ring depth come from a search that keeps the first setting that fits, so shapes that already
  worked keep their old choice. `C=64 dk=128 dv=64` at 7.6e-05. **Staging `V` in L1**
  (`ce0c840`): the 512 KB `Mat` space was entirely unused while the 184 KB vector buffer
  overflowed — `C=64 dk=64 dv=128` at 5.7e-05. Dims need only be a multiple of **16**, not
  powers of two: 48/80/96 all run, 24 and 40 are rejected by codegen.
- **F3.1b (part) — `dk = dv = 128`, by not carrying the state.** Head-dim blocking then hit a
  wall it could not move: the `[dk,dv]` state was a *loop carry*, live across the whole kernel,
  and three copies of it at 128×128 are 60% of the vector budget however finely everything else
  is cut. The recurrence is linear in its starting state, so the carry is removable —
  `S_n(boundary) = G_n · boundary + S_n(0)`. stage1 already walks from zero, so it now records
  `S_n(0)` and `G_n` per chunk and stage2 rebuilds each chunk's state a `[BK,DV]` block at a
  time. stage1's own blocks are independent for the *whole* slice (nothing in it contracts over
  the head dim), so its block loop moved outside the chunk scan and its carry is `[BK,DV]` too.
  `C=64 dk=dv=128` runs at **7.6e-05 (P=1)** and **9.2e-05 (P=2)** — a shape unreachable at any
  blocking before. It also picks *cheaper* settings for shapes that already worked
  (`dk=128 dv=64` went from 4 head blocks / ring depth 1 to 2 / depth 4). Costs `N·dk·dv` of
  stores and makes stage1 live at P=1, where it used to be dead code — one extra dispatch.
  Chunks are now mutually independent, which is a parallelism lever we did not have.
  Found en route, and now the reason a ones vector is built on-device: **`pl.create_tensor(...,
  init_value=<non-zero>)` silently delivers zeros when the tensor is created in a HOST
  orchestrator** (honoured in a chip orchestrator; `init_value=0` honoured everywhere) — no
  error, no warning. See `../allscan/issues/pypto-host-init-value-zeroed/`.
- **F3.1b (done) — chunk 128, by blocking the key-row axis.** The last shape ceiling, and the
  reason it survived so long is that it was misdiagnosed: `C=128` was recorded as failing on
  operand width when it actually missed by **256 B of vector buffer**. A third of that buffer is
  the cross-core ring reserve, sized by the widest tile *crossing* cube↔vector — at `C=128` the
  `[C,C]` score matmul **result**, which is fp32 by construction and so cannot be narrowed.
  Blocking the key-row (contraction) axis shrinks it exactly: `tril[:, r]` already carries the
  causal zeros, so each block is an independent product needing no scan, and the MAC count is
  unchanged. Both chunk kernels now walk it, and the mask reuses the same `[C, BC]` tile as the
  decay operand. `C=128 dk=dv=128` runs at **1.411e-04** with the search choosing
  head=4/value=2/key-row=2 on its own; forward sweep **30 passed / 1 skipped**. Latency measured
  unchanged (two interleaved A/B passes, 30 iterations; the box is shared and the spread swamps
  any effect). Evidence: `../allscan/issues/pypto-c128-wall/`.
- **F3.1b (part) — value dim blocked, and the ring with it. Head dims reach 1024 on HW.**
  What still scaled with `dv` was the `[C,dv]` output accumulator and the `[C,dv]` right-hand
  operand of `scores @ V` — the latter a 64 KB L0 operand at `dv=256`, the whole buffer.
  Nothing contracts over `dv`, so it splits cleanly; both chunk kernels walk it in `NV` blocks.
  The split is a loop *around the whole chunk body*, not a second pass, so **at one value block
  the body is exactly what it was** — measured: forcing 2 and 4 value blocks on shapes that do
  not need them reproduces the unsplit answer to the digit. At more than one it recomputes the
  within-chunk decay and score matrix (both depend only on `dk`) per value block, which is why
  the plan search reaches for it last.
  Blocking the chunk kernels then promoted the **AllScan ring** to the binding constraint at
  `P > 1`: it moves the `[dk,dv]` end-of-slice state in `K` pieces and `K` was pinned at 1, so
  `dk=dv=256` needed a single 262144 B tile. `K` is now searched like everything else.
  Hardware (a2a3, worst of 2 seeds): `dk=dv=256` **1.68e-04** at P=1 and **1.83e-04** at P=2;
  `dk=256 dv=64` 1.53e-04 with no value split at all; `dv=256` 5.34e-05; **`dv=1024`
  8.39e-05**; **`dk=1024`** 4.88e-04. The search resolves `dk=dv=1024` in 18 s because each
  (value split, ring split) group is rejected as a whole by one compile rather than eighteen.
  `C=128` still does not fit and fails on the **left**-operand buffer — a `[128,128]` fp32 tile
  is the entire 64 KB — which no blocking here can move. The search now says so and stops.
  That is A3's wall.
- **F3.1c — `dv < C` silently wrong.** Root-caused to pto-isa striding the consumer's local
  L1 FIFO ring by the popped tile's own size while the GM ring beside it uses a fixed
  `SLOT_SIZE`, so two differently-sized tiles held at once alias in L1. For `[M,K] @ [K,N]`
  they overlap iff `N < M` — `K` cancels. One-line fix: bare-matmul probe 8/18 → **18/18**,
  GLA 3/7 → **7/7**. Filed upstream. (The original write-up added "which is why `dk` never
  mattered" — **false**, see task 2: `dk < C` hits the same predicate through
  `b = tril[C,C] @ la[C,dk]`, it just does so intermittently.)
- **F3.2 / F3.3 — simpler at any tile size** `{16,32,64,128}`, square and rectangular `C≠D`,
  via runtime-scalar → compile-time template dispatch. Found and fixed two real bugs en
  route, one of them a pto-isa pipe-tagging gap (`TCOLEXPAND` needs `PIPE_ALL`).
- **F3.5 — simpler correctness sweep**, 8 curated shapes, HW-pass P=1 (~1e-7) + P=2 rect.
- **F4 — steady-state benchmark harness.** `measure()`/`amortized_timing` on `ZeCoImpl`;
  pypto builds its `DistributedWorker` once, taking forward from ~9 s to **11.5–55.7 ms**
  (~200–800×). F4.1 (a persistent multi-callable simpler worker) was investigated and was a
  **negative result on the old runtime** — silent corruption — but is now viable and shipped
  as F6.2 on runtime `9922afdb`. **Closed out 2026-08-17**: the residual constraint (a
  callable *re-dispatched* after others have run, which the backward needs and the forward
  does not) was probed directly against the runtime and is also correct — see
  `allscan/issues/simpler-l3-callable-redispatch/`. Both directions now share one held-worker
  path; the only surviving constraint is that a device hosts one worker at a time, so the
  compute workers are dropped around the boundary AllScan (`_release_devices`).
  The probe was turned into an upstream ST and filed as **simpler PR #1938** (2026-08-20) —
  the existing `dynamic_register` cases build two callables that compute the *same* value, so
  they pass whichever one runs. Review finding addressed 2026-08-21 (`47e60a86`): both cases
  now open their `try` at `Worker(...)` rather than after `init()`, since a failed startup
  otherwise skips `close()` and leaves the card held. 7/7 on `a2a3sim` and on `a2a3`.
- **F5 — shared-box hardening.** Rendezvous auto-clean, `LD_PRELOAD` check, per-config
  device-health guard, failure isolation in `finally`.
- **F6.1–F6.4 — the first honest forward numbers.** All 6 shared-config cases correct;
  simpler amortized and verified **bitwise identical** to the safe path. The decomposition is
  the real finding: **stripping all worker lifecycle, simpler costs ~142 ms/call vs pypto's
  12.15 ms for its whole fused forward.** The hand-written *compute kernels* are fast — 6.0 ms,
  below pypto's entire call — but the *boundary collective* is ~11× pypto's forward. The fused
  single-program design wins on the boundary, not on kernels. **99.5%** of simpler's
  as-implemented 29 s/call is worker lifecycle, a runtime-integration artifact.
- **F7.1–F7.6 — generalization.** `dk != dv`; a differentiable `torch.autograd.Function` +
  `nn.Module` wrapper over any backend; multi-head; boundary-AllScan amortization
  (17.3 → 11.85 → **8.85 s/head**, ≈49%); batching folds into the head axis; arbitrary `L`.

## Backward

- **B1 / B2 — math + references.** Chunk-recurrent GLA backward derived as the reverse of the
  forward; `expected_gla_backward` analytic golden cross-checked against autograd; torch and
  `torch.distributed` SP decompositions matching across `P∈{1,2,4} × dk≠dv × C`.
- **B3 — simpler operator backward, HW-validated.** Two new orchestrations and **no new device
  kernel** — every backward matmul maps onto the general `matmul` kernel, and the gate
  backward's reverse-cumsum reuses `gate_cumsum` fed an upper-triangular matrix. P=1 5.7e-7,
  P=2 6.7e-7, P=2 `dk≠dv` 3.5e-7.
- **B5.1–B5.3 — hardening.** 10-shape sweep 10/10; 16-dispatch back-to-back stress 16/16;
  steady-state latency measured and attributed (orchestration-setup-bound, not compute).

---

# Remaining work

Three tasks. Everything finished is folded into **What's done** above or into
`../allscan/issues/`; only what is still open appears here, split into stages.

Closed out and removed from this list since the last revision: carrying the merged pto-isa
fixes and dropping the `min(dk,dv) >= C` guard (2026-08-13), the `dk`-side corruption root
cause (2026-08-12), the forward-at-scale sweep (2026-08-12), **B4** — the pypto fused
distributed backward, correct on HW at P=1/2/4 (2026-08-17), **B5.4** — steady-state numbers
for both directions and both backends (2026-08-18), and the measurement discipline that made
the median the headline (2026-08-19). Root causes for all of them live in `../allscan/issues/`.

---

## Task A — Realistic problem sizes

**The target.** Head dims 64 / 128 / 256, chunk 64 (128 if it earns its keep), `L` 1–4k,
multi-head. `L` is already unbounded — it is the chunk-loop trip count and no tile is
`L`-sized. What is bounded is the per-chunk working set.

**What changed on 2026-08-21.** Two measurements reset this task:

* The 512 KB `Mat` (L1) space was **entirely unused** — the allocation dump showed
  vec/left/right/acc/ddr and no mat — while we fought a 184 KB vector budget. Staging `V`
  there took `C=64, dv=128` from impossible to passing.
* The `[dk,dv]` state carry was **60% of the vector budget** in three copies, and no head-dim
  blocking could touch it. It is removable: the recurrence is linear, so
  `S_n(boundary) = G_n * boundary + S_n(0)`, and stage1 already walks from zero. Proven exact
  in fp64 (24 shapes, 3.6e-15) and **proven on a2a3**: a carry-free stage2 runs `C=64,
  dk=dv=128` at 6.1e-05, a shape the carried version could not reach at any blocking.

Design and measurements: `../allscan/issues/pypto-tile-blocking/ARBITRARY-SIZES.md`.
Probes: `../devtools/t5_{math_check,block_probe,snapshot_math,nocarry_probe,verdict,split_check}.py`.

**A1 and A2 are done** (2026-08-21) — snapshot state instead of a carry, then value-dim
blocking and a searched ring split; see the F3.1b entries under Forward. Both head dims now
reach 1024 on hardware. The stage labels below keep their original numbers so earlier notes and
commit messages still resolve.

**A4 is done** (2026-08-24) — `C=128` runs on hardware. See the F3.1b entry under Forward.
A3 was investigated first and could not deliver it; it stays available as an optional, lossy
performance lever and is not a size-unlocking stage.

    A5, A6, A7 remain.   A3 optional (lossy; operand-buffer pressure + traffic only)

### A3 — Narrower matmul operands *(investigated 2026-08-24 — does NOT deliver `C=128`)*
**The description this section used to carry was wrong, and the correction matters.** It said
`C=128` fails on operand width, that a `[128,128]` fp32 tile is the whole 64 KB buffer, and that
"no amount of tiling addresses it". That came from the failure message at ONE blocking setting;
the search only ever reports the first buffer to overflow. Walking the whole space says `C=128`
misses on the **vector buffer by 256 B** (head=4, value=2), with the two operand buffers
4096 B short. Full evidence: `../allscan/issues/pypto-c128-wall/`.

Narrowing cannot fix the binding constraint. 65536 B of the vector budget — a third — is
reserved by the cross-core ring, sized by the largest tile *crossing* cube↔vector, and at
`C=128` that is the `[C,C]` score matmul **result**. Results cannot be narrowed:
`pl.matmul: out_dtype=fp16 is not supported ... the Cube accumulator fixes the result dtype`.
Casting `tril` (lossless, it is only 0.0/1.0) does not help either — the score result is the
same size and the reserve takes the max. Mixed precision is refused outright, so `tril` cannot
go narrow while `la` stays fp32.

GLA also cannot narrow most operands at all: `k/b` reaches **8.8e+07** against fp16's 65504, so
the score matmul and stage1 update go **NaN**. These are not flash-attention's operands.
Measured cost where narrowing IS legal: fp16 3.0e-04 relative, bf16 2.4e-03 (device-measured,
matching the torch study).

**What is left of A3:** an opt-in performance lever for `Left`/`Right` pressure and operand
traffic, at a deliberate ~3e-04 relative cost. Not a size-unlocking stage. Do not schedule it
as one.

### A4 — Block the two `[C,C]` matmuls over their CONTRACTION axis *(DONE 2026-08-24)*
Both `[C,C]` matmuls split exactly over their contraction (key-row) axis — verified in fp64,
residual 2.8e-14 at every block size:

    b       = tril @ la    =  sum over key-row blocks r of  tril[:, r] @ la[r, :]
    o_intra = scores @ v   =  sum over key-row blocks r of  scores[:, r] @ v[r, :]

The widest tile the cube ever sees becomes `[C, BC]`, so the ring reserve shrinks with it — and
the reserve is what `C=128` is short of.

**This is NOT the rewrite this section used to describe.** It was written as flash-attention row
blocking, whose hard part was that "the within-chunk decay is a cumulative product down rows, so
it becomes a sequential scan carrying a `[1,BK]` running total". That is true when blocking the
output ROWS. Blocking the CONTRACTION needs no scan at all: `tril[:, r]` already carries the
right zeros, and every block is an independent product summed afterwards. Mechanical, exact, and
the same shape of change as A1/A2.

Cost, as with the value split: the per-key-row-block quantities are recomputed once per block,
so keep the search reaching for it last. Chunk size remains a tuning knob rather than a model
dimension — within-chunk work grows linearly with it while state work shrinks, so past ~128 it
is strictly worse. Useful range 64–256; bound it with a clear error rather than chasing
arbitrary values.

### A5 — Block sizes from a cost model *(partly relieved)*
The search is no longer the bottleneck it was: rejecting a whole (value split, ring split) group
with one compile brought `dk=dv=1024` from ~100 attempts to ~25, and `dk=dv=256` at `P=2` to 25
seconds. What remains is that "first that fits" is a *feasibility* rule, not a *performance*
one — it has no way to prefer a plan that fits comfortably over one that barely fits.
`blocking_plans` + `compile_fused_forward` currently try candidate settings and keep the first
that fits. That is the right shape for a kernel that *nearly* fits and the wrong shape for a
properly tiled one. Once A1–A3 land, choose block sizes from occupancy / pipe depth / traffic
with the budget as a constraint. Keep the search until then — it fails honestly, which a
hardcoded table would not.

**One correctness gate for the whole task**: the plan search must reject a plan whose device
kernel cannot build. `ir.compile` returning success does not mean the kernel builds — the
device kernels are built at `prepare()` — and that gap produced two wrong conclusions on
2026-08-21. `../devtools/t5_split_check.py` detects the known case statically;
`../devtools/t5_verdict.py` reports bytes, wrong-core ops **and** a real build.

### A6 — Same treatment for the backward
The four backward kernels have a tighter ceiling than the forward (`grad_o` is the widest in
either direction) and none of A1–A3 has been applied to them.

### A7 — F6.5, the fair numbers at realistic sizes
The point of the whole task. Needs A1–A3, and F6.6 below for a compute-vs-comm split to mean
anything.

---

## Task B — Fully on-device work placement (F6.6)

Steps 1–2 done 2026-08-19; results `../devtools/F66-STEP2-RESULTS.md`. pypto is already
98–99.9% on-device. simpler's off-device share is **4–5% of the forward's compute and ~10% of
the backward's**, and 0.03–0.13% of its whole call.

**The constraint that decides the design**: a dispatch costs **~33 ms (fwd) / ~39 ms (bwd),
flat** in `L` and `D` — the round trip, not the kernel. So a port that *adds* a dispatch is a
net loss: a standalone `shift_snaps` kernel would cost ~33 ms × P to save 0.5–2.1 ms.
Everything must fold into an existing dispatch.

**And the honest proportion**: the boundary collective is **26.6–27.1 s per phase, flat in
P/L/D** — 83–91% of every simpler call, with the backward paying exactly 2×. Compute-worker
stand-up after a boundary release adds 2.1 s (P=2) / 4.2 s (P=4). Against that, everything in
this task is worth 8–20 ms forward and 42–85 ms backward. It is parity work, not a speedup.

### B1 — Fold the leftovers into existing dispatches
`_S_total` is free: `chunk_h_orch.cpp` already computes it as the carried state and discards
it — widen `s_snap` to `[N+1,dk,dv]` and point `S` at slot N. `_shift_snaps` and `torch.log(A)`
fold into dispatches that already exist.

### B2 — The backward's three per-chunk Python loops
78–87% of the remaining glue; `gate_h_loop` alone is 53–58%. Merge the duplicated multi-head
copy (`gla/implementations/pypto/impl.py:848-928`) first.

---

## Task C — Upstream

Two findings from task A, both written up, **neither filed**. Re-test each against pypto
`origin/main` before filing — the last two dependency bugs written up here were already fixed
upstream. Isolated env: `../devtools/pypto_main_env.sh`.

### C1 — A pre-loop GM→GM copy lands on the cube core
`../allscan/issues/pypto-cube-side-gm-copy/`. Triggered by a tensor-to-tensor copy before a
kernel's main loop when the kernel also has a nested loop: 0 bad ops at 1 block, 3 at 2/4/8.
The cube has no vector buffer, so it cannot build. `ir.compile` returns success and **a2a3sim
passes**, so only a hardware build catches it.

### C2 — The cross-core ring depth knob is not where its error message says
The overflow message names `pl.cross_core_slot(slot_num=N)` "on the enclosing `pl.at(...)`",
but a declared `@pl.function(type=InCore)` has none and adding one fails. `ExpandMixedKernel`
reads it off a **function attribute**. Worse, the non-deprecated `pl.func_attr` accepts only a
**literal** — a shape-derived depth reaches the pass as an `ir::Expr` and is rejected — so the
only route that takes a computed value is the deprecated `attrs={...}` decorator form.

### C3 — Already filed, awaiting review
pypto **#2398** (per-rank comm ordering token; 14 checks green) and simpler **#1938** (L3
dispatch runs the callable it names; review addressed 2026-08-21).

---

# Parked

Deliberately not doing, with the reason:

- **F3.4 — simpler tiles > 128 (head dim 256).** Output (M/N) tiling to 256 works, but
  reducing `Kc > 128` needs `TMATMUL_ACC` across 128-blocks, and **fp32 cube K-accumulation is
  unsupported on a2a3** — plain accumulate drops a block, explicit `AccPhase` hangs the AICore
  (`507018`), and the CPU sim accumulates fine so it cannot catch either. Blocks exactly the
  two matmuls contracting over the head dim. Write-up:
  `allscan/issues/fp32-cube-k-accumulation/`. The deferred design is recorded there; it is the
  same vector-unit-summation shape as task 5.
- **F7.5b — fused `[H·dk,dv]` collective.** Would cut the residual ~6.3 s/head ring, but needs
  AllScan K-blocking past the 128-row cap plus a head dim threaded through every compute
  orchestration. F7.5a already captured the cheap share of the win.
- **F7.7 — dims outside `{16,32,64,128}`** (e.g. `D=96`). Host-side zero-padding to the next
  dispatchable size. The dispatchable set already covers real head dims.
- **F5's last item — per-backend process isolation.** The health guard made cross-config
  contamination rare enough that this has not been needed.

---

# Upstream ledger

| What | Where | State |
|---|---|---|
| F2 — a2a3 DIR_BOTH rings alias the same GM slots | pto-isa issue **#516** / MR **!1438** | **merged / closed** |
| F2 — size the GM pipe workspace for both rings | pypto PR **#2271** | **merged** (`d26e0f6c`) |
| F1 — producer drain before `TNOTIFY` | pypto #2183, #2168 + ptoas 0.54 | **fixed upstream** |
| F3.1c — consumer local FIFO ring aliases | pto-isa issue **#521** / MR **!1457** | **merged** — carried locally until the pin moves |
| CPU-SIM `TASSIGN` arity break | — | **fixed upstream** (`439faf48`); write-up retired |
| simpler second-callable silent corruption | — | **gone** on runtime `9922afdb`; write-up retired |
| `comm_alloc_domain_windows -1` | — | environment, not a code bug; retired |

Written up but **not filed**, kept for reference:
`fp32-cube-k-accumulation` (real a2a3 limitation, blocks F3.4), `tcolexpand-pipe-sync`
(`TCOLEXPAND` is tagged `PIPE_V` but lowers to a copy-engine op — our fix is `PIPE_ALL`;
note pto-isa PR212's `PIPE_MTE1` retag **faults the AICore**, so `PIPE_ALL` is the only
correct pipe), `pypto-jit-distributed-segfault`, `taskqueue-unresolved-auto-device`.

---

# Constraints worth not rediscovering

- **One callable per simpler worker.** An L3 chip child binds to the first callable it runs;
  a later dispatch of a *different* registered callable returned wrong data with **no error**
  on runtime `a756969c`. Fixed on `9922afdb`, and F6.2 gates on a bitwise-identical check
  against the safe path — keep that gate.
- **Device exclusivity forbids holding an AllScan across compute**, which is why per-phase
  batching (F7.4/F7.5a) is the achievable amortization and cross-call persistence is not.
- **`halMemCtl rc=42` is not a bad card.** It is `DRV_ERROR_PROCESS_EXIT`, a per-process
  device-context error — once traced to a co-tenant container leaking ~20k workers. `npu-smi`
  reports Health OK and queue occupancy is blind to non-queue users. Check `ps` and re-run a
  canary elsewhere before blaming hardware.
- **fp32 ≠ fp16 on the cube.** K-accumulation and the FIFO aliasing boundary both differ by
  dtype; a2a3 gemm references are all fp16 and do not exercise the fp32 paths.
- **The CPU sim is not a correctness oracle for these bugs.** It was clean for F2, F3.1c and
  the fp32 accumulation break alike. Every one of them was HW-only.
