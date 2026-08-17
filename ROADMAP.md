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
| **Forward** (GLA compute + AllScan boundary) | ✅ | ✅ | ✅ HW P=1/2/4 | ✅ HW `C,D ≤ 64`, any `dk`/`dv` |
| **AllScan-collective backward** (building block) | ✅ | ✅ | ✅ HW | ✅ HW |
| **ZeCO/GLA operator backward** (dQ,dK,dV,dA) | ✅ | ✅ | ✅ HW P=1/2 | ✅ HW **P=1/2** — `C ≤ 32`, `D ≤ 64` (P=4 untested: needs 4 comm-capable cards) |

**B4 is correct on HW at P=1 and P=2 (2026-08-14).** The pypto backward is one fully-fused
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

**One live dependency:** the pto-isa FIFO local-slot fix (MR !1457, merged) is **carried
locally** because our pin predates it — see task 1. Without that patch, `dv < C` corrupts every
dispatch and `dk < C` about 1 in 20, silently. With it, all head-dim ratios are correct and the
shape guard is gone.

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
| pto-isa | pin `83d01313` (`/opt/pto-isa`), **+1 local patch** — the DIR_BOTH V2C ring offset |
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
  as F6.2 on runtime `9922afdb`.
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

Six tasks, in dependency order. Everything else is parked below.

### 1. Carry the merged pto-isa fixes; guard removed — DONE 2026-08-13
pypto is on main and its F2 patch is gone (upstream as #2271). pto-isa is the remaining
carried piece, and it stays carried for a structural reason: **the pin is not ours to move.**
`runtime/pto_isa.pin` in the simpler submodule sets it, the runtime enforces `HEAD == pin`,
and pypto main still pins `83d01313` — which predates both fixes we need.

Both are now **merged upstream** and carried locally against that pin
(`allscan/issues/pto-isa-fifo-local-slot-alias/carried-on-pin-83d01313.patch`):

| fix | upstream | why we carry it |
|---|---|---|
| DIR_BOTH V2C ring offset | `69a81f3b` (MR !1438, issue #516) | pin predates it |
| consumer local-slot stride | MR **!1457** (issue #521) | pin predates it |

With the second one applied, the `min(dk, dv) >= C` guard in `PyPtoZeCo.build` is **removed**
and the four shapes it refused are correctness cases again. Validated at 40 dispatches each:

| shape | before | after |
|---|---|---|
| `C=64,dv=32` / `C=32,dv=16` (`dv < C`) | 20/20 wrong | **0/40** |
| `C=64,dk=32` / `C=32,dk=16` (`dk < C`) | ~1/20 wrong | **0/40** |

`SIZES` regained all four, so **the suite is now the regression test for the carried patch** —
if it is ever lost, those cases go red rather than silently corrupting. The `dk < C` pair needs
`ZECO_TEST_REPEATS` raised to be a meaningful check: at ~1/20, the default 3 repeats would miss
a regression 86% of the time.

Drop the patch and this note when the runtime bumps `pto_isa.pin` past both commits.

### 2. `dk`-side corruption — DONE 2026-08-12, guard extended to `min(dk,dv) >= C`
The failure at `C=64, dk=32, dv=64` is **the same defect as F3.1c**, reached through the `dk`
side. Paired A/B alternating the stock and F3.1c-fixed `TPush.hpp` in one job on one card:
**12/84 dispatches wrong on stock, 0/84 with the fix.**

Mechanism: `b = tril[C,C] @ la[C,dk]` is `[M=C, K=C, N=dk]`, so `N < M` exactly when `dk < C`
— F3.1c's predicate on the `dk` side rather than the `dv` side.

Measured at 20 dispatches per shape, fresh inputs each, one process per config:

| shape | relation | wrong |
|---|---|---|
| `C=64,dv=32` / `C=32,dv=16` | `dv < C` | **20/20 — deterministic** |
| `C=64,dk=32` / `C=32,dk=16` | `dk < C` | **1/20 — intermittent** |
| `C=64,dk=16`, `C=32,dk=16,dv=64` | `dk < C` | 0/20 |
| everything with `dk,dv >= C` | — | 0/20 |

**The two sides are not symmetric in severity**: `dv < C` corrupts every dispatch, `dk < C`
about 1 in 20. That rarity is why it survived a year of testing, and it is also why the guard
does **not** carve out the quiet `dk < C` shapes: at a 5% rate a clean run of 20 still misses
the failure 36% of the time, so "quiet at N=20" is not evidence of safety.

`PyPtoZeCo.build` now requires **`min(dk, dv) >= C`**, naming the offending dimension and
stating whether that side is deterministic or intermittent. `GUARDED_SIZES` gained the two
`dk < C` shapes; `SIZES` swapped the now-guarded `(128, 64, 32, 64)` for `(128, 32, 64, 32)`,
which keeps a `dk != dv` case in the suite. **`test_pypto_gla.py` is green: 14 passed, 1
skipped, 0 failed.**

Withdrawn along the way: it is not `P >= 2`-only (P=1 and P=4 fail too) and not `dk != dv`
(both discriminators are clean). The F3.1c write-up's "which is why `dk` never mattered" was
false and is corrected.

Remaining: tell !1457 that its fix also covers this shape class — **draft for review, not yet
sent**. The guard comes out when the fix merges and the pin moves (task 1).

Method notes (probes in `../devtools/t2_*`): patch `/opt/pto-isa` directly rather than
overriding `PTOAS_ROOT`, which gave byte-identical halves; **prove the header is live** by
injecting `#error` and requiring the sentinel in the compiler output — a null A/B is
meaningless without it; never run a second HW job while one mutates `/opt/pto-isa`; and on an
intermittent failure, never conclude from a single run in either direction.

### 3. Forward-at-scale sweep — DONE 2026-08-12
F2 was fixed and spot-checked at 12/12, but the full sweep had never been run, so "correct at
scale" was an inference. Now measured (`../devtools/f_scale_sweep.py`), N-axis at a fixed shape
plus the shape corners, each config built once and dispatched against **distinct seeds**:

| backend | repeats/config | result |
|---|---|---|
| pypto | 10 | **19/19 configs clean** — `N ∈ {2,4,8,16,32}` × `P ∈ {1,2,4}`, up to `L=1024` (128 chunks at P=4), plus all five reachable shape corners |
| simpler | 3 | **13/13 configs clean** — same axes, plus `C=D=128` |

Worst error anywhere: **1.07e-04** (simpler, `C=D=128`); pypto's worst was 9.16e-05. No
config drifted with `N`, which is the property F2's fix was supposed to restore.

**Read the clean result with its power in mind.** At R repeats a defect corrupting a fraction
`p` of dispatches is missed with probability `(1-p)^R`: pypto's R=10 misses a 5%-rate defect
**60%** of the time, simpler's R=3 misses it **86%**. The sweep prints this itself. It is
strong evidence against a *systematic* N- or P-scaling error and weak evidence against another
rare intermittent one — which is the honest reading given task 2's defect reproduced at ~1/20.
simpler's R=3 is a cost decision: its per-call worker cycling makes a dispatch ~30 s against
pypto's ~1 s.

Test-suite gap closed at the same time: `make_gla_inputs` seeds torch itself (default 42), so
every run had been replaying **one** input point per shape. `_run_case` now dispatches
`ZECO_TEST_REPEATS` times (default 3) against distinct seeds and names the failing seed.

Two configs errored rather than failed — `LocalMailboxEndpoint child failed ... native
finalize`, both immediately after a slow P=2 config, and both **correct when re-run in
isolation**. That is a simpler-runtime teardown race between consecutive build/close cycles,
not a correctness result. Worth watching; it would also hit the bench harness.

### 4. B4 — pypto fused distributed backward — **P=1 + P=2 DONE on HW** (2026-08-14)

**Unblocked 2026-08-14.** `gla/tests/test_pypto_gla_backward.py::test_pypto_zeco_backward`
is **2 passed, 1 skipped** (P=1, P=2; P=4 skips on a 2-card grant) — `err < 1e-3` over 3 seeds.
The operator was never wrong: the blocker was a pypto distributed-codegen bug (comm dispatches
carry no dependency edge, so a fan-in-free `wait` dispatch is scheduled ahead of its own rank's
send and owns the sole worker forever). Fixed by a per-rank comm ordering token; issue,
reproducer and patch in `../allscan/issues/pypto-comm-dispatch-ordering/`. **The fix is carried
locally in `/opt/pypto` and is NOT upstream yet** — a stock pypto rebuild reintroduces the
deadlock. Remaining: P=4, which needs four comm-capable cards.

`gla/implementations/pypto/fused_backward_program.py`: the whole SP backward as ONE
distributed program, five phases per rank with no host round-trip — **recompute → forward
ring → grad_o → reverse ring → grad_h**. Phase 1 recomputes the forward chunk scan and
snapshots `S_prev[n]` / `c_prev[n]` (the backward is stateless by contract, so activations
are regenerated, not carried).

**The reverse ring is simpler fused than standalone.** `d[p] = g_out[p] + gamma[p+1]*d[p+1]`
with `g_out[p] = dS_recv[p+1]`, so the message `p+1` sends — `dS_recv[p+1] + gamma[p+1]*d[p+1]`
— **is** `d[p]`: the receiver adds nothing, and no host-side `g_out` shuffle is needed. And
`out_prev[p] == S_recv[p]`, already device-local from phase 2, so `dgamma` reduces locally.
Both inputs `allscan/…/program_backward.py` takes from the host disappear.

**Three DSL restructurings**, all forced by shapes and validated at torch level first
(`../devtools/b4_math_check.py`, 11/11 across `P∈{1,2,3,4}` × `dk≠dv`):

| problem | what the kernels do instead |
|---|---|
| `gamma` is `[dk,1]`, so `k*(gamma/b)` is a *column* broadcast over `[C,dk]` — inexpressible | push gamma onto the `[dk,dv]` state once: `dV_h = (k/b) @ (gamma*dSloc)`, `dK_h = (v @ (gamma*dSloc)^T)/b`. One `row_expand_mul` replaces three column broadcasts per chunk |
| `db[C-1] += dgamma` is a single-row update | apply it as a whole-tile `col_expand_add` **after** the reverse cumsum — row `C-1` is `>= t` for every `t`, so it adds the same constant to every row |
| no scan primitive | the reverse cumsum is `triu @ dg_cs`, a matmul |

The gate gradient is carried in the **log domain** (`dg_cs = db*b`), so `db` never exists and
the `b` factors cancel.

#### What is verified

| | |
|---|---|
| math decomposition vs `expected_gla_backward` (torch, `b4_math_check.py`) | **11/11**, `P∈{1,2,3,4}`, `dk≠dv` |
| the 5 DSL ops the forward never used (`b4_op_probe.py`, sim + HW) | **pass** |
| all three kernels vs their torch emulation, **with non-zero** `S_recv`/`dS_total`/`dgamma` (`b4_kernel_probe.py`, sim) | **13/13 outputs**, ≤2.6e-07 |
| P=1 end-to-end on a2a3 — 6 shapes + back-to-back dispatch guard | **7 passed / 0 failed**, worst rel err 2.4e-07 |
| `loss.backward()` through `ZeCoModule` on the pypto kernels (sim) | **pass** |

#### What is blocked: every P>1 config fails on device

`test_pypto_zeco_backward[2]` and `[4]` fail with a device-side stall
(`ACL_ERROR_RT_AICPU_EXCEPTION`, then `sched_error_code=100 SCHEDULER_TIMEOUT,
sub_class=S1:running-stalled`): a ring wait never returns.

**The cause is not known.** An earlier version of this section asserted that a cross-rank
`remote_store` + `notify` was silently discarded. **That was wrong, and it was wrong because
the instrument was wrong.** Every probe behind it (`b4_readsig_probe.py`, the L0–L6 shrink
ladder, `b4_smallest_repro.py`, and the winbuf / ctxread / dispatchpos / min variants) was
built by replacing `pld.system.wait` with a plain load of the signal word, so a failing run
would complete instead of hang. But `wait` is the *only* construct that orders the two ranks:
without it, rank 1 simply reads before rank 0's send has landed. The compute dispatches those
probes inserted "to give the send a wide margin" were a timing assumption, never a guarantee.

The disproof is self-controlled, so the layout sensitivity that made every cross-program
comparison untrustworthy cancels exactly. Ladder rung L0 verbatim, with the receiver reading
the same window **three** times, compute chains between:

| read | payload | signal |
|---|---|---|
| read 1 (where L0 read) | 6/36 | 6/36 |
| read 2 | **36/36** | **36/36** |
| read 3 | **36/36** | **36/36** |

`arrived-late 30/36, never-arrived 0/36`. **Nothing is ever lost.** Payload and signal always
arrive and always together — the "they vanish together" observation was a correct correlation
with an incorrect explanation: neither had arrived *yet*.

This retro-explains the whole false trail: the layout sensitivity (a two-rank timing race, so
any dispatch added or removed shifts relative timing), the non-monotonic shrink ladder,
"send as its rank's first dispatch delivers 12/12" (sending earlier beats the read), and
allscan passing at `P=4` (it has real waits).

**Proven correct, by measurement:** the send kernel, the `CommRemoteOffset` peer address
translation, the peer store path, and cross-rank delivery. Also ruled out earlier and still
ruled out: the compute kernels; the `host_orch` wiring; worker/comm lifecycle; submission
order / ring direction; "two comm windows per program"
(`tests/st/distributed/test_l3_ep_dispatch_combine.py` allocates eight and passes); window
buffer alignment; a stale `CommContext` read; and the cards.

Two framework defects were investigated during this search and **both are dead ends**:

* the peer `pto.tstore` → `pto.cmo.cacheinvalid` pair has no barrier between an async MTE3
  DMA and a scalar-pipe `dcci`. Adding one changes nothing (L0, interleaved S/F/F/S blocks:
  **STOCK 30/30 vs FIXED 29/30** instantiations lost). It is also **already fixed upstream** —
  pypto PR #2168 relanded `InsertCommFence` and ptoas 0.54 lowers
  `pto.fence.barrier_all <gm>` to a combined `pipe_barrier(PIPE_ALL); dsb(DSB_DDR)`, so
  `dcci → barrier → dsb` is the *intended* order (see `allscan/issues/UPSTREAM-ISSUES.md` #3
  and the archived `pypto-remote-store-notify-drain/` folder, closed 12/32 → 0/128);
* that same `dcci` covers one 64-byte line of a multi-line payload. Not load-bearing: the
  closed repro above pushed 128×128 f32 = **1024 cache lines** through it and measured 0/32
  wrong. MTE3 GM writes reach DDR without it — the sender's own local store gets no `dcci` at
  all and is correct in every run.

**Where to look next.** Delivery is reliable, so the wait is not waiting on data that never
came — it is waiting on a count that never completes. Instrument the *real* fused backward
with its waits in place and compare, per rank and per ring, the notifies actually sent against
the value each `pld.system.wait` expects; a slot-indexing error or interference between the
forward and reverse rings' signals shows up immediately as a count that stalls one short or
overshoots. `../devtools/b4_ring_diag.py` and `../devtools/b4_bisect.py` already operate on the
real program and are the right starting points; `b4_stub_program.py` / `b4_stub_fwd.py` stub
one phase at a time.

**Method note, paid for twice in one day.** Before trusting any distributed measurement, audit
what the program actually *guarantees* — a probe that deletes the synchronisation primitive
cannot demonstrate a delivery bug. And before investigating any comm-path defect, grep
`allscan/issues/UPSTREAM-ISSUES.md` and the `.archive/issues-resolved-*.tar.gz` tarballs:
several are already filed, fixed, or explicitly dropped as stale-build artifacts.

Nothing is filed or pushed, and as of 2026-08-14 there is **nothing to file**: no framework
defect survived measurement, and the reproducer that appeared to demonstrate one was invalid.

#### Measured shape ceiling

A Vec overflow is a *compile* failure, so `../devtools/b4_shape_probe.py` maps this with no
NPU at all:

| shape | Vec bytes vs 188416 limit |
|---|---|
| `C=16`, `D ≤ 64` · `C=32`, `D ≤ 32` · `C=32` with one dim 64 | fits |
| `C=32, dk=dv=64` | 189440 — **over by 1 KB (0.5%)** |
| `C=64, D=32…64` | 209k–247k |
| `C=D=128` | 985k |

`grad_o` is the widest kernel in either direction. Moving `dc_prev` from `grad_o` into
`grad_h` was tried and **reverted**: it moved the peak to `grad_h`, reached *exactly the same
shape set*, and pushed the near-miss from 1 KB to 17 KB. So `C=32/D=32` is the largest shape
both directions reach, which is where a forward-vs-backward comparison has to be run.

Two DSL restrictions found and written up in the probes: `pl.tile.full([shape], dtype, value)`
is **not** callable from the DSL (the parser binds it to the scalar overload), and a tuple
assignment inside an InCore kernel (`a, b = X, Y`) does not compile — one name per line. A
third is upstream-relevant: an Orchestration function whose result is **unpacked** at host
level must declare a `pl.Tuple[...]` return type, or distributed codegen fails with an
internal error (`TupleGetItemExpr unpacking found no Out parameter`) rather than a diagnostic.

### 5. F3.1b — real tile blocking for pypto
`C=D=64` is at 96% of the vector budget; every next step overflows 3–4×, and `C/D = 128` also
blows the 64 KB cube `Left`/`Right` buffers. The live set is dominated by 5 `[DK,DV]` state
tiles and 5–6 `[C,DK]` row tiles, so **DV blocking alone is not enough — DK blocking is
required**. Both `o_inter = qt @ S` and `scores = qt @ (K/b)^T` contract over `DK`, so partials
must be summed **in the vector unit** — fp32 cube K-accumulation is broken on a2a3 (see
parked). That is the same design F3.4 landed on for simpler, so one design serves both.

### 6. The final fair numbers — F6.5, F6.6, B5.4
- **F6.5** — realistic sizes (`C,D` 64/128, `L` up to 1–4k). Needs task 5.
- **F6.6** — work-placement parity. simpler runs `_S_total`, `_shift_snaps` and `_gammas` on
  the **host**; pypto does all of it on device. Measured ~5.5 ms/call of host work against
  7.8 ms of warm kernel time — comparable. **Until this is closed, do not publish any
  compute-vs-comm or kernel-vs-kernel split**, including F6.4's "simpler compute = 6.0 ms",
  which counted device dispatches only. Close it by porting the glue on-device (real parity,
  touches the same kernels as task 5 — sequence it after) or by explicitly redefining
  simpler's "compute" as device + host and never quoting the device-only number.
- **B5.4** — the same treatment for the backward, once B4 lands.

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
