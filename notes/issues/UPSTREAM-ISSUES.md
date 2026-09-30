# Upstream issues ready to file — re-validated on a CLEAN rebuild of PR #2135

## FILED 2026-08-17 — pypto comm-dispatch ordering

| what | where |
|---|---|
| issue | **hw-native-sys/pypto#2397** — <https://github.com/hw-native-sys/pypto/issues/2397> |
| PR | **hw-native-sys/pypto#2398** — <https://github.com/hw-native-sys/pypto/pull/2398> |
| folder | `pypto-comm-dispatch-ordering/` (issue + MR drafts, `repro.py`, `fix.patch`, validation) |
| branch | `vloncar/pypto:fix/comm-dispatch-ordering`, rebased onto `main` `b10cae3d` |

Distributed codegen derives dispatch dependencies only from tensor tags, so comm windows carry
no edge and a fan-in-free `wait` dispatch is scheduled ahead of its own rank's send — a
deterministic `P>1` deadlock. Fix threads a per-rank comm ordering token as `INOUT`. **The fix
is carried locally in `/opt/pypto` until this merges**; a stock rebuild reintroduces the
deadlock, and it spans both a C++ codegen change and `distributed_runner.py`, so the two halves
must be reverted or re-applied together.

Note pypto upstream is **GitHub** (`hw-native-sys/pypto`, fork `vloncar/pypto`, default `main`),
not GitCode — the `gc` workflow used for pto-isa does not apply; use `gh`. A `git worktree` at
`/tmp/pypto-pr` holds the PR branch so `/opt/pypto`'s patched, built tree stays untouched; keep
it while the PR is open in case review asks for changes.

---


**All results below were re-confirmed on a clean-from-scratch build** of stock pypto
`main` (`31251f74`, PR #2135) — `rm -rf build` + full `pip install --force-reinstall`
(no local patch) — with ptoas **0.52**, on a2a3 (910B2). This matters: an earlier
pass validated on a *stale* build (only the `.so` had been swapped, build dir not
wiped) and produced one false positive and one understated result. Clean-build truth:

| # | Title | Layer | Folder | Repro | Clean-build result |
|---|-------|-------|--------|-------|--------------------|
| 1 | Cross-branch phi of a tensor ref → undefined name in generated `host_orch.py` (`NameError` at `prepare()`) — **FIXED locally**, see below | pypto host codegen | `pypto-crossbranch-phi-nameerror/` | `python repro.py` | `ir.compile` OK → `NameError: name 'zero__ssa_v0' is not defined` |
| 2 | `TCOLEXPAND` tagged `PIPE_V` but lowers to a copy engine — under-synchronized | pto-isa | `tcolexpand-pipe-sync/` | static (`event.hpp:269` vs `a2a3/TColExpand.hpp:28`) + `PIPE_V` toggle → `test_chunk_h` | C=32 FAIL (s_snap 0.036–0.71), C=128 PASS |
| 3 | `remote_store` → `notify` has no producer drain; signal overtakes payload (stale read) | pypto codegen / PTOAS #872 | `pypto-remote-store-notify-drain/` | `python repro.py` | **12/32** dispatches WRONG (correct = 0/32) |
| 4 | ptoas 0.52: `pto.fence.barrier_all <gm>` inside a comm InCore kernel deadlocks | ptoas 0.52 | `ptoas-fence-barrier-all-deadlock/` | `python repro_comm.py` | PTO2 scheduler timeout, S1:running-stalled, completed=0/1 |

## Against latest `main` (`00cacc25`) + ptoas 0.48 (what main pins)
Full "checkout main + install 0.48 + rebuild + run" was **not** completed: ptoas 0.48
ships as a wheel with a renamed package (`ptodsl`) and no standalone `ptoas` CLI (the
executable is constructed by the pypto-tooling image's bootstrap, not shipped), and
main pins runtime `9922afdb` ≠ the installed `8cdb306c` (mismatch blocks HW execution).
Instead: the 8 new commits on `main` were diffed, and the fence lowering was tested on
the ptoas **0.50** backup directly.

| # | On `main` + ptoas 0.48? | Basis |
|---|-------------------------|-------|
| 1 phi `NameError` | **present** (unchanged) | main's new commits touch only docs in orchestration codegen; the host-codegen path is untouched. Not rebuilt-and-run on main. |
| 2 drain race | **present** | `pto_ops_distributed.cpp` unchanged on main; the race is a HW weak-ordering + codegen-gap issue, independent of ptoas version. |
| 3 fence deadlock | **NOT present** (0.51/0.52 regression) | **Empirical:** ran ptoas 0.50 on the same fence `.pto` → `dsb(DSB_DDR)` only, no split barriers (vs 0.52's split `pipe_barrier(MTE2/MTE3/FIX)`). 0.48 predates 0.50 → dsb-only, no deadlock. |
| 4 TCOLEXPAND | **present** | pto-isa + simpler runtime; never touches ptoas, so the ptoas version cannot change it. |

**So #3 is confirmed a ptoas 0.51/0.52 barrier regression** (matches the reported 0.52
barrier issues); #1/#2/#4 hold on main + 0.48.

## Simulator (a2a3sim) vs hardware (a2a3)
Tested each on the CPU simulator. **Only the host-codegen bug (#1) reproduces on sim;
the three device-level bugs (#2/#3/#4) are hardware-only** — the sim's functional model
does not capture async pipe / weak-memory-ordering / FFTS-event behavior, so it gives
**false confidence** for them.

| # | bug | a2a3 (HW) | a2a3sim |
|---|-----|-----------|---------|
| 1 | phi `NameError` at prepare | reproduces | **reproduces** (host codegen) |
| 2 | remote_store→notify drain race | 12/32 wrong | 0/32 — **does NOT reproduce** |
| 3 | `pto.fence.barrier_all` deadlock | timeout/hang | runs clean — **does NOT reproduce** |
| 4 | TCOLEXPAND PIPE_V mis-tag | C=32 FAIL | C=32 PASS — **does NOT reproduce** |

## DROPPED — was a stale-build artifact, NOT a real bug
- **loop-alloc window aliasing** — on a clean build the minimal repro returns rings
  **disjoint (correct)**; it only "aliased" on the stale build. Folder + repro removed
  2026-07-28. The `batched_program.py` splicer workaround is kept (the real-AllScan
  loop-alloc still misbehaves build-sensitively), but there is no clean minimal repro,
  so it is not fileable — revisit only with a clean-build repro that survives.

## STATUS 2026-07-30 — environment moved to pypto `main` + ptoas **0.54**; 3 of 4 issues fixed upstream
Env now: pypto `main` (`f621eca4`), ptoas **v0.54** (sha256 `011e980d…`, matches the pin in
`toolchain/versions.env`), runtime **`9922afdb`**, pto-isa `83d01313` (= runtime's
`pto_isa.pin`). **No local pypto patches** — the tree is stock.

| # | Issue | Status on main + 0.54 | Evidence |
|---|-------|-----------------------|----------|
| 1 | phi `NameError` | **FIXED upstream** (PR #2183, merged `555bd4d1`) | our repro passes; local patch dropped |
| 2 | `TCOLEXPAND` `PIPE_V` mis-tag | **STILL OPEN** (pto-isa; being discussed on gitcode) | `PIPE_ALL` workaround still required |
| 3 | `remote_store`→`notify` drain race | **FIXED upstream** (PR #2168 relanded `InsertCommFence`) | **4 runs × 32 = 128 dispatches, 0 wrong** (was 12/32 on 0.52); markers verified in `.pto` + `.cpp`; race guard green. **Our local drain patch is obsolete and removed.** |
| 4 | `pto.fence.barrier_all <gm>` deadlock | **FIXED in ptoas 0.54** | 0.54 lowers it to `pipe_barrier(PIPE_ALL); dsb(DSB_DDR)` (0.52 used split `MTE2/MTE3/FIX`); repro now runs clean |

**Why #3 and #4 both closed.** ptoas 0.54 changed the fence lowering to a *combined*
`pipe_barrier(PIPE_ALL)` — exactly the fix suggested in issue #4 — which removes the
`pipe_barrier(PIPE_MTE3)` that corrupted the FFTS event slot. That in turn makes the relanded
`InsertCommFence` pass usable, and it emits a **stronger** drain than our local patch did:
```
PTOAS__DCCI_SINGLE_CACHE_LINE(v32);   // cache release
pipe_barrier(PIPE_ALL);               // combined barrier (0.54)
dsb(DSB_DDR);                         // DDR fence
...  pto::comm::TNOTIFY(...)          // signal only after the drain
```
vs our patch's bare `pto.barrier <PIPE_ALL>`. So the data-before-signal contract is now
implemented upstream end-to-end.

**Validated on a2a3 (stock main, no patches):** AllScan pypto forward K=1,2,4 + 32-dispatch
race guard **4/4**; pypto backward **3/3**; simpler AllScan forward **4/4** / backward **4/4**;
`chunk_h` **8/8** (no runtime-API drift from the `8cdb306c`→`9922afdb` bump).

**Note for PR #2168:** its checklist leaves `dist-system-tests` unrun ("needs a 2-device host")
and lists the pass's benefit as *inferred, not observed*. Our results above are that missing
evidence — the race guard and the 12/32→0/32 delta demonstrate what the pass fixes on device.

## NEW 2026-07-31 — issue #5: InCore loop-carried tile corrupted on HW (cube↔vector pipe race)
Folder: `pypto-incore-loop-cube-vector-race/` (`ISSUE.md` + standalone `repro.py`, torch+pypto
only). Found by root-causing **F2**, which we had mis-tracked for weeks as a *distributed*
loop-carry miscompile.

| # | Title | Layer | Repro | Result on main + 0.54 |
|---|-------|-------|-------|-----------------------|
| 5 | Bidirectional cube↔vector `TPipe` aliases its two GM rings — silent tile corruption | pto-isa a2a3 `TPush.hpp` + pypto workspace sizing | `python3 repro.py <dev> a2a3 2,4,5,6,7,8,16 3` | N=2/4/6 clean; N=5/7/8/16 corrupted by O(1..8). **FIXED** by the two diffs below |

Three independent facts make it a **race**, not a codegen/value bug:
1. Same compiled binary + same inputs → different results across dispatches (N=5: run0 clean,
   run1/2 corrupted; N=7: run0 `bad=[6]`, run1 `bad=[2..6]`).
2. Generated device code for a passing N=4 and a failing N=8 is **byte-identical** except the
   trip-count constant (`v11 = 4` vs `= 8`); all 47 IR pass dumps likewise.
3. **a2a3sim is clean at every N** — hardware only. (Same false-confidence pattern as #2/#3/#4.)

**ROOT CAUSE: a bidirectional cube↔vector `TPipe` aliases its two rings on a2a3.** Both
directions index the shared GM buffer as `(tileIndex % SLOT_NUM) * SLOT_SIZE + entryOffset`, and
`entryOffset` — which exists exactly to separate them (`Producer/Consumer::setEntryOffset`) — is
**never set**: `grep setEntryOffset` over the emitted AIC/AIV sources returns nothing, and the
`TPipe` ctor does not set it either. The ISA reference says it must be non-zero
(`docs/zh/reference/pto-isa/01-tpush_tpop.md`: `v2c_ring_buf = GM_SLOT_BUFFER + SLOT_NUM *
SLOT_SIZE`). So the cube's matmul results and the vector's operands overwrite each other.

Matching defect on the pypto side: `ComputeGMPipeWorkspaceElements`
(`src/codegen/orchestration/orchestration_analysis.cpp`) sizes a bidirectional pipe as **one**
ring and ignores `slot_num` — which is exactly why the overlapping layout "fit".

**Fix — two diffs, both required** (either alone does nothing; fixing only the allocation leaves
the corruption bit-identical):
- `pto-isa-bidirectional-ring-offset.diff` — set the V2C entry offset in the `TPipe` ctor for
  `DIR_BOTH`.
- `pypto-gm-pipe-workspace-size.diff` — size the workspace for **both** rings and honour an
  explicit `slot_num`.

**Validated on a2a3 at the stock ring depth, no DSL overrides:** standalone repro CLEAN at C=16
N=2..16 ×3 and C=32 N=4..32; fused ZeCO forward **12/12 at C=16** and **12/12 at C=32** (was
5/12 and 6/12 — `C=32/P=2/N=4` went 6.905 → 1.7e-5); no regression (`test_pypto` 4,
`test_pypto_backward` 3, `test_pypto_gla` 3). Full analysis in the folder's `ROOT-CAUSE.md`.

**PRIOR ART (checked 2026-08-03): the device half was already reported — and withdrawn.**
pto-isa **#195** (luohuan19, 2026-07-09 → closed 2026-07-12) states our root cause verbatim
("C2V and V2C share the same physical GM slots", "`entryOffset` is never set") with the same
experiment (35/50 FAIL unfixed, 50/50 PASS with the directions separated), and its downstream
pypto **#1981**. The author closed both himself — *"we found that it was not a problem with
pto-isa"* — with no reasoning and **no fix**: main (`e8f558d5`) still aliases. pto-isa PR #193
fixed this class of bug in the **CPU-SIM path only**, which is why a2a3sim is clean. The
workspace-sizing half has **never** been reported. So we file as *supersedes #195*, leading with
what is new: the spec citation, a minimal repro with no `split_aiv`/`syncall`, the missing pypto
half (which is likely why a device-only fix looked wrong), and HW validation at stock depth.

**FILED 2026-08-03.** pypto issue **#2269** + PR **#2271** (`vloncar:fix/gm-pipe-workspace-two-rings`,
based on `43891652`) — **all CI green**: pre-commit, clang-tidy, toolchain, build, unit-tests
(ubuntu+macos), codegen-tests, system-tests ×3, dist-system-tests, pypto-lib-model. pto-isa issue
**#226** + PR **#227** (`vloncar:fix/a2a3-dir-both-ring-offset`, based on `e8f558d5`) — hooks clean
locally; CI needs a `compile` comment, and the PR still needs the CLA + `/assign` on #226. Comment
posted on the withdrawn #195 linking both.

Lessons banked: **run `pre-commit run --files <changed>` locally before pushing** (CI fails the
whole pipeline if any hook modifies a file — clang-format reflowed two lines I had hand-wrapped at
100 cols against a 120 limit), and **no `Co-Authored-By` / AI-attribution trailers** on upstream
commits. pto-isa's `main` is itself not clang-format-clean (`TPush.hpp` `shouldNotifyFree`);
verified pre-existing against a pristine `e8f558d5` checkout and deliberately excluded from #227.

**Filing: two PRs, one per repo, and pypto must land FIRST.** pypto alone just over-allocates
(harmless); pto-isa alone writes past the end of a workspace an unfixed pypto sized for one ring.
Ordering, compatibility notes and the "why did no test catch it" answers are in
`UPSTREAM-PLAN.md`; ready-to-paste issue + PR + commit text in `PR-pto-isa.md` / `PR-pypto.md`.
Both diffs now carry their tests: pypto updates one existing assertion that encoded the one-ring
assumption (`..._sizes_workspace_and_resolves_callees`, 512 → 1024) and adds two; pto-isa sizes
its own `tpushpop_dir_both` ST buffer for two rings. Also flagged for maintainers: a5
`DIR_BOTH_GM` has the identical pattern (`a5/TPush.hpp:290/319/399/655`, empty ctor at `:704`),
untouched here for lack of A5 hardware.

A ring-depth detour (`pypto-cross-core-slot-depth.diff`, default 4 → 8) *masked* the bug at 1 KiB
slots but never at 4 KiB slots; it is **reverted**, and kept only as the evidence trail that
pointed at the ring layout. Worth raising separately: the docs justify the 4:8 default with a
per-slot flag budget (8 flags → 8 slots unidirectional, 4+4 bidirectional), but the shipping
implementation uses a fixed 4 event ids for `DIR_BOTH` regardless of depth — the rule is a
leftover from a design the code moved away from.

## Upstream PR status
- **#1 (phi `NameError`)** → pypto **PR #2183** (georgebisbas, open). Validated on our HW:
  repro passes, `test_pypto_allscan` K=1,2,4 green. Two open review findings (Codex P1
  unconditional-submit during phi-init probing; CodeRabbit's single shared `tensor_phi_init`)
  don't affect our programs. Our alternative patch is below.
- **#2 (TCOLEXPAND pipe)** → tracking moved **upstream of GitHub**. GitHub pto-isa PR #212
  (`PIPE_MTE1`) is **not expected to be merged or tracked**; the live change is
  **gitcode `cann/pto-isa` PR !1406** (<https://gitcode.com/cann/pto-isa/pull/1406>), which
  will trickle down to GitHub pto-isa and only then become a dependency for us.
  **Nothing to do but wait.** Our measurement stands and matters if `PIPE_MTE1` is what
  lands there: **it faults the AICore (507018) on a2a3, failing 8/8 cases including ones
  that pass today** — `PIPE_ALL` is the only value that works (full pipe matrix in
  `tcolexpand-pipe-sync/ISSUE.md`). Worth surfacing into !1406 if the opportunity arises.
  Our `pipe_barrier(PIPE_ALL)` workaround in `chunk_h_prep.cpp` stays until then.

## #1 — FIXED locally (patch ready for upstream)
Root-caused in the HOST-orchestrator emitter and fixed in
`distributed_codegen.{cpp,h}`: the phi merge is now materialized as a
`tensors["phi"] = tensors["value"]` dict copy in each branch (new
`VisitStmt_(YieldStmtPtr)` + `current_return_vars_`), and a plain tensor-to-tensor
alias routes through the `tensors` dict instead of bare Python locals. Patch:
`pypto-crossbranch-phi-nameerror/pypto-host-orch-phi-materialize.patch`. Validated on
a clean rebuild (PR #2135 + ptoas 0.52, a2a3, P=2): repro passes with correct values,
`test_pypto_allscan` K=1,2,4 still green. Unblocks removing the in-branch stage2
workaround in the fully-fused ZeCO forward.

## #3 and #4 are two halves of one problem
`remote_store→notify` needs a producer drain (#3), and the natural DSL way to add one
(`pl.system.fence()`) deadlocks on 0.52 (#4). Our fix is a codegen `pto.barrier
<PIPE_ALL>` drain in `MakeRemoteStoreCodegenPTO` (parity with put/get) —
`ptoas050-commremoteoffset-inline/ptoas052-remote-store-drain.patch`.

## How to reproduce cleanly
- Env: `source <cann>/set_env.sh`; `LD_PRELOAD=<cann>/lib64/libhccl.so`;
  `PTOAS_ROOT=/opt/ptoas-bin` (0.52); `PTO_ISA_ROOT=/opt/pto-isa`;
  `PYTHONPATH=/root/workspace/allscan/pto-zeco`.
- **Build hygiene (learned the hard way):** to test on stock, `rm -rf /opt/pypto/build`
  then `pip install --no-build-isolation --no-deps --force-reinstall .` — a `.so`-only
  swap over a stale build dir gives unreliable results.
- Delete stale `/tmp/barrier_pto_multi_comm_*` and kill leftover `repro.py` chip
  workers between runs; use a fresh device set; #4 deadlocks and force-resets its
  devices.
