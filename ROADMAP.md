# ZeCO Roadmap

Status snapshot: **2026-07-27**. ZeCO = sequence-parallel single-head Gated Linear
Attention (GLA) built on the AllScan collective. Backends: `torch` (reference),
`torch.distributed` (reference SP decomposition), `simpler` (hand-written PTO-ISA
kernels), `pypto` (fully-fused distributed `@pl.program`).

**Update 2026-07-27 (ptoas 0.50 + PR #2076 REVERTED upstream; env moved to the 0.52 base):**
- **Toolchain churn.** Upstream **reverted ptoas 0.50 and PR #2076 (InsertCommFence)** —
  PR #2138 (merged 07-25), "buggy ptoas 0.50 dependency." `main` is back on **v0.48**. The
  forward path to a working assembler is **PR #2135** (`fix-potas`): post-revert main + **ptoas
  v0.52** + a ptoas-discovery fix (v0.51 changed the tarball layout — `<root>/ptoas` is now a
  Python package dir, `<root>/bin/ptoas` the exe). PR #2135's HW CI is green.
- **Env now on PR #2135** (`31251f74`, ptoas **0.52**, runtime `8cdb306c`, **no InsertCommFence**),
  fully rebuilt (`pypto_core` + `pip install`). ptoas 0.52 installed in-container from the
  release tarball (sha `a64e7ee9`); rollback backups kept.
- **F1 is FIXED on the 0.52 base (2026-07-27).** With InsertCommFence gone, the `remote_store →
  notify` path had no producer drain (`tstore → tnotify`) and the **32-dispatch race guard
  FAILED** (correctness). Fix = a trailing **combined `pto.barrier <PIPE_ALL>`** in
  `MakeRemoteStoreCodegenPTO` (parity with the `put`/`get` codegen, which already drain) — one
  codegen change covering **fwd + bwd + gla** (all use `remote_store`), no DSL/program edits.
  Validated a2a3 P=4: **race guard 0/32**, forward K=1/2/4 **4 passed**, backward K=1/2/4
  **3 passed**. — The DSL route (`pl.system.fence()`) was tried first and **rejected**: on 0.52
  it lowers to *split* `pipe_barrier(MTE2/MTE3/FIX)+dsb` which **deadlocks** this comm InCore
  kernel (`SCHEDULER_TIMEOUT`, all K), and `pl.system.bar_all` has no PTO codegen — so the DSL
  cannot express the safe combined drain. Writeup + patch:
  `allscan/issues/ptoas050-commremoteoffset-inline/ptoas052-remote-store-drain.{md,patch}`.
  Proof the underlying base is sound: pypto's **own** distributed HW tests pass under 0.52
  (`test_l3_notify_wait`, `test_l3_remote_store` — 3/3). The earlier "0.52 AIV `507015` fault"
  was our stale `efb78378` running the reverted-buggy InsertCommFence against 0.52 — a local
  artifact. **Carried locally** (the `.so`; Python tree unchanged) pending an upstream drain on
  the `remote_store→notify` path.
- **simpler backend** — unchanged from the 07-24 re-port (runtime `8cdb306c`: allocate
  `share_memory_()` before the eager `Worker.init()` fork `#1397`; `run_multi` via
  `build(multi_h=H)`). pr2135 pins the same runtime, so it stays green: AllScan fwd **4/4** +
  bwd **4/4**, GLA multi-head PASS. `a2a3sim` init segfault (`#1396/#1397`) still deferred.
- **F2** — still the top blocker; **now needs re-verification on the pr2135/0.52 base** (post-revert
  codegen differs; the N-threshold may move again). Untested there yet.

## Where we are

| Piece | torch | torch-dist | simpler | pypto |
|---|---|---|---|---|
| **Forward** (GLA compute + AllScan boundary) | ✅ | ✅ | ✅ HW P=1/2/4 | ⚠️ HW P=1 to N=16; **P=2 only to N=4** (F2) |
| **AllScan-collective backward** (building block) | ✅ | ✅ | ✅ HW | ✅ HW |
| **ZeCO/GLA operator backward** (dQ,dK,dV,dA) | ✅ | ✅ | ✅ HW P=1/2 | ✗ |

**Important caveat on "Forward":** the pypto forward still does **not** scale in `L`.
After the PTOAS 0.50 update (2026-07-21) the N-threshold moved up but the distributed
loop-carry miscompilation (**F2**) survives: **P=1 is now correct through N=16**
(previously broke at N≥8), **P=2 through N=4** (previously N≥4) but **wrong at N≥8**
(`max_diff` 8.4 at N=8, 11.6 at N=16 — structural, not fp). So the forward is **not**
general-purpose and cannot be **fairly benchmarked** yet. The ring race (**F1**) is **FIXED
again on the 0.52 base (2026-07-27)** after PR #2076's reverted fence pass: a trailing combined
`pto.barrier <PIPE_ALL>` drain in the `remote_store` codegen (parity with put/get) restores
0/32 on the race guard, covering fwd+bwd+gla (see the 07-27 update). **F2 remains the sole
forward correctness gate.** The forward roadmap below is dependency-ordered around these.

---

# Section 1 — Forward pass: toward general-purpose, scalable, fairly-benchmarkable

The forward *math* is done in all four backends. What remains is turning the toy-size
proof into a robust operator. Dependency order: **F2 is now the correctness gate**
(N>2 must work before anything scales), F3 gates larger tiles, F4–F5 gate fair
measurement, F6 delivers the numbers. F1 (the race) is **fixed**, bar upstreaming.

### F1 — pypto ring correctness race  *(FIXED 2026-07-27 on the ptoas-0.52 base: combined `pto.barrier <PIPE_ALL>` drain in the remote_store codegen; race guard 0/32. DSL `pl.system.fence()` was tried and rejected — deadlocks on 0.52. See the 07-27 update at top + `issues/ptoas050-commremoteoffset-inline/ptoas052-remote-store-drain.md`)*
The cross-rank producer race (`pypto-allscan-race-fix` / `pypto-allscan-race-rootcause`) is
**reproduced and matched to known upstream issues**. Reproduced deterministically: back-to-back
batched AllScan at 128² fails **262/640 (41%)** rings with **no** mitigation. It is
**PTOAS #744** (`remote_store`/`pto.tstore` → `TNOTIFY` cross-rank DDR-observability, CLOSED)
— sibling of **#872** (same for `pto.comm.tput`, OPEN). The `emitTNotifyMteDrain` pipe-barrier
covers the *pipe-drain* half but not the *DDR-observability* half; `dsb(DSB_DDR)` before the
signal store is the missing piece.
- **The real upstream fix is PTOAS PR 873** ("Fixes #872", OPEN): a `pto-memory-consistency`
  pass + explicit semantic ops (`pto.fence.release<ddr>`, `pto.cmo.clean/invalidate`); PyPTO
  emits `pto.tstore/tput → pto.fence.release → pto.comm.tnotify` and PTOAS auto-inserts the
  MTE3 drain → `pipe_barrier(MTE3); dsb; TNOTIFY`. Track it; adopt when merged.
- **Interim stopgap (applied, validated):** `pipe_barrier(PIPE_ALL); dsb(DSB_DDR)` at the start
  of `TNOTIFY_IMPL` (`/opt/pto-isa/.../a2a3/TNotify.hpp`) — exactly the fix the #872 reporter
  recommends and validated 20/20 on device. Our HW: **0/640** with it and no fence. (Trim the
  earlier `dcci(ENTIRE_DATA_CACHE)` — unneeded for the MTE3 store path; #872 and PR 873 both
  use only pipe_barrier+dsb there. Correction: the `pld.system.fence()` = `pto.barrier PIPE_ALL`
  = pipe_barrier **only, no dsb**, so it's NOT guaranteed DDR-observable — #744's ~2.6% residual;
  0/640 at 128² was config-luck. The stopgap's dsb is the real fix.)
- [x] **DONE 2026-07-21 — adopted at the framework level on PTOAS 0.50.** Both workarounds are
      **removed**: the `/opt/pto-isa` `TNotify.hpp` stopgap (wiped by the env refresh, left
      stock) and the DSL `pld.system.fence()` calls (deleted from all three programs). The
      ordering is now **compiler-emitted** by a local pypto patch
      (`allscan/issues/ptoas050-commremoteoffset-inline/pypto-ptoas050-adoption.patch`):
      `MakeNotifyCodegenPTO` emits `pto.barrier <PIPE_ALL>` **+** `pto.fence.barrier_all <gm>`
      before every `tnotify`, giving exactly `TSTORE → pipe_barrier(PIPE_ALL) → dsb(DSB_DDR) →
      TNOTIFY`. **Key finding: PR 873's fence alone is NOT sufficient** — PTOAS 0.50 lowers
      `pto.fence.barrier_all <gm>` to the `dsb` *only*, so nothing drains the MTE3 store pipe;
      pypto emitted that drain for TPUT but not for the `remote_store → notify` path. Verified
      on HW: AllScan K=1/2/4 P=2 3/3, back-to-back 32-dispatch race guard pass, backward 3/3.
- [ ] **Upstream the patch** (see the issue dir). pypto still pins `PTOAS_VERSION=v0.48`, so
      the 0.50 adoption is not yet upstream; PR #2076 (`InsertCommFence`) is the related draft.
      Optionally contribute our 128² 41%→0% datapoint to #744/#872.
- [x] Back-to-back stress test added + HW-validated: `test_pypto.py::test_pypto_allscan_back_to_back`
      dispatches 32 fresh-input AllScans at 128² P=2 and asserts every one correct — **32/32 pass**
      on HW with the stopgap (the race regressed to 41% without it).

### F2 — The N>2 chunk failure  *(ROOT-CAUSED AND FIXED 2026-07-31; upstream PRs pending)*
The fused forward was wrong for **N = L//C > 2** because, on a2a3, the two directions of a
**bidirectional cube↔vector `TPipe` index the same GM slots**. `pto-isa`'s a2a3 `TPush.hpp`
addresses every ring entry as `(tileIndex % SLOT_NUM) * SLOT_SIZE + entryOffset`, and
`entryOffset` — which exists precisely to separate the two directions — is **never set**: the ISA
reference's `v2c_ring_buf = GM_SLOT_BUFFER + SLOT_NUM * SLOT_SIZE` is not applied anywhere. So the
cube's matmul results and the vector's operands overwrote each other; `gla_stage1`'s loop-carried
tile was just the visible victim. Writeup: `issues/pypto-incore-loop-cube-vector-race/`
(`ROOT-CAUSE.md`, `ISSUE.md`, standalone `repro.py`, two diffs).
- **Isolated to one InCore kernel on ONE device** — no distribution, no ring, no stage2. A stage
  split of the fused program (`scratchpad/f2_diag.py` + `f2_diag_prog.py`) showed the AllScan ring
  bitwise exact and stage2 correct given its actual boundary; only stage1's own output was wrong.
- **Established as a race, three ways:** the same binary + same inputs gave different results
  across dispatches; the device code for a passing N=4 and a failing N=8 was byte-identical except
  the trip-count constant; and `a2a3sim` was clean at every N.
- **The ring-depth sweep is what pointed at the layout:** corruption periodicity tracked
  `slot_num` exactly, and a deeper ring separated the two aliasing counters — which is why
  `slot_num=8` masked it at 1 KiB slots and never masked it at 4 KiB slots.
- [x] **Fix (two halves, both required, carried locally):**
      `pto-isa-bidirectional-ring-offset.diff` (set the V2C entry offset in the `TPipe` ctor for
      `DIR_BOTH`) + `pypto-gm-pipe-workspace-size.diff` (size the GM workspace for **both** rings
      and honour an explicit `slot_num`; it was sized for one ring, which is exactly why the
      overlapping layout "fit"). Fixing only the allocation changes nothing — measured, bit for bit.
- [x] **HW-validated at the stock ring depth**, no DSL overrides: standalone repro CLEAN at
      N=2..16 ×3 and at C=32 N=4..32; fused forward **12/12 at C=16** *and* **12/12 at C=32**
      (was 5/12 and 6/12). `C=32/P=2/N=4` went 6.905 → 1.7e-5.
- [ ] **File upstream** — pto-isa (the offset) and pypto (the workspace size). Also worth raising:
      the docs describe a per-slot-flag budget (8 flags → 8 slots unidirectional, 4+4
      bidirectional) that the shipping implementation does not use — it takes 4 event ids for
      `DIR_BOTH` regardless of depth — so the 4:8 default is a leftover from a design the code
      moved away from.
- [ ] Correctness sweep `N ∈ {2,4,8,16,32}` × `P ∈ {1,2,4}` × `C,D` vs `expected_gla` before
      declaring forward-at-scale.
- **Unblocks:** F3–F6 and the pypto backward (B4) — all reuse these kernels.

### F3 — Scale the compute kernels to realistic tiles  *(scale gate; simpler Phase 1+2 DONE — square + rectangular C≠D)*
Both backends were single-size only; the fair benchmark needs them at a *shared* size (`C=D=32`):
- **pypto** compiles only up to `C=32` — the InCore kernels materialize whole `[C,C]`,
  `[C,D]`, `[D,D]` tiles (no blocking), so `C=64` overflows the 184 KB vec-buffer limit
  ("Vec buffer usage exceeds platform limit"). `D=64` fits only because `C` drives the
  `[C,C]` tiles.
  - [ ] **F3.1 (pypto tiling):** Block/tile the chunk kernels (stage1, stage2) so `C` and `D`
        scale independently under the vec-buffer budget; keep the loop-carry detach +
        gamma-broadcast tricks.
- **simpler** — the 6 incore kernels had `TILE=128` hard-coded (never ran at any other size;
  the `507018`/nan at `C=32` was OOB from 128-sized tiles on 32-sized buffers).
  - [x] **F3.2 — Phase 1 (square `C==D`, sizes `{16,32,64,128}`) — DONE, HW-validated.** Each incore
        now dispatches a runtime tile-size scalar to a compile-time template (the
        `benchmark_bgemm` pattern); the orchestrations thread `S` in via `add_scalar`. When
        `C==D` every GLA matmul is square `S×S×S`, so one size drives the whole pipeline.
        Two real bugs found + fixed en route: (a) the `intra` matmul was missing its size
        scalar → read garbage → `default:128` (accidentally correct at 128, OOB/nan at 32);
        (b) `chunk_h_prep`'s `TCOLEXPAND` lowers to a ubuf copy not covered by a `PIPE_V`-only
        barrier → strengthened to `PIPE_ALL` (a real pto-isa pipe-tagging gap — see
        `allscan/issues/tcolexpand-pipe-sync/`). `impl.py` asserts `C==dk` + `C∈{16,32,64,128}`.
        Validated on HW (dev6/6,7): SceneTests `chunk_h`/`chunk_o`/`gate_cumsum` at `C=32` and
        `C=128` (no regression) + `chunk_o` at `C=64`; e2e `SimplerZeCo` P=1 (1.5e-5) and P=2
        with the real AllScan boundary (2.3e-5).
  - [x] **F3.3 — Phase 2 (non-square `C != D`, e.g. the bench's `D=64` configs) — DONE, HW-validated.**
        The `matmul` kernel now takes independent `M,N,Kc` (runtime scalars → a 3-level
        `{16,32,64,128}` compile-time dispatch, `mode` picks NN/TN/NT); the transpose recipe is
        the same row-major-load → `TRESHAPE` to the ZN layout (with the *transposed* dims,
        confirmed on HW) → `TEXTRACT` proven in Phase 1. Every GLA matmul maps cleanly: KV
        `[D,D]←[C,·]` (TN, M=D,N=D,Kc=C), inter `[C,D]` (NN, Kc=D), Aqk `[C,C]` (NT, Kc=D),
        intra `[C,D]` (NN, Kc=C) — inter/intra are the *same* NN kernel, just different `Kc`
        scalars, so no op split was needed. `gate_cumsum` (`tril[C,C]@g[C,D]`) generalised the
        same way; the aiv stages (already `template<R,Cc>`) now dispatch two dims (chunk_h/o
        prep `[C,D]`, chunk_o elt `[C,C]` mask vs `[C,D]` add; chunk_h update stays `[D,D]`,
        always square). Orchestrations thread per-task dims via `add_scalar`; `impl.py` now
        asserts only `C,D ∈ {16,32,64,128}` (dropped `C==dk`). Validated on HW (dev6/6,7):
        `test_matmul` 4 rectangular GLA triples + squares (10/10); SceneTests `gate_cumsum`/
        `chunk_h`/`chunk_o` at `C=32,D=64` (N=2,4) + all square regressions; e2e `SimplerZeCo`
        `C=32,D=64,L=256` (N=8): P=1 `2.8e-7`, P=2 real-AllScan `3.3e-7`; P=1 `C=D=128`
        regression `4.6e-7`.
  - [ ] **F3.4 — Phase 3 (tiles > 128, i.e. head dim 256): partially scoped, DEFERRED (2026-07-10).**
        Investigated + found a hard HW blocker; landed nothing (tree stays at the Phase-2
        `<= 128` state). Findings, all HW-checked on a2a3:
    - Matmul **output (M/N) tiling** to 256 works — a blocked kernel that tiles M,N into
          `<= 128` output blocks passes for `KV`-shaped (`k_rest^T@v`, Kc=C) and `intra`-shaped
          (`Aqk_m@v`, Kc=C) matmuls at D=256 (their contraction stays `<= 128`).
    - **fp32 cube K-accumulation is unsupported on a2a3** (`allscan/issues/fp32-cube-k-accumulation/`):
          reducing `Kc > 128` needs `TMATMUL_ACC` across 128-blocks, which is correct for fp16
          (all a2a3 gemm refs) but **wrong for fp32** — plain accumulate drops a block (~half
          wrong), explicit `AccPhase` **hangs the AICore (507018)**. The CPU sim accumulates
          fine, so sim can't catch it. This blocks exactly the two matmuls that contract over
          the head dim: `inter = q_eff@S` and `Aqk = q_eff@k_eff^T` (both Kc=D). `KV`/`intra`/
          `gate_cumsum` keep Kc=C `<= 128`, so are unaffected.
    - **Deferred design** (the robust path when picked up): give the matmul kernel `kStart`+`fullK`
          scalars (compute one `<= 128` K-slice; the strided sub-tile GM load already supports it),
          have chunk_o issue `ceil(D/128)` slice-matmuls into partials and **sum them in the vector
          unit** (`chunk_o_elt` add) — no fp32 cube K-accumulation. Plus the (mechanical) vector-kernel
          D-column blocking (`prep`/`elt` `[C,D]`, `update` `[D,D]`) that D=256 needs regardless.
          Also worth a retry: fp32 `BK=64` ping-pong with `flash_atten`'s deferred sync (the 507018
          may be my inline `M`-drain, not a fundamental limit).
    - Still shares the tile-blocking need with pypto's ceiling below (head dim 256).
- [ ] pypto still needs the tile-blocking above (F3.1) to reach `C>32`.
- [x] **F3.5 —** Larger-shape correctness sweep for **simpler** (`test_simpler_gla.py --sweep`): 8 curated
      shapes (square + rectangular `C≠D`, `C,D∈{16,32,64,128}`, larger `L` → `N` up to 16) all
      HW-pass P=1 (~1e-7), plus P=2 rect `C=64/D=128` with real AllScan. pypto still capped at
      `C=32` (needs its tiling + the F2 fix).
- **Note:** simpler Phase-1 is orthogonal to F2 (simpler has no pypto loop-carry bug); it now
  runs the identical `C=D=32` workload as pypto's ceiling, so the fair benchmark (F6) needs
  only the pypto N>2 fix.

### F4 — Steady-state (fair) benchmark harness  *(DONE for pypto; simpler assessed)*
End-to-end `forward()` was **dominated by per-call worker setup, not compute** (pypto
≈ 9 s/forward, almost entirely `DistributedWorker.prepare()`/`close()` every call). A fair
*operator* comparison measures **steady state**.
- [x] `measure()` / `amortized_timing` API on `ZeCoImpl` (mirrors `AllscanImpl`); `gla/bench.py`
      uses it and reports `build` / `cold` / steady-state split + an `SS` (amortized) flag.
- [x] **pypto** prepares the `DistributedWorker` **once** in `build()` (shared IO buffers reused
      in place); `forward`/`measure` only copy inputs + dispatch. Validated on HW (P=2, N=2):
      **11.5–55.7 ms/forward steady-state vs ~9 s before** (~200–800×), correct (7.6e-6).
- [~] **simpler** left at per-call timing (`amortized_timing=False`). Its per-kernel worker
      cycling is *forced* by the runtime (one-callable-per-worker + device-exclusive) and is
      cheap (compile is session-cached), so it is already near steady-state. Re-measure its
      real per-call overhead once it runs at size (F3); only chase persistence if it dominates.
- [x] **F4.1 — persistent multi-callable compute worker: INVESTIGATED, NEGATIVE (2026-07-22).**
      B5.3 showed the simpler operator is orchestration-setup-bound (P=1 backward ~7.6 s/call,
      P=2 ~71 s/call), so holding ONE L3 worker with all kernels registered is the obvious win.
      It does not work on this runtime, and fails **silently**: an L3 chip child binds to the
      first callable it runs, so a later dispatch of a *different* registered callable returns
      wrong data with no error. Measured on the backward — dQ/dK/dV correct (~4e-7) but
      **dA `max_rel = 1.0`**, because dA is the only output fed by the 5th dispatch (the
      reverse-cumsum's second `gate_cumsum`), the first dispatch that is not first-on-its-worker.
      Registering post-`init()` instead fails loudly (`run failed with code -1`); registering
      all callables pre-`init()` (the supported ordering) is what corrupts silently.
      **Reverted**; `_ComputeRunner` keeps the per-kernel worker and its docstring now records
      the constraint so this is not re-attempted blindly. Full writeup + upstream fix options:
      `allscan/issues/simpler-second-callable-silent-corruption/`. So the F4 `[~]` verdict stands:
      the cycling is genuinely forced, and a real steady-state number needs a runtime change,
      not a client-side one.
- **Only meaningful at scale once F2/F3 let both run correctly** (measures nothing new until then).

### F5 — Operational hardening for shared-box HW benchmarking  *(mostly done)*
Heavy HW iteration degrades the shared devices: leaked `/tmp/barrier_pto_multi_comm_*`
rendezvous files hang the next run at rootinfo and `507018` on one config broke the next.
- [x] Auto-clean `/tmp/barrier_pto_multi_comm_*` before each config **and** on failure
      (`gla/bench.py` `clean_rendezvous`; `--no-clean` to disable).
- [x] `LD_PRELOAD=<cann>/lib64/libhccl.so` checked + warned at startup (`check_preload`).
- [x] A failed config releases its worker and clears its rendezvous in `finally`, so it can't
      poison the next config.
- [x] Per-config device-health guard (`gla/bench.py`): a config that fails with `507018`
      marks its devices suspect and later configs reusing them are **skipped** (that error
      wedges the AICore → follow-on `-1` cascade); prints an `npu-smi` snapshot on failure.
      (AICore% is unreliable on 25.5.1 — reads ~100% idle — so it's not an auto-gate.)
- [ ] Consider isolating each backend in its own process/run to avoid cross-contamination.

### F6 — Produce the fair simpler-vs-pypto forward numbers  *(first honest numbers 2026-08-05)*
- [x] **F6.1 — Bench runs clean at the shared `C=D=32` config, both backends, all 6
      configs correct** (`P∈{2,4}` × `L∈{128,256}` × `D∈{32,64}`, 12/12 `OK`). Unblocked by
      the F2 fix — the `C=32` configs were 6/12 failing before it.
- [x] **F6.2 — simpler amortized (`SS=Y`).** Added `_PersistentComputeRunner` +
      `SimplerZeCo.measure()`: one held multi-callable worker per device instead of the
      `3P+1` per-kernel stand-ups. Verified **bitwise identical** to the per-kernel path
      (`vs plain = 0.000e+00`, P=1/P=2, ×3 repeats) and gated — a mismatch or any error
      falls back to the safe path with `amortized_timing` False, so the bench never
      reports steady-state numbers it did not actually measure. The old silent-corruption
      trap (`allscan/issues/simpler-second-callable-silent-corruption/`) is **gone on
      runtime `9922afdb`**; it was real on `a756969c`. Per-call: 35–43 s → 29–32 s.
- [x] **F6.3 — Phase breakdown: the remaining gap is orchestration, not compute.**
      `scratchpad/f6_phase_breakdown.py`, P=2/L=128/C=D=32, mean of 3:

      | phase | ms | share |
      |---|---|---|
      | stage1 (2 kernel dispatches, workers open) | **13.3** | 0.0% |
      | close compute workers | 373.8 | 1.3% |
      | **as_build (HCCL AllScan worker stand-up)** | **22772.6** | **78.4%** |
      | as_run (boundary collective) | 2411.8 | 8.3% |
      | as_close | 1223.7 | 4.2% |
      | stage2 (worker REOPEN + chunk_o) | 2240.8 | 7.7% |
      | TOTAL | 29035.9 | |

      **compute dispatch 2254 ms vs runtime orchestration 26782 ms (92%).** `stage1`'s
      13.3 ms against pypto's **12.15 ms** for its whole fused forward at the same config
      says the hand-written kernels are competitive; the ~2400× wall-clock gap is one
      HCCL distributed-worker build per call. pypto pays none of it — one
      `DistributedWorker` serves its fused program and is held across calls.
      `stage2` >> `stage1` for the same reason: it reopens the workers the boundary forced
      closed (device-exclusivity), not because `chunk_o` is slow.
- [x] **F6.4 — Decomposed (2026-08-05, `scratchpad/f6_comm_breakdown.py`, P=2/L=128/C=D=32,
      5 repeats each).** Both composites split by repeating the operation on a warm worker:

      | | ms |
      |---|---|
      | gate_cumsum + chunk_h + chunk_o, **warm** | **6.0** (2.0 / 1.9 / 2.0) |
      | boundary collective, **warm** | **135.9** (min 111.8; first run 2456.2) |
      | AllScan worker build | 22740.0 |
      | compute worker `open()` — the stage2 reopen tax | 2793.0 |
      | AllScan first-run warmup | ~2320 |
      | AllScan worker close | 1146.6 |

      `as_run`'s 2.4 s **was** mostly first-run warmup (18x drop once warm), as suspected.
      **But the conclusion changes:** stripping *all* worker lifecycle, simpler still costs
      6.0 + 135.9 = **~142 ms/call vs pypto's 12.15 ms for its whole fused forward**. So:
      - the hand-written **compute kernels are fast** — 6.0 ms, below pypto's entire call;
      - the **collective is not competitive** at this config — 136 ms is ~11x pypto's whole
        forward. The fused single-program design wins on the boundary, not on kernels.
      - **99.5%** of simpler's as-implemented 29 s/call is worker lifecycle, which is a
        runtime-integration artifact rather than a property of either implementation.

      Caveat before quoting the 136 ms as "comm": it bundles host glue (`torch.stack`,
      `_gammas`' per-rank prod over L tokens, shm staging) with dispatch and the HCCL
      exchange. It is an *as-implemented boundary-phase* number, not wire time. It also
      does not obviously contradict [[allscan-amortized-benchmark]] (comm tied at
      P=4/128²) — different shape (32x32 here) and a different harness; reconcile the two
      before drawing a general conclusion about AllScan comm.
- [ ] **F6.5 — Realistic sizes** (`C=64/128`, `D=64/128`, `L` up to 1–4k): still blocked on
      **F3.1** (pypto tile-blocking; pypto caps at `C=32`).
- [ ] **F6.6 — Work-placement parity: simpler runs part of the algorithm on the HOST, pypto
      does not.** Raised 2026-08-05; a gap in this roadmap's own definition of "fair", which
      until now only ever meant *same problem size* (F3), *steady-state timing* (F4/F5) and
      *verified correctness*. Nothing tracked **where the work runs**.

      simpler's forward computes on host, per rank per call:
      - `_S_total` — `exp(g_total)*s_snap[-1] + k_rest^T @ v_last`, i.e. a real matmul plus
        two `exp` broadcasts;
      - `_shift_snaps` — the boundary shift across all chunks, including a `cumprod`;
      - `_gammas` — the decay product over all `L` tokens.

      The backward is further out: its own comments call the cross-chunk grad recurrence,
      the gate arithmetic and the reverse-cumsum "the cheap host linear glue". pypto does
      all of it **on device** inside the fused program.

      Not negligible: stage1 measured **13.3 ms** for two ranks while its warm kernels
      account for only **7.8 ms** (2 x (2.0 + 1.9)) — leaving **~5.5 ms on the host**,
      comparable to the kernel time itself. (Subtraction across two runs; measure directly.)

      **What this does and does not invalidate.** The F6 end-to-end numbers stand — host glue
      runs inside the measured wall-clock for both backends, so 29 s vs 12.15 ms is a valid
      *operator-as-implemented* comparison. What breaks is any **kernel-vs-kernel or
      compute-vs-comm claim**, including the "simpler compute = 6.0 ms" figure in F6.4, which
      counted device dispatches only and dropped the host portion.

      Two ways to close it:
      1. **Port the glue on-device** — extend the simpler kernels so `S_total`, the snapshot
         shift and the gamma products run in-core. Real parity, and it removes host
         round-trips from the critical path. Touches the hand-written kernels, so sequence it
         after **F3.1** (same code).
      2. **Scope the claim** — keep the split, define simpler's "compute" as device + host,
         and never quote the device-only number. Cheap and honest, but leaves the two
         implementations doing structurally different work, so "hand-written vs generated
         kernels" stays unanswerable.

      Only (1) delivers what this section set out to compare. Until one is done, **do not
      publish a compute-vs-comm split** (see also F6.4's caveat on the 136 ms).

### F7 — Generalization (stretch, for "general-purpose")
- [x] **F7.1 — `dk != dv` for simpler** (DONE 2026-07-10, HW-validated). The three GLA dims are now
      C, dk, dv (state `[dk,dv]`, gates/keys `[C,dk]`, values `[C,dv]`); config is `[C,dk,dv,N]`.
      Only `chunk_h_update` needed a kernel change (square→`<dk,dv>` two-dim dispatch); the matmul
      is already `M,N,Kc`-general and prep/elt/gate_cumsum already two-dim, so the rest was
      orchestration dim-threading + host glue. Dropped the `dk==dv` assert. HW dev6/6,7: SceneTests
      `dk32/dv64` + `dk64/dv32`; e2e `SimplerZeCo` P=1 both directions (~2.3e-7) + P=2 real-AllScan.
      (torch/pypto backends were already `dk≠dv`-capable; pypto kernels not yet swept.)
- [x] **F7.2 — Differentiable operator / model harness (DONE 2026-07-20).** `gla.common.ZeCoFunction`
      (a `torch.autograd.Function`) + `ZeCoModule` (`nn.Module`) wrap ANY `ZeCoImpl` backend
      (forward + analytic backward) into an autograd graph, so `O = mod(Q,K,V,A); loss.backward()`
      produces `(dQ,dK,dV,dA)` through the real kernels — a drop-in differentiable GLA op. The
      backward runs under `torch.enable_grad()` (the torch reference builds a local autograd graph
      inside its backward) and returns detached grads (first-order only). Validated: CPU
      `test_zeco_autograd.py` — `torch.autograd.gradcheck` (finite-diff, fp64) on TorchZeCo P=1/2,
      golden-match P∈{1,2,4}×dk≠dv, a gradient-descent-step smoke test; real simpler kernels via
      `test_simpler_gla_backward.py --module` — **P=1 sim 2.6e-7, P=2 HW 128² 6.4e-7**.
- [x] **F7.3 — Multi-head host-loop baseline (DONE 2026-07-20, HW-validated).**
      `gla.common.MultiHeadZeCo` loops a single-head `ZeCoImpl` over the head axis (`[P,H,L,dk/dv]`;
      heads independent — own state + own boundary AllScan), reusing the single-head kernels
      unchanged; wrappable by `ZeCoModule` for a differentiable multi-head op. Correct: CPU
      (forward + differentiable vs per-head golden, P=1 H=3 / P=2 H=4); simpler HW (`--multihead`):
      P=1 H=3 fwd 2.6e-7 / bwd 3.1e-7, P=2 H=2 fwd 2.4e-7 / bwd 4.4e-7.
      **Latency findings:** fixed compile (~56 s P=1 / ~187 s P=2, incl. first AllScan HCCL setup) is
      one-time, amortized across heads; per-head forward increment ~2.5 s at P=1 (pure compute-worker
      cycling), **~17.3 s at P=2** — the gap is the per-head boundary AllScan (host-loop rebuilds it
      every head). So multi-head at P>1 scales ~linearly at the per-head AllScan cost.
- [x] **F7.4 — Multi-head boundary AllScan amortization (DONE 2026-07-20, HW-validated).**
      `SimplerZeCo.forward_multihead` / `backward_multihead` restructure the head loop into phases —
      all heads' stage-1 compute → **ONE** AllScan worker built + run once per head → all heads'
      stage-2 (backward: two such boundary phases) — so the HCCL distributed-worker *build* is paid
      once per boundary phase instead of once per head (device-exclusivity forbids keeping the
      AllScan alive across the compute, so cross-call persistence is impossible; per-phase batching
      is the achievable amortization). `MultiHeadZeCo` auto-dispatches to it. NO new device kernels;
      boundary factored into `_make_allscan`/`_boundary_on`/`_boundary_backward_on`; single-head path
      unchanged (regression 2.3e-7). **Result (P=2 128², within-job increment): batched 11.85 s/head
      vs host-loop 17.3 s/head — the per-head AllScan *build* (~5.5 s, ~32%) is amortized away;
      correctness preserved (fwd 6.1e-7, bwd 4.4e-7).** Backward benefits ~2× (two boundary phases).
- [x] **F7.5a — Comm-domain-batched multi-head runs (DONE 2026-07-20, HW-validated).** The residual
      per-head cost after F7.4 was the AllScan *run* (comm-domain alloc/free + HCCL ring). New
      `SimplerAllscan.run_multi` / `run_multi_backward` do all `H` heads' boundary rings in ONE
      `worker.run` under ONE comm domain (disjoint per-head slots + per-head IO buffers — the
      `run_batch` structure generalised to per-head data); `forward_multihead` / `backward_multihead`
      call them, so both the HCCL worker build (F7.4) *and* the comm-domain alloc/free are now paid
      once per boundary phase. NO device kernels; `_submit_iter[_backward]` gained an optional
      per-head `bufs` arg (single-head `run`/`run_batch` unchanged — regression `run()` `[dk=dv=64,K=4]`
      exact `0.0`). **Result (P=2 128², within-job per-head increment): 8.85 s/head — down from
      host-loop 17.3 → F7.4 11.85 → 8.85 (≈49% total), correctness fwd 6.1e-7.** Residual ~8.85 s/head
      ≈ 2.5 s compute + ~6.3 s ring (the ring data movement is fundamental to H distinct collectives).
- [~] **F7.5b — Fused `[H·dk,dv]` collective / head-looped compute (DEFERRED — no-go for now).** Only
      a single fused collective (one ring over `H·dk` stacked rows) would cut the residual ~6.3 s/head
      ring; it needs the AllScan's K-blocking past the 128-row cap **and** a head dim threaded through
      every compute orchestration (large kernel change, HW-debug risk). It overlaps the F4/F6
      amortization + the pypto env work, and the whole op is compile/orchestration-dominated (~185 s),
      so the marginal gain is small. **Reserve for after the pypto env unblocks** (so both backends
      gain heads consistently); F7.5a already captured the env-independent share of the win.
- [x] **F7.6 — Batching + arbitrary `L` (DONE 2026-07-20).** A batch of `B` independent sequences is
      `B·H` independent GLA operators, so batching folds into the head axis — `MultiHeadZeCo` handles
      any number of heads, and `[P,B,H,L,·]` flattens to `[P,B·H,L,·]` and back (test
      `test_batching_folds_into_heads`, differentiable). Arbitrary (non-power-of-two) chunk count
      `N=L//C` works — the kernels just loop chunks (tests at `N=3,5`); any `L` divisible by `C` is
      supported.
- [ ] **F7.7 —** Arbitrary head/chunk dims outside `{16,32,64,128}` (e.g. `D=96`). Needs host-side
      padding to the next dispatchable tile size (zero-pad `dk`/`dv`; `C` is a chosen tuning knob so
      practically picked from the set). The dispatchable set already covers real head dims, so this is
      a low-priority follow-up.

**Done this cycle (forward):** F1 dcci-flush ring race **reproduced (41% at 128²) and fixed
properly** in `TNOTIFY_IMPL` (validated 0/640 on HW without the fence) — see
`pypto-allscan-race-fix`. Uncovered the **N>2 loop-carry miscompilation** (now F2) that was
previously misread as the race. (Earlier: fully-fused pypto forward built + shipped; native
`P=1` path; `stage2` as a distributed chip kernel on sim+HW; `P=1` `device=r` unroll fix
`1a18fb26`; `gla/bench.py`.)

---

# Section 2 — Backward pass (mostly not done)

**Building block DONE:** the **AllScan-collective backward** (reverse ring: `dS`, `dgamma`)
is complete and HW-verified in all four backends, and benchmarked (`zeco-backward-progress`,
`allscan/.../program_backward.py`, `bench_backward.py`). This provides the **boundary-state
gradient** exchange.

**NOT STARTED:** the **ZeCO/GLA operator backward** — the gradients `dQ, dK, dV, dA` through
the chunk-recurrent GLA compute, with the boundary gradient flowing through AllScan-backward.
No reference, no kernels yet. Phased exactly like the forward was:

### B1 — Math + torch reference  *(DONE 2026-07-10)*
- [x] Derived the chunk-recurrent GLA backward as the reverse of the forward
      (transposed matmuls; reverse linear-recurrence for the state `dS`).
- [x] `gla/common.py::expected_gla_backward` — analytic sequential golden; cross-checked
      against `torch.autograd` on `expected_gla` (~1e-6, `dA` exact).
- [x] `TorchZeCo.backward` — the SP decomposition mirroring the forward's composition
      (local autograd for each rank's stage A/C + `TorchAllscan.run_backward` for the
      boundary ring). Matches the golden across `P∈{1,2,4} × dk≠dv × C` (~3e-5).
      `ZeCoImpl` gained an optional `backward(Q,K,V,A,dO)->(dQ,dK,dV,dA)`.
      Bug found + fixed en route: the backward tape must hold each rank's *actual*
      boundary `S_recv = out[p-1]` (not 0) or `dQ/dA = ∂O/∂Q = hist` read a zero'd
      `hist` (only `∂O/∂S_recv` is value-free).

### B2 — torch.distributed  *(DONE 2026-07-10)*
- [x] `run_distributed_zeco_backward` (gloo): each rank does local autograd for stage A/C,
      one `dS_recv` exchange hands its stage-C boundary grad to the lower neighbour, and the
      existing `_all_scan_backward_p2p` reverse ring produces `(dS_total, dgamma)`. Verified
      per-rank == `expected_gla_backward` on the full sequence for `P=2,4`.
      Tests: `gla/tests/test_gla_backward.py` (15 cases, all pass).

### B3 — simpler  *(DONE 2026-07-20, HW-validated)*
- [x] **Chunk-parallel backward math + kernel blueprint.** `gla.common.gla_chunk_backward`
      — the explicit (no-autograd) reverse of `gla_chunk_scan`+`gla_reconstruct`, written
      op-for-op so the kernels implement it directly. Reduces to `expected_gla_backward` for
      P=1; matches autograd on the composed SP local half with a non-zero boundary. Two new
      CPU tests in `test_gla_backward.py` (19 pass). The `dg_cs = dq_o∘q − dk_o∘k` simplification
      and the `g_total` row-C−1 coupling were validated in a g_cs-space stage-split reference
      before any C++ was written.
- [x] **Two new orchestrations, NO new device kernel.** grad_o (`grad_o_orch.cpp`, output-stage)
      and grad_h (`grad_h_orch.cpp`, state-stage) are **pure matmul orchestrations** — all backward
      matmuls map onto the general `matmul` kernel (M,N,Kc × NN/TN/NT), and `chunk_o_prep`/
      `chunk_h_prep`/`chunk_o_elt` recompute the forward intermediates + mask-muls. The gate
      backward's reverse-cumsum reuses the `gate_cumsum` kernel fed an upper-triangular matrix
      (`triu @ dg_cs`). The cross-chunk grad recurrence (dSloc/dcvec), the gate arithmetic
      (dq/dk scaling, dg_cs assembly), and the dgamma corrections are the cheap host linear glue
      (mirroring the forward's `_S_total`/`_shift_snaps`). Matmuls write directly to external
      output views (`add_inout`) — a pattern the forward didn't use but which works on HW.
- [x] **SceneTests HW-validated** (dev6): `test_grad_o.py` + `test_grad_h.py`, 7 cases each —
      square {32,64,128}, rectangular C≠D, both dk≠dv directions — all pass.
- [x] **End-to-end `SimplerZeCo.backward`** reuses the existing simpler AllScan-backward for the
      boundary reverse ring (forward AllScan for `S_recv` + `run_backward` for `dS_total`/`dgamma`,
      sequenced around the compute so neither holds the devices while the other needs them).
      Validated vs `expected_gla_backward`: **P=1 HW 128² worst 5.7e-7; P=2 HW 128² worst 6.7e-7;
      P=2 HW dk≠dv worst 3.5e-7** (`test_simpler_gla_backward.py`, P=1 sim CI-safe).
- Note: no new WAR-hazard debug was needed — grad_o/grad_h are per-chunk independent (like
  chunk_o), so the cross-block WAR hazard that bit the AllScan-collective backward (the on-device
  reverse recurrence) does not arise; the reverse recurrence is on host here.

### B4 — pypto
- [ ] Fused distributed backward `@pl.program` (chunk-gradient InCore kernels + the existing
      reverse-ring `program_backward`), mirroring the forward fusion.
- **Depends on:** F2 (shared N>2 loop-carry fix) and F3 (shared tiling/scale work) — build
  backward on the hardened forward kernels, not the toy version.

### B5 — Tests + fair benchmark  *(simpler backward hardened 2026-07-20; fair vs-pypto still needs B4)*
- [x] **B5.1 — Backward correctness sweep** (`test_simpler_gla_backward.py --sweep`): P=1 sweeps the full
      10-shape set (square + rectangular `C≠D` + `dk≠dv`, `N=L//C` up to 16) — **10/10 HW-pass**
      (~1e-7); P>1 sweeps a boundary-focused 4-shape subset (compute is identical to P=1) —
      **P=2 4/4 HW-pass** through the real AllScan boundary (small-square/N=16, rect C<D, 128²,
      dk≠dv).
- [x] **B5.2 — Back-to-back backward stress** (`--stress`): build once, 16 fresh-input backward dispatches
      at P=2 128² through the two-AllScan-session pipeline (forward AllScan + reverse-ring
      `run_backward`) + the per-kernel worker cycling — **16/16 correct, worst 6.8e-7**. The full
      operator backward is stable/deterministic under repeated dispatch (the F1 stability check,
      now for the operator not just the collective).
- [x] **B5.3 — Steady-state backward latency** (`--bench`, build excluded): P=1 128² N=2 **mean ~7.6 s/call**
      — per-call cost is dominated by the runtime's one-callable-per-worker device-exclusive cycling
      (stage1 + grad_o + grad_h + reverse-cumsum each cycle a fresh worker; compile session-cached),
      the same limitation as the forward (F4 `[~]`), *not* compute. **P=2 128² ~71 s/call** — the ~63 s
      jump over P=1 is the *two* AllScan build/run/close sessions per backward (forward AllScan for
      `S_recv` + reverse-ring `run_backward`), each a full HCCL distributed-worker + comm-domain
      setup/drain. So the backward's per-call cost is orchestration-setup-bound (worker cycling + 2×
      AllScan setup), exactly the F4 pattern; a real steady-state number needs persistent compute
      workers + a persistent AllScan (amortization), not new compute.
- [ ] **B5.4 — Fair simpler-vs-pypto backward benchmark** — blocked on **B4** (no pypto operator backward
      to compare against yet). The AllScan-*collective* backward is already benchmarked head-to-head
      (`zeco-backward-progress`); extend to the full operator once B4 lands. Persistent-worker
      amortization (à la F4 for the forward) is the way to a meaningful steady-state operator number.

---

## Critical path

```
F1 (race fix, DONE bar upstreaming) ─ independent
F2 (N>2 loop-carry) ──> F3 (tiling/scale) ──┬─> F4 (steady-state bench) ──> F6 (fair numbers)
                        │                    └─> F5 (op hardening) ─────────┘
                        └────────────────────────────────────────> B4 (pypto backward)
B1 (bwd math/ref) ──> B2 ──> B3 ──> B4 ──> B5
```

**F2 (the N>2 loop-carry miscompilation) now unblocks everything** — it is the real
correctness gate. F1 (the race) is fixed at the framework level and only needs upstreaming +
workaround removal. **B1→B2→B3 are done (torch, torch-dist, simpler operator backward, HW-validated).**
The only remaining backward backend is **B4 (pypto fused)**, which reuses the forward's F2/F3
work, so it is most efficient **after** F2–F3 land.
