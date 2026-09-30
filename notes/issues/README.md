# allscan/issues/ — pypto / simpler / pto-isa bugs

**Canonical bug location** for the pto-zeco (AllScan / GLA / ZeCO) work. Each subfolder
holds a write-up (symptom → root cause → resolution), a runnable repro, and any patch,
ready to submit upstream. Historical index of the older batch: `UPSTREAM-ISSUES.md`.

Diagnostic scripts these write-ups cite live in **`../../devtools/`** (moved out of
`pto-zeco/scratchpad/` on 2026-08-07 — it was tracked in git and should never have been).

## Filed upstream, awaiting review
- **pto-isa-dir-both-ring-alias-repro/** — a2a3 `DIR_BOTH` `TPipe` gives both directions the
  same GM slot (`entryOffset` never set), so C2V and V2C overwrite each other. GitCode issue
  **#516** + MR **#1438** (`eaf9c09c`), which now also carries the regression test
  `tpushpop_dir_both_concurrent`. Reproducer here is byte-identical to the shipped one:
  without the fix case2 fails 40/40 and case1 30/40; with it, 0/40. The pypto half of the fix
  (workspace sized for two rings) merged as pypto **#2271**.
- **pypto-incore-loop-cube-vector-race/** — the original pypto-level framing of the same
  defect (F2: loop-carried tile corrupted at N>2). Kept until #1438 lands.

## Open
- **tcolexpand-pipe-sync/** — `TCOLEXPAND` declared `PIPE_V` but lowers to a ubuf copy →
  under-synchronised at small tiles (C=32 fails, C=128 passes; pto-isa `TColExpand.hpp`).
  Under discussion on GitCode; the `PIPE_ALL` workaround is still required. **`PIPE_MTE1`
  is not a fix** — it faults the AICore (507018) on all 8 cases.
- **fp32-cube-k-accumulation/** — fp32 cube K-accumulation (`TMATMUL_ACC`) unsupported on
  a2a3; blocks the GLA simpler backend at head dim > 128 (F3.4 / D=256).
- **simpler-second-callable-silent-corruption/** — a 2nd dispatch of a *different* callable
  on one L3 `Worker` silently returns wrong data (no error). Forces per-kernel worker
  cycling (F4.1 negative result).
- **pypto-jit-distributed-segfault/** — within one process, a `@pl.jit` dispatch then a
  distributed `prepare()` on the same device hangs/segfaults in the forked chip worker.
  The ZeCO operator works around it with `prepare→close→jit` phase ordering.

## Written up, not yet filed
- **pto-isa-cpu-sim-tassign-arity/** — from `82c91680` onward (incl. main tip) no CPU-SIM
  `TASSIGN` compiles: `cpu/TAssign.hpp:29` calls a 2-arg `assignData` that commit removed.
  This is why the pto-isa pin stops at `1cb027c8`. Repro + bisect included.
- **taskqueue-unresolved-auto-device/** — `task-submit` execs `npu-lock auto` when the
  supervisor's read races the broker's grant, killing the task in ~1 s with
  `invalid device_id 'auto'`. The guard at `task-submit:514` excludes the `none` sentinel but
  not the unresolved `auto`. One-line fix; resubmitting is the workaround.

## Diagnostics (not bugs)
- **hccl-comm-alloc-domain-windows/** — smallest 2-rank program that needs a communication
  domain. The box-wide `comm_alloc_domain_windows failed with code -1` outage it was written
  for was **resolved by a reboot on 2026-08-07**; kept because it cheaply separates "the box
  is broken" from "my code is broken" before a debugging session starts.

## Removed
_(2026-08-07, fixed upstream — archived at `../../.archive/issues-resolved-2026-08-07.tar.gz`)_
`pypto-crossbranch-phi-nameerror` (pypto #2183 `555bd4d1`) ·
`pypto-remote-store-notify-drain` (pypto #2168 relanded `InsertCommFence`; 0/128 dispatches
wrong) · `ptoas-fence-barrier-all-deadlock` (fixed in ptoas 0.54, which lowers to
`pipe_barrier(PIPE_ALL); dsb(DSB_DDR)`) · `ptoas050-commremoteoffset-inline` (the
0.50→0.52 F1 saga, superseded by the 0.54 environment) ·
`pto-isa-cpu-stub-cache-line-macro` (a pin problem, not a code problem — already fixed
upstream by `2d9d4288`; was marked do-not-file) · `CARD-STATUS-2026-08-05.md` (cards 1 and 6
recovered on reboot) · `incore_loop_bug.zip`, `pypto-ptoas-issues-pr2135.zip` (old
submission bundles).

_(2026-07-27, fixed upstream: `pypto-p1-unroll-codegen` — pypto #1984 `949ac6b9`;
`pypto-stage2-distchip-hang` — resolved, HW-verified via the fused ZeCO forward.)_

## Build hygiene
Always clean-rebuild before trusting a repro (`rm -rf /opt/pypto/build` + full
`pip install --force-reinstall`). A `.so`-only swap over a non-wiped build dir is
unreliable: it once produced a false positive ("loop-alloc window aliasing", withdrawn) and
understated the drain race (2/32 stale → 12/32 on a clean build).

## Operational notes (shared a2a3 box)
- **Every distributed / HCCL run must set**
  `LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so` — without it the
  base-communicator rootinfo handshake never completes and the run *hangs* (`comm_hccl.cpp:286
  Timeout waiting for rootinfo`). The hang is in `allocate_domain`'s lazy HCCL comm init, not
  `Worker.init()`.
- Before a run: delete stale `/tmp/barrier_pto_multi_comm_*`, ensure no stray
  `simpler`/`chip_process` workers, and prefer devices confirmed free via `npu-smi info`.
  Heavy HW iteration can leave the runtime degraded (leaked device memory, `507018`, hung
  rendezvous).
- The box is **shared** — another host runs jobs across all 8 cards, so a submission can sit
  pending for 20+ minutes. That is normal and is *not* an error.
- `task-submit` has a race: if the supervisor reads the task record before the broker has
  resolved `--device auto` to a card number, it runs `npu-lock auto …` and the task dies in
  ~1 s with `invalid device_id 'auto', must be a non-negative integer`. **Just resubmit** —
  card exhaustion shows up as *pending*, never as this error, so the message never means
  "no free card".
