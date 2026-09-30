# devtools/ — diagnostic and probe scripts

Working scripts for the pto-zeco (AllScan / GLA / ZeCO) effort: shape probes, race
diagnostics, benchmark decomposition, environment shims, and the pto-isa ST reproducer
harness.

**This lives outside `pto-zeco/` on purpose.** It used to be `pto-zeco/scratchpad/` and was
tracked in git, which it never should have been — these are throwaway diagnostics, not part
of the deliverable. `scratchpad/` is now in `pto-zeco/.gitignore` so it cannot come back.

Nothing here is a supported interface. Scripts hardcode paths, assume a device is granted,
and are kept only because a ROADMAP entry or an open issue write-up cites their output.

## Environment shims

| script | use |
|---|---|
| `tq_env.sh` | env for TaskQueue-submitted runs that touch an NPU: `task-submit ... "bash devtools/tq_env.sh <cmd>"` |
| `host_env.sh` | env for anything that does *not* touch an NPU |
| `sim_run.sh` | a2a3sim pytest runner: clears stale rendezvous, forces unbuffered output |

## F2 — cube↔vector pipe race (`pypto-incore-loop-cube-vector-race/`)

`f2_diag.py` + `f2_diag_prog.py` (stage-split of the fused program), `f2_stage1_only.py`,
`f2_stage1_first_bad.py` (first-bad-chunk / repeat stability), `f2_stage1_sync.py`
(barrier-in-loop variants), `f2_slotnum.py`, `f2_sweep054.py`.

## F3.1 — pypto shape ceiling

`f31_compile_probe.py` parses `passes_dump/33_after_AllocateMemoryAddr.py` and reports peak
bytes per memory space — **it selects the dump by set-difference against the dumps present
before the run**, not by newest mtime, which silently reported another config's numbers.
`f31_gamma_probe.py` checks the `col_sum` → gamma chain. `f31_shape_check.py` sweeps shapes.

## F6 — benchmark decomposition

`f6_phase_breakdown.py`, `f6_comm_breakdown.py`, and the captured `f6_bench.json` /
`f6_bench_amortized.json` that ROADMAP F6.3/F6.4 quote.

## pto-isa ST reproducer harness

| script | use |
|---|---|
| `isa_st_run.sh` | build + run one a2a3 ST case with the DIR_BOTH fix applied or reverted: `isa_st_run.sh <repo> <case> <stock\|fixed>` |
| `repro_flakiness.sh` | pass/fail rate over N runs in both modes; set `PTO_ISA_REPO` |

`isa_st_run.sh` asserts the header state matches the requested mode and aborts otherwise —
an earlier version used `git stash`, which is a no-op against a *committed* fix and produced
a "stock" run that silently still contained it. Note `build_st.py` **only builds**;
`run_st.py` runs. Both cap the kernel at 180 s, because a hung kernel otherwise spins at
100% CPU holding the device lock until the queue's 40-minute supervisor timeout.

## Persistent-worker probes

`check_persistent_forward.py`, `probe_persistent_worker.py` — for
`simpler-second-callable-silent-corruption/` (still open).
