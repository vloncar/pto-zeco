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
| **Forward** (GLA compute + AllScan boundary) | ✅ | ✅ | ✅ HW P=1/2/4 | ✅ HW `C,D ≤ 64`, `min(dk,dv) >= C` (guarded) |
| **AllScan-collective backward** (building block) | ✅ | ✅ | ✅ HW | ✅ HW |
| **ZeCO/GLA operator backward** (dQ,dK,dV,dA) | ✅ | ✅ | ✅ HW P=1/2 | ❌ — **B4, the only gap** |

Both correctness gates that dominated this roadmap for months are **closed**: the cross-rank
producer race and the `N = L//C > 2` loop-carry corruption. Neither was ever a bug in this
operator — both were toolchain defects, and both fixes are now upstream.

**Two live constraints on pypto:**

1. **`dv >= C`.** Below that a `pl.matmul` silently returns corrupt results — a pto-isa FIFO
   defect (filed, fix in review; see the upstream ledger). `PyPtoZeCo.build` refuses those
   shapes rather than returning wrong data.
2. **`dk >= C` too.** The same defect reaches the `dk` side via `b = tril[C,C] @ la[C,dk]`,
   intermittently (~1 in 20 dispatches, vs every dispatch for `dv < C`). Both are now guarded
   as `min(dk, dv) >= C` — see task 2.

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

### 1. Move pto-isa off `83d01313` and drop the last carried patch
**Partly done 2026-08-12:** pypto is on main and the pypto-side F2 patch is gone (upstream as
#2271). What remains is pto-isa, which pypto still pins at `83d01313` — predating both
`69a81f3b` (so we carry the DIR_BOTH patch) and !1457.

The pin comes from `runtime/pto_isa.pin` in the simpler submodule, so it moves when the
runtime bumps it; we cannot advance it unilaterally without diverging from what the runtime
enforces (`HEAD == pin`). Once !1457 merges and the runtime pin moves past both commits:
drop the DIR_BOTH patch, remove the `dv >= C` guard in `PyPtoZeCo.build`, and restore
`(128, 64, 32, 32)` to `SIZES` as a correctness case (`ZECO_ALLOW_TALL=1` validates the fix
ahead of that).

The old pin *ceiling* is gone: the CPU-SIM `TASSIGN` arity break that blocked bumping past
`1cb027c8` is fixed upstream (`439faf48`, `831ef9d2`). Re-verify the sim `P=1, L=128, C=D=32`
hang before trusting any new pin — that hang, not the arity break, was the real reason the
last bump was reverted, and it was never bisected.

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

### 4. B4 — pypto fused distributed backward
The last unimplemented backend: chunk-gradient InCore kernels plus the existing reverse-ring
`program_backward`, mirroring the forward fusion. Its dependencies (F2, F3) are now met, and
B3 supplies a validated op-for-op blueprint (`gla.common.gla_chunk_backward`) that the
simpler kernels already implement. Build on the hardened forward kernels.

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
| F3.1c — consumer local FIFO ring aliases | pto-isa issue **#521** / MR **!1457** | **in review** — gates task 1 |
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
