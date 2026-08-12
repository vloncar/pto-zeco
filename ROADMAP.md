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
| **Forward** (GLA compute + AllScan boundary) | ✅ | ✅ | ✅ HW P=1/2/4 | ✅ HW P=1, `C,D ≤ 64`; ⚠️ **P=2 wrong at `dk != dv`** |
| **AllScan-collective backward** (building block) | ✅ | ✅ | ✅ HW | ✅ HW |
| **ZeCO/GLA operator backward** (dQ,dK,dV,dA) | ✅ | ✅ | ✅ HW P=1/2 | ❌ — **B4, the only gap** |

Both correctness gates that dominated this roadmap for months are **closed**: the cross-rank
producer race and the `N = L//C > 2` loop-carry corruption. Neither was ever a bug in this
operator — both were toolchain defects, and both fixes are now upstream.

**Two live constraints on pypto:**

1. **`dv >= C`.** Below that a `pl.matmul` silently returns corrupt results — a pto-isa FIFO
   defect (filed, fix in review; see the upstream ledger). `PyPtoZeCo.build` refuses those
   shapes rather than returning wrong data.
2. **`dk != dv` is wrong at `P >= 2`** — found 2026-08-12, see task 2. Not guarded yet.

Re-validated on HW 2026-08-12 (`test_pypto_gla.py`, full matrix, a2a3):

| L, C, dk, dv | P=1 | P=2 |
|---|---|---|
| 128, 32, 32, 32 | ✅ | ✅ |
| 128, 32, 64, 64 | ✅ | ✅ |
| **128, 64, 32, 64** | ✅ | ❌ **max diff 186.05** |
| 256, 64, 64, 64 | ✅ | ✅ |

simpler forward re-validated green in the same session.

## Environment

| | |
|---|---|
| pypto | `f621eca4` (main), **104 commits behind** `origin/main` |
| ptoas | **0.54** (`/opt/ptoas-bin`) |
| pto-isa | pin `83d01313` (`/opt/pto-isa`), **+1 local patch** — the DIR_BOTH V2C ring offset, since merged upstream as `69a81f3b` |
| simpler runtime | `9922afdb` |

Both carried patches are now redundant with upstream and go away on the next pin bump
(task 1). Run notes: `LD_PRELOAD=<cann>/lib64/libhccl.so` or HCCL hangs at rootinfo;
`PYTHONPATH` must *prepend* `pto-zeco` (`set_env.sh` resets it); delete stale
`/tmp/barrier_pto_multi_comm_*`; keep multi-card sets inside one HCCS group (0-3 | 4-7).

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
  they overlap iff `N < M` — `K` cancels, which is why `dk` never mattered. One-line fix:
  bare-matmul probe 8/18 → **18/18**, GLA 3/7 → **7/7**. Filed upstream.
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

### 1. Refresh the toolchain pin and drop the carried patches
Once pto-isa **!1457** merges, bump `/opt/pto-isa` past it and past `69a81f3b`, and bump
pypto past `d26e0f6c` (#2271). That retires **both** local patches — verified redundant:
our `orchestration_analysis.cpp` is byte-identical to upstream main. Then remove the
`dv >= C` guard in `PyPtoZeCo.build` and restore `(128, 64, 32, 32)` to `SIZES` as a
correctness case (`ZECO_ALLOW_TALL=1` already validates the fix ahead of the merge).

The old pin ceiling is **gone**: the CPU-SIM `TASSIGN` arity break that blocked bumping past
`1cb027c8` is fixed upstream (`439faf48`, `831ef9d2`). Re-verify the sim `P=1, L=128, C=D=32`
hang before trusting the new pin — that hang, not the arity break, was the real reason the
last bump was reverted, and it was never bisected.

### 2. pypto `dk != dv` is wrong at `P >= 2` — correctness gate
Found 2026-08-12 while re-validating this roadmap. `L=128, C=64, dk=32, dv=64` passes at
**P=1** and fails at **P=2**; the other three swept shapes pass at both. Two properties make
this its own bug rather than a variant of F3.1c:

- the failing shape is the **only one with `dk != dv`**, and it only breaks on the P>=2 path,
  i.e. the one that runs the real boundary ring;
- the error **magnitude varies between runs** (1.40, then 186.05 on the same shape) — race-like,
  where F3.1c was deterministic corruption.

**Hypothesis, not yet tested:** the boundary carries the `[dk, dv]` state across ranks, so
something on the exchange path is assuming a square state tile. P=1 never exercises it, which
is exactly why this survived — and F7.1 noted at the time that "pypto kernels not yet swept"
for `dk != dv`. That note was the outstanding risk; this is it coming due.

Steps: reproduce standalone (a P=2 `dk != dv` case, no pytest); check the boundary buffer
sizing/strides for a `dk == dv` assumption; A/B against the F3.1c-fixed pto-isa to rule it in
or out. **Note:** `PTO_ISA_ROOT` is honoured by `device_runner.py` (early return, no pin
check), but an A/B attempt on 2026-08-12 produced byte-identical output for both halves
including wall time — verify the override actually takes effect before trusting such a run.

Add a guard once characterized, as F3.1c got, so the operator cannot return corrupt results.

### 3. Forward-at-scale sweep
`N ∈ {2,4,8,16,32}` × `P ∈ {1,2,4}` × shapes, both device backends, against `expected_gla`.
F2 was fixed and spot-checked at 12/12, but the full sweep was never run — so "forward is
correct at scale" is an inference, not a measurement. Task 2 is precisely what a sweep would
have caught a month ago; run it before making any scale claim, and fold in `dk != dv` and
`P=4` explicitly.

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
