# Validation

Stack: pypto `71020585` + ptoas 0.57 + simpler `3165cc89` + pto-isa pin `83d01313` (+2 carried
patches). Platform a2a3, cards 0/1, comm canary green (`comm-capable pairs: [(0, 1)]`) at the
head of every run.

## 1. The mechanism, before any framework change

The reproducer plus ONE artificial RAW edge (the send's `Out` tensor threaded into the recv as
an extra `INPUT`), payload-neutral — `recv` stores `dst + (dep - dep)` — and nothing else
changed. That edge is exactly the fan-in the diagnosis says is missing.

| arm | result |
| --- | --- |
| `sendfirstdep` ×2 | **completed 2.3s / 2.4s**, both payloads correct |
| `bothdep` ×2 | **completed 2.2s / 2.4s**, both payloads correct |
| `sendfirst` control | **SCHEDULER_TIMEOUT -100** (dev=1) |
| `both` control | **SCHEDULER_TIMEOUT -100** (dev=0) |

Stalled runs report `dispatch_id=2`; passing runs report `dispatch_id=5`.

## 2. The fix, on the unmodified programs

Same programs as the controls above — **no DSL change**, only the patched codegen.

| arm | before | after |
| --- | --- | --- |
| `sendfirst` | SCHEDULER_TIMEOUT -100 | **completed 2.2s**, `fwd_payload_ok=True rev_payload_ok=True` |
| `both` | SCHEDULER_TIMEOUT -100 | **completed 2.3s**, `fwd_payload_ok=True rev_payload_ok=True` |

## 3. No regression

| gate | result |
| --- | --- |
| `fwd` (one ring) | completed 2.3s, payload correct |
| `samedir` (two rings, one direction) | completed 2.3s, payloads correct |
| `fusedcomm` (comm fused per rank) | completed 2.3s, payloads correct |
| pypto `tests/ut/codegen/` + `test_distributed_compiled_program.py` | **865 passed** |
| `allscan/tests/test_pypto.py` (incl. the P=4 race guard) | **4 passed** |
| `allscan/tests/test_pypto_backward.py` | **3 passed** |
| `gla/tests/test_pypto_gla_backward.py` P=1/P=2/P=4 | **3 passed** (cards 0-3, `--device 0,1,2,3`) |

The last row is the point of the exercise: the ZeCO fused distributed backward deadlocked at
every `P>1` config before this change and is now numerically correct at **P=2 and P=4**
(`err < 1e-3`, 3 seeds each), with no change to the operator itself.

Caveat on reading that row: `conftest.py` derives `device_ids` from the pytest `--device` flag
alone (default `0,1`) and ignores `ASCEND_RT_VISIBLE_DEVICES`. Without `--device 0,1,2,3` the P=4
case skips its `len(device_ids) < P` guard and pytest prints "2 passed, 1 skipped" — identical to
a real 2-card run. The 3-passed result above is from a run that passed the flag explicitly.

Two golden assertions in `tests/ut/codegen/test_distributed_codegen.py` were updated for the
intentional `_alloc_intermediates(tensors, world_size=1)` signature change; they are the only
test edits in the patch.

Generated-code shape verified by host-only compile (no NPU):

| program | comm dispatches tokened | alloc lines |
| --- | --- | --- |
| reproducer P=2 | 4 (send/recv on both ranks), 0 compute dispatches | 1 |
| ZeCO fused backward P=2 | 6 — every `chip_orch_*` / `chip_bwd_*` | 1 |
| ZeCO fused backward P=1 (no comm) | 0 | 0 |
| allscan forward / backward P=4 | 3 each | 1 |
