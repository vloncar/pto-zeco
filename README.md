# pto-zeco

Multi-backend implementations and benchmarks for **ZeCO** (Zero Communication
Overhead sequence parallelism for Gated Linear Attention) and its communication
core, the **AllScan** collective. Two layers:

```
allscan/   the AllScan collective — the SP communication primitive (forward + backward)
gla/       the ZeCO / GLA operator — the linear-attention compute built on AllScan
common/    backend-agnostic bench/CLI harness shared by both layers
```

The GLA operator defers its cross-device boundary-state hand-off to an AllScan
backend, so `gla/` literally composes `allscan/`.

---

## AllScan (`allscan/`)

Over P ranks arranged in a ring:

```
out[0] = S_local[0]
out[p] = S_local[p] + gamma[p] * out[p-1]      (p = 1 .. P-1)
```

`gamma[p]` is `[dk, 1]` and broadcasts across the `dv` columns of the `[dk, dv]`
state. Work is pipelined over `K` blocks of `dk // K` rows.

### Backward pass

Given the upstream gradient `g_out[p] = dL/dout[p]`, the adjoint `d[p]` (total,
including the downstream chain) is a **reverse** scan with `gamma` shifted by one,
from which the input gradients are local:

```
d[P-1] = g_out[P-1]
d[p]   = g_out[p] + gamma[p+1] * d[p+1]         (p = P-2 .. 0)

dS_local[p] = d[p]                              (all p)
dgamma[p]   = rowsum_dv( d[p] * out[p-1] )      (p = 1 .. P-1)  -> [dk, 1]
dgamma[0]   = 0                                 (gamma[0] is unused)
```

The forward ring flows `rank -> rank+1`; the adjoint flows `rank -> rank-1`. Each
rank forwards `m = gamma[p] * d[p]` into the previous rank's recv slot, and the
receiver adds its own `g_out` to form `d`. `out[p-1]` (needed for `dgamma`) is
exactly the block each rank received during the forward pass, so the `dgamma`
row-reduction is fully local. Roles by rank: rank `P-1` source (no recv),
`1..P-2` middle, rank `0` terminal (no send, `dgamma[0] = 0`).

Backends expose the pass via `run` / `run_backward(g_out, gammas, outs, dS, dgamma)`
on the `AllscanImpl` interface. Status: torch, simpler, pypto — forward + backward,
all HW-verified.

---

## ZeCO / GLA (`gla/`)

Sequence-parallel Gated Linear Attention. Single head, data-dependent per-key-dim
decay. The sequential golden (`gla.common.expected_gla`) is plain recurrent GLA
over the full `P*L`-token sequence:

```
S_t = diag(a_t) @ S_{t-1} + k_t^T v_t           # state  S in R^{dk x dv}
o_t = q_t @ S_t                                  # output o_t in R^{dv}
```

ZeCO computes the same thing in parallel over `P` devices, each holding a
contiguous `L`-token slice split into `N = L // C` chunks:

- **A — local chunk scan** (`gla_chunk_scan`): per chunk, within-chunk cumulative
  decay `b_t = prod_{j<=t} a_j`, chunk total decay `gamma = b_{C-1}`, the local
  state `S_[n]` and running cumulative decay, plus the device totals `S_total`,
  `g_total`.
- **B — intra-chunk masked attention** (overlaps comm): `scores[t,s] =
  (q_t*b_t)·(k_s/b_s)` for `s <= t`, `O_intra = scores @ V`. In parallel,
  **AllScan** the boundary state: `S_local[p] = S_total`, `gamma[p] = g_total`,
  yielding each device's incoming prefix `S_recv = out[p-1] = S_{(p-1)L}`.
- **C — output reconstruction** (`gla_reconstruct`): `O_inter[n] = (Q[n]*b) @
  (S_prev[n] + diag(c_prev[n]) @ S_recv)`, then `O = O_inter + O_intra`.

The chunk math lives in `gla/common.py` so every torch-level backend shares one
implementation and agrees by construction.

The **pypto** backend uses the same **chunk-recurrent `O(L·C)` form** (`N = L//C`
chunks) as the torch/simpler backends, but is forced into a **hybrid** split by two
pypto limitations:

- **stage1** (local end-of-slice state, scan from `S = 0`) is **fused with the
  AllScan ring** into one distributed `@pl.program` (`dist_program.py`), and
- **stage2** (per-rank output `O[r]`, the chunk recurrence initialised from
  `S_recv[r] = out[r-1]`) runs as a `@pl.jit` kernel (`program.py`) **after** the
  distributed worker `close()`s.

The split is not by choice: a `@pl.jit` dispatch *before* `DistributedWorker.prepare()`
segfaults, so the sole jit (stage2) must run last; and the wide-DAG stage2 kernel
**hangs as a distributed chip kernel** (`507018`), so it cannot be fused into the
distributed program and must stay on `@pl.jit`. A fully-fused single distributed
pypto program (the "full" pypto ZeCO) is therefore **postponed** — see `issues/`.
**Hardware-only:** the chunk kernels deadlock the a2a3sim simulator.

Backends implement the `ZeCoImpl` interface (`build` once, then
`forward(Q, K, V, A)` many, `close`). **Status:** forward done and verified
against the golden for the torch reference (in-process), torch.distributed,
**pypto** (composes `stage1`/`stage2` `@pl.jit` kernels with the PyPTO AllScan),
and **simpler** (hand-written PTO-ISA kernels + the real AllScan boundary —
validated on a2a3 HW at P=1/2/4, exact to ~1e-6). The full backward is out of
current scope.

---

## Layout

```
common/
  harness.py                    parse_devices / percentile / print_table (shared bench plumbing)
conftest.py                     shared --platform / --device pytest fixtures (repo root)

allscan/
  common.py                     AllscanImpl; expected_allscan / make_inputs + backward variants
  bench.py                      forward benchmark — every registered backend
  bench_backward.py             backward benchmark — simpler vs pypto (amortized)
  implementations/
    __init__.py                 REGISTRY (degrades gracefully if a backend's deps are missing)
    torch_ref.py                torch — in-process CPU ring + standalone torch.distributed refs
    pypto/                      DSL programs (fwd/bwd, single + batched) + impl.py
    simpler/                    PTO-runtime adapter + AIV/orchestration kernels (fwd/bwd)
  tests/                        test_{torch,pypto,simpler}[_backward].py

gla/
  common.py                     ZeCoImpl; expected_gla / make_gla_inputs; gla_chunk_scan / gla_reconstruct
  implementations/
    __init__.py                 REGISTRY
    torch_ref.py                TorchZeCo (in-process, composes AllScan) + torch.distributed ref
    pypto/
      program.py                stage2 @pl.jit chunk-recurrent GLA kernel (shape-baked)
      dist_program.py           stage1 + AllScan-ring fused as one distributed @pl.program
      impl.py                   PyPtoZeCo — hybrid: fused dist stage1+ring, then @pl.jit stage2 (HW-only)
    simpler/                    hand-written aic/aiv + orchestration kernels + impl.py (real AllScan boundary)
  tests/
    test_torch_gla.py           CPU: chunk==recurrent, in-process ZeCO, and gloo ring vs golden
    test_pypto_gla.py           on-device: PyPtoZeCo vs golden (P=1 compute, P>=2 full SP path)
    test_simpler_gla.py         on-device/sim: SimplerZeCo vs golden (defaults to a2a3sim P=1)
```

---

## Running

Target is selected with `--platform` (`a2a3` hardware, `a2a3sim` simulator) and
`--device`. On real hardware, preload HCCL via `LD_PRELOAD`.

### Tests

```bash
# CPU-only references (any platform) — AllScan + ZeCO torch layers
pytest allscan/tests/test_torch.py allscan/tests/test_torch_backward.py gla/tests/test_torch_gla.py

# on-device AllScan backends: simulator / real hardware
pytest allscan/tests/ --platform a2a3sim --device 0-3
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    pytest allscan/tests/ --platform a2a3 --device 4-7

# on-device pypto ZeCO (compute + AllScan boundary): simulator / real hardware
pytest gla/tests/test_pypto_gla.py --platform a2a3sim --device 0,1
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    pytest gla/tests/test_pypto_gla.py --platform a2a3 --device 4,5
```

`test_torch_backward.py` cross-checks the AllScan backward against `torch.autograd`;
`test_torch_gla.py` locks the GLA chunk math against plain recurrent GLA;
`test_pypto_gla.py` checks the on-device pypto ZeCO against the same golden.

### Benchmark

Both AllScan benchmarks amortize the fixed per-dispatch orchestration overhead
across `B = 16` dispatches (marked `*`), reporting marginal kernel+comm cost.

```bash
python allscan/bench.py --platform a2a3sim --device 0-3
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    python allscan/bench.py --platform a2a3 --device 4,5,6,7
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    python allscan/bench_backward.py --platform a2a3 --device 4-7 --json results_bwd.json
```

### Standalone

```bash
# direct PTO-runtime AllScan, ad-hoc config
python allscan/implementations/simpler/impl.py -p a2a3sim -d 0-3 --dk 128 --dv 128 --K 4

# torch.distributed references (spawn one gloo process per rank)
python -m allscan.implementations.torch_ref     # AllScan forward + backward
python -m gla.implementations.torch_ref         # ZeCO forward
```

Run these from the repo root (`pto-zeco/`) so the packages import.

## The `simpler` AllScan kernels

**Forward** — one uniform AIV kernel runs on every rank and selects behaviour from
`rankId`: rank 0 source (emit `S_local`, push, no wait); ranks `1..P-2` receive,
fuse `S_local + gamma (*) recv` (`TROWEXPANDMUL` row broadcast), push; rank `P-1`
receive, fuse, terminate.

**Backward** — a second uniform kernel runs the reverse ring: rank `P-1` source,
rank `0` terminal, `peer = rank - 1`. Each rank forms `d`, sends `gamma * d` to the
previous rank, and computes `dgamma = rowsum_dv(d * out_prev)` locally (`TROWSUM`).
It fits the 192KB UB by sending the message **before** the row-sum and aliasing the
reduction scratch onto the (now dead) `d` tile.

Both kernels forward the computed block straight into the peer's recv slot in the
shared HCCL window (remote `TSTORE`), flush (`dcci`/`dsb`) and signal it (`TNOTIFY`);
the receiver `TWAIT`s before reading. The domain window is zeroed at allocation, so
per-block signals stay correct across repeated runs (`epoch = 1` each call).

## Troubleshooting

**`comm_init … Timeout waiting for rootinfo` (or `HcclCommInitRootInfo failed`).**
A killed/crashed run leaves stale HCCL rendezvous files in `/tmp`
(`barrier_pto_multi_comm_*…rootinfo…`); the next run's ranks then wait on stale
state and time out. Clean them up and re-run:

```bash
find /tmp -maxdepth 1 -name 'barrier_pto_multi_comm_*' -delete
```

Also make sure the target NPUs are free (`npu-smi info` → "No running processes"),
and always run on hardware with the `LD_PRELOAD=…/libhccl.so` prefix. Only one
distributed worker may be prepared per device set at a time, so run the forward and
backward benchmarks sequentially, not concurrently, on the same devices.
