# AllScan

Implementations and benchmark for the **AllScan** collective. Over P ranks
arranged in a ring:

```
out[0] = S_local[0]
out[p] = S_local[p] + gamma[p] * out[p-1]      (p = 1 .. P-1)
```

`gamma[p]` is `[dk, 1]` and broadcasts across the `dv` columns of the `[dk, dv]`
state. Work is pipelined over `K` blocks of `dk // K` rows.

## Backward pass

Given the upstream gradient `g_out[p] = dL/dout[p]`, the adjoint `d[p] = dL/dout[p]`
(total, including the downstream chain) is a **reverse** scan with `gamma` shifted
by one, from which the input gradients are local:

```
d[P-1] = g_out[P-1]
d[p]   = g_out[p] + gamma[p+1] * d[p+1]         (p = P-2 .. 0)

dS_local[p] = d[p]                              (all p)
dgamma[p]   = rowsum_dv( d[p] * out[p-1] )      (p = 1 .. P-1)  -> [dk, 1]
dgamma[0]   = 0                                 (gamma[0] is unused)
```

The forward ring flows `rank -> rank+1`; the adjoint flows `rank -> rank-1`. Each
rank forwards the message `m = gamma[p] * d[p]` into the previous rank's recv slot,
and the receiver adds its own `g_out` to form `d`. `out[p-1]` (needed for `dgamma`)
is exactly the block each rank received during the forward pass, so the `dgamma`
row-reduction is fully local — no extra exchange. Roles by rank: rank `P-1` is the
source (no recv), ranks `1..P-2` are middle, rank `0` is the terminal (no send,
`dgamma[0] = 0`).

Backends expose the backward pass via `run_backward(g_out, gammas, outs, dS, dgamma)`
on the `AllscanImpl` interface (with the saved forward outputs `outs` passed in).

## Layout

```
common.py                       AllscanImpl interface; expected_allscan / make_inputs
                                and expected_allscan_backward / make_grad_inputs
conftest.py                     shared --platform / --device pytest fixtures
bench_allscan.py                forward benchmark — every registered implementation
bench_allscan_backward.py       backward benchmark — simpler vs pypto (amortized)

implementations/
  __init__.py                   REGISTRY (degrades gracefully if a backend's deps are missing)
  torch_ref.py                  torch — in-process CPU ring baseline; forward + backward,
                                plus standalone torch.distributed forward/backward references
  pypto/
    program.py                  build_allscan_program — the forward DSL program
    program_backward.py         build_allscan_backward_program — the reverse-ring DSL program
    batched_program.py          B-ring batched forward program generator (amortized timing)
    batched_backward_program.py B-ring batched backward program generator (amortized timing)
    impl.py                     PytoAllscan (forward) + PytoAllscanBackward (backward)
  simpler/
    impl.py                     simpler — direct PTO-runtime adapter + standalone CLI
    kernels/aiv/allscan_kernel.cpp                    forward per-rank AIV kernel
    kernels/aiv/allscan_backward_kernel.cpp           backward (reverse-ring) AIV kernel
    kernels/orchestration/allscan_orch.cpp            forward: one AIV task per chip
    kernels/orchestration/allscan_backward_orch.cpp   backward: one AIV task per chip

tests/
  test_torch.py                 CPU forward reference (any platform)
  test_torch_backward.py        CPU backward reference + torch.autograd cross-check (any platform)
  test_pypto.py                 forward DSL on --platform / --device
  test_pypto_backward.py        backward DSL on --platform / --device
  test_simpler.py               forward PTO runtime on --platform / --device
  test_simpler_backward.py      backward PTO runtime on --platform / --device
```

Every backend implements the `AllscanImpl` interface (`build` once, then `run` /
`run_backward` many, `close`), so the benchmark and tests drive them uniformly.

## Running

The execution target is selected with `--platform` (`a2a3` hardware, `a2a3sim`
simulator) and `--device` everywhere. On real hardware, preload HCCL via
`LD_PRELOAD` (see below).

### Tests

```bash
# simulator (virtual devices, no NPU occupied) — forward + backward
pytest tests/ --platform a2a3sim --device 0-3

# real Ascend hardware
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    pytest tests/ --platform a2a3 --device 4-7

# a single backend, or only the backward tests
pytest tests/test_simpler.py --platform a2a3sim --device 0-3
pytest tests/test_pypto_backward.py --platform a2a3 --device 4-7      # (with LD_PRELOAD)
```

`test_torch_backward.py` also independently validates the closed-form backward
against `torch.autograd`, so it anchors the math the device backends verify against.

### Benchmark

Both benchmarks time each backend's marginal kernel+comm cost by amortizing the
fixed per-dispatch orchestration overhead (comm-domain alloc/free + drain) across
a batch of `B = 16` dispatches (marked `*` in the table). The in-process torch
baseline has no real communication and is excluded from the backward benchmark.

```bash
# forward — simulator / real hardware / subset + JSON
python bench_allscan.py --platform a2a3sim --device 0-3
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    python bench_allscan.py --platform a2a3 --device 4,5,6,7
python bench_allscan.py --platform a2a3sim --device 0-3 --impl simpler pypto --json results.json

# backward — same flags; head-to-head simpler vs pypto
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    python bench_allscan_backward.py --platform a2a3 --device 4-7 --json results_bwd.json
```

### Standalone

```bash
# direct PTO-runtime implementation, ad-hoc config
python implementations/simpler/impl.py -p a2a3sim -d 0-3 --dk 128 --dv 128 --K 4

# torch.distributed point-to-point references (forward then backward)
python implementations/torch_ref.py
```

## The `simpler` kernels

**Forward** — one uniform AIV kernel runs on every rank and selects its behaviour
from `rankId`:

- **rank 0** — source: emit `S_local`, push block to rank 1, no wait.
- **rank 1..P-2** — receive from prev, fuse `S_local + gamma (*) recv` (`TROWEXPANDMUL` row broadcast), push to next.
- **rank P-1** — receive from prev, fuse, terminate the chain.

**Backward** — a second uniform kernel runs the reverse ring: rank `P-1` is the
source, rank `0` the terminal, `peer = rank - 1`. Each rank forms `d`, sends
`gamma * d` to the previous rank, and computes `dgamma = rowsum_dv(d * out_prev)`
locally (`TROWSUM`). It fits the 192KB UB (three `[128,128]` tiles) by sending the
message **before** the row-sum and aliasing the reduction scratch onto the (now
dead) `d` tile.

Both kernels forward the computed block straight into the peer's recv slot in the
shared HCCL window (remote `TSTORE`), flush it (`dcci`/`dsb`) and signal it
(`TNOTIFY`); the receiver `TWAIT`s before reading. The domain window is zeroed at
allocation, so per-block signals stay correct across repeated runs (`epoch = 1`
each call).

## Troubleshooting

**`comm_init … Timeout waiting for rootinfo` (or `HcclCommInitRootInfo failed`).**
A benchmark/test that was killed or crashed mid-init leaves stale HCCL rendezvous
files in `/tmp` (`barrier_pto_multi_comm_*…rootinfo…`). The next run's ranks then
wait on stale rendezvous state and time out. Clean them up and re-run:

```bash
find /tmp -maxdepth 1 -name 'barrier_pto_multi_comm_*' -delete
```

Also make sure the target NPUs are actually free (`npu-smi info` → "No running
processes"), and always run on hardware with the `LD_PRELOAD=…/libhccl.so` prefix.
Only one distributed worker may be prepared per device set at a time, so run the
forward and backward benchmarks sequentially, not concurrently, on the same devices.
