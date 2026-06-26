# AllScan

Implementations and benchmark for the **AllScan** collective. Over P ranks
arranged in a ring:

```
out[0] = S_local[0]
out[p] = S_local[p] + gamma[p] * out[p-1]      (p = 1 .. P-1)
```

`gamma[p]` is `[dk, 1]` and broadcasts across the `dv` columns of the `[dk, dv]`
state. Work is pipelined over `K` blocks of `dk // K` rows.

## Layout

```
common.py                       AllscanImpl interface, expected_allscan, make_inputs
conftest.py                     shared --platform / --device pytest fixtures
bench_allscan.py                compares every registered implementation

implementations/
  __init__.py                   REGISTRY (degrades gracefully if a backend's deps are missing)
  torch_ref.py                  torch — in-process CPU ring baseline (+ torch.distributed script)
  pypto/
    program.py                  build_allscan_program — the pypto.language DSL program
    impl.py                     pypto — DSL compiled via pypto.ir.compile
  simpler/
    impl.py                     simpler — direct PTO-runtime adapter + standalone CLI
    kernels/aiv/allscan_kernel.cpp          uniform per-rank AIV kernel
    kernels/orchestration/allscan_orch.cpp  one AIV task per chip

tests/
  test_torch.py                 CPU reference (any platform)
  test_pypto.py                 DSL on --platform / --device
  test_simpler.py               PTO runtime on --platform / --device
```

Every backend implements the `AllscanImpl` interface (`build` once, `run` many,
`close`), so the benchmark and tests drive them uniformly.

## Running

The execution target is selected with `--platform` (`a2a3` hardware, `a2a3sim`
simulator) and `--device` everywhere.

### Tests

```bash
# simulator (virtual devices, no NPU occupied)
pytest tests/ --platform a2a3sim --device 0-3

# real Ascend hardware
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    pytest tests/ --platform a2a3 --device 4-7

# one backend
pytest tests/test_simpler.py --platform a2a3sim --device 0-3
```

### Benchmark

```bash
# simulator
python bench_allscan.py --platform a2a3sim --device 0-3

# real hardware (preload HCCL)
LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so \
    python bench_allscan.py --platform a2a3 --device 4,5,6,7

# subset of implementations / save results
python bench_allscan.py --platform a2a3sim --device 0-3 --impl simpler pypto --json results.json
```

### Standalone

```bash
# direct PTO-runtime implementation, ad-hoc config
python implementations/simpler/impl.py -p a2a3sim -d 0-3 --dk 128 --dv 128 --K 4

# torch.distributed point-to-point reference
python implementations/torch_ref.py
```

## The `simpler` kernel

One uniform AIV kernel runs on every rank and selects its behaviour from `rankId`:

- **rank 0** — source: emit `S_local`, push block to rank 1, no wait.
- **rank 1..P-2** — receive from prev, fuse `S_local + gamma (*) recv` (`TROWEXPANDMUL` row broadcast), push to next.
- **rank P-1** — receive from prev, fuse, terminate the chain.

Each rank forwards its computed block straight into the next rank's recv slot in
the shared HCCL window (remote `TSTORE`) and signals it (`TNOTIFY`); the receiver
`TWAIT`s before reading. The domain window is zeroed at allocation, so per-block
signals stay correct across repeated runs (`epoch = 1` each call).
