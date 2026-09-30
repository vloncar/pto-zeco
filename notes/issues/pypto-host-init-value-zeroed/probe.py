"""Scope the create_tensor(init_value=NONZERO) silent-zero: value, dtype, and orchestration level."""
import sys, torch
import pypto.language as pl
from pypto import ir
from pypto.runtime.runner import RunConfig

DK = 64
PLATFORM = sys.argv[1] if len(sys.argv) > 1 else "a2a3"


def probe_host(val, dtype, tdtype):
    @pl.program
    class Probe:
        @pl.function(type=pl.FunctionType.InCore)
        def copy(self, src: pl.Tensor[[DK, 1], dtype],
                 O: pl.Out[pl.Tensor[[DK, 1], dtype]]) -> pl.Tensor[[DK, 1], dtype]:
            return pl.store(pl.load(src, [0, 0], [DK, 1]), [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, src: pl.Tensor[[DK, 1], dtype],
                 O: pl.Out[pl.Tensor[[DK, 1], dtype]]) -> pl.Tensor[[DK, 1], dtype]:
            return self.copy(src, O)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, O: pl.Out[pl.Tensor[[1, DK, 1], dtype]]) -> pl.Tensor[[1, DK, 1], dtype]:
            t = pl.create_tensor([DK, 1], dtype=dtype, init_value=val)
            for r in pl.range(1):
                self.chip(t, O[r], device=r)
            return O

    c = ir.compile(Probe, platform=PLATFORM)
    O = torch.zeros(1, DK, 1, dtype=tdtype).share_memory_()
    rt = c.prepare()
    try:
        rt(O)
    finally:
        rt.close()
    return O.min().item(), O.max().item()


def probe_chip(val):
    """Same fill, but the tensor is created inside the CHIP orchestrator instead."""
    @pl.program
    class Probe2:
        @pl.function(type=pl.FunctionType.InCore)
        def copy(self, src: pl.Tensor[[DK, 1], pl.FP32],
                 O: pl.Out[pl.Tensor[[DK, 1], pl.FP32]]) -> pl.Tensor[[DK, 1], pl.FP32]:
            return pl.store(pl.load(src, [0, 0], [DK, 1]), [0, 0], O)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip(self, O: pl.Out[pl.Tensor[[DK, 1], pl.FP32]]) -> pl.Tensor[[DK, 1], pl.FP32]:
            t = pl.create_tensor([DK, 1], dtype=pl.FP32, init_value=val)
            return self.copy(t, O)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(self, O: pl.Out[pl.Tensor[[1, DK, 1], pl.FP32]]) -> pl.Tensor[[1, DK, 1], pl.FP32]:
            for r in pl.range(1):
                self.chip(O[r], device=r)
            return O

    c = ir.compile(Probe2, platform=PLATFORM)
    O = torch.zeros(1, DK, 1).share_memory_()
    rt = c.prepare()
    try:
        rt(O)
    finally:
        rt.close()
    return O.min().item(), O.max().item()


print(f"platform={PLATFORM}")
for val in (0, 1, 2, 7):
    try:
        lo, hi = probe_host(val, pl.FP32, torch.float32)
        print(f"  HOST create_tensor FP32 init_value={val}: got [{lo}, {hi}] "
              f"{'OK' if lo == hi == val else 'WRONG'}")
    except Exception as e:
        print(f"  HOST FP32 init_value={val}: raised {type(e).__name__}: {str(e)[:120]}")
for val in (0, 1, 5):
    try:
        lo, hi = probe_host(val, pl.INT32, torch.int32)
        print(f"  HOST create_tensor INT32 init_value={val}: got [{lo}, {hi}] "
              f"{'OK' if lo == hi == val else 'WRONG'}")
    except Exception as e:
        print(f"  HOST INT32 init_value={val}: raised {type(e).__name__}: {str(e)[:120]}")
for val in (0, 1):
    try:
        lo, hi = probe_chip(val)
        print(f"  CHIP create_tensor FP32 init_value={val}: got [{lo}, {hi}] "
              f"{'OK' if lo == hi == val else 'WRONG'}")
    except Exception as e:
        print(f"  CHIP FP32 init_value={val}: raised {type(e).__name__}: {str(e)[:120]}")
