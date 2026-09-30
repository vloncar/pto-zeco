#!/usr/bin/env python3
"""B4 bisect: with the reverse ring removed, does the P>1 backward still STALL?

Everything else is exonerated by measurement:
  * ring direction / notify-before-wait — `test_pypto_backward` passes at P=4 with exactly
    that shape (descending flow, ascending `device=r` loop);
  * two comm windows + two opposite-direction rings in one program — `b4_tworing_probe`
    passes, both transfers exact;
  * the three compute kernels, with non-zero boundary values — `b4_kernel_probe`, 13/13;
  * the host_orch wiring — read out of the generated `host_orch.py`;
  * the cards — post-reset canary green on all six pairs, and P=2 still fails on them.

So the fault is in the interaction inside the full 5-dispatch program. This runs the stubbed
copy (`b4_stub_program.py`, reverse ring not dispatched) at P=2. Numbers are WRONG by
construction; the only question is whether it completes.

  completes -> the reverse-ring KERNELS own the stall (they differ from the probe's trivial
               ones: blocked K loop, row_expand_mul / row_sum inside the step, and they read
               GM tensors written by an earlier dispatch)
  stalls    -> the forward ring or the recompute -> grad_o chain owns it, and the reverse
               ring is innocent

Usage: python3 devtools/b4_bisect.py <device_csv> [platform]
"""

from __future__ import annotations

import sys
import time

import torch

from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

sys.path.insert(0, "/root/workspace/allscan")

# Which half to stub is a CLI arg so one driver serves both bisect directions.
_VARIANT = "reverse" if "--stub-forward" not in sys.argv else "forward"
if _VARIANT == "reverse":
    from devtools.b4_stub_program import (  # noqa: E402
        build_fused_backward_program as build_stubbed,
    )
else:
    from devtools.b4_stub_fwd import (  # noqa: E402
        build_fused_backward_program as build_stubbed,
    )

L, C, DK, DV = 32, 16, 16, 16


def main() -> int:
    devices = [int(x) for x in sys.argv[1].split(",")][:2]
    print(f"variant: {_VARIANT} ring stubbed out")
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    P = 2

    from gla.common import make_gla_inputs

    Q, K, V, A = make_gla_inputs(P, L, DK, DV, seed=42)
    torch.manual_seed(7)

    def sm(t):
        return t.contiguous().share_memory_()

    h = [sm(Q), sm(K), sm(V), sm(A), sm(torch.randn(P, L, DV)),
         sm(A.prod(dim=1).reshape(P, DK, 1)),
         sm(torch.tril(torch.ones(C, C))), sm(torch.triu(torch.ones(C, C))),
         sm(torch.zeros(DK, DV)), sm(torch.zeros(DK, 1)), sm(torch.ones(DK, 1))]
    outs = [sm(torch.zeros(P, L, DK)), sm(torch.zeros(P, L, DK)),
            sm(torch.zeros(P, L, DV)), sm(torch.zeros(P, L, DK))]

    print(f"compiling backward with the {_VARIANT} ring removed, P={P} on {devices}", flush=True)
    compiled = ir.compile(
        build_stubbed(L, C, DK, DV, 1, P), platform=platform,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    rt = compiled.prepare()
    try:
        t0 = time.time()
        rt(*h, *outs)
        print(f"DISPATCH COMPLETED in {time.time() - t0:.1f}s "
              f"-> this half is NOT the stall", flush=True)
        nz = sum(int(o.abs().sum().item() > 0) for o in outs)
        print(f"(sanity: {nz}/4 gradient buffers non-zero; values are WRONG by construction)")
    finally:
        rt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
