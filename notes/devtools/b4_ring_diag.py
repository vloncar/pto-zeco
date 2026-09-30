#!/usr/bin/env python3
"""B4 debug: localise the P>1 backward failure to a ring, a rank, and a dispatch index.

What is already established, so this does not re-test it:
  * all three compute kernels are correct in isolation, INCLUDING with artificially
    non-zero `S_recv` / `dS_total` / `dgamma` (`b4_kernel_probe.py`, sim);
  * the host_orch tensor wiring is correct (read out of the generated `host_orch.py`);
  * P=1 is correct end to end on HW.

So the fault is in how the two rings behave at run time. Two independent questions, each
answered by one axis of this script:

  **which rank** — the boundary enters rank r's result by exactly two routes. If only rank 0
  is wrong, the REVERSE ring (`dS_total[0]`, the terminal step) is the suspect; if only ranks
  >0 are wrong, the FORWARD ring (`S_recv[r]`) is; if all ranks are wrong, both or neither.

  **which dispatch** — dispatch 0 correct and 1..n wrong means cross-dispatch state: the ring
  windows and their AtomicAdd signal slots are reused every dispatch, so a signal that is not
  reset makes `wait(expected=1, Ge)` fall straight through from the second dispatch on. The
  suite takes the WORST over 3 repeats, which hides exactly this distinction.

Usage: python3 devtools/b4_ring_diag.py <device_csv> [platform] [P]
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla_backward, flatten_seq, make_gla_inputs
from gla.implementations.pypto.impl import PyPtoZeCo

L, C, DK, DV = 32, 16, 16, 16


def main():
    devices = [int(x) for x in sys.argv[1].split(",")]
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    devices = devices[:P]

    impl = PyPtoZeCo()
    impl.build(P, L, C, DK, DV, device_ids=devices, platform=platform)
    try:
        Q, K, V, A = make_gla_inputs(P, L, DK, DV, seed=42)
        torch.manual_seed(7)
        dO = torch.randn(P, L, DV)
        gQ, gK, gV, gA = expected_gla_backward(
            flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO))
        ref = [g.reshape(P, L, -1) for g in (gQ, gK, gV, gA)]

        # Forward first, as a control: it uses the SAME forward ring. If the forward is
        # correct at this config then S_recv is right and the reverse ring owns the failure.
        fwd_err = (impl.forward(Q, K, V, A) - _fwd_ref(Q, K, V, A)).abs().max().item()
        print(f"control: fused FORWARD max err {fwd_err:.3e}"
              f"   ({'ok' if fwd_err < 1e-2 else 'ALSO WRONG'})", flush=True)

        print(f"\n{'disp':>4}  " + "  ".join(f"{nm}[rank]" for nm in ("dQ", "dK", "dV", "dA")))
        for d in range(4):
            got = impl.backward(Q, K, V, A, dO)
            cells = []
            for g, r in zip(got, ref):
                per_rank = [((g[p] - r[p]).abs().max() / (r[p].abs().max() + 1e-6)).item()
                            for p in range(P)]
                cells.append(" ".join(f"{e:7.1e}" for e in per_rank))
            print(f"{d:>4}  " + "  |  ".join(cells), flush=True)
    finally:
        impl.close()
    return 0


def _fwd_ref(Q, K, V, A):
    from gla.common import expected_gla
    P, L_, dv = V.shape
    return expected_gla(flatten_seq(Q), flatten_seq(K), flatten_seq(V),
                        flatten_seq(A)).reshape(P, L_, dv)


if __name__ == "__main__":
    sys.exit(main())
