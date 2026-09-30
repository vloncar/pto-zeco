#!/usr/bin/env python3
"""How long does simpler's leftover host arithmetic take, on its own? (CPU only, no cards.)

``host_device_split.py`` measures the leftover as a *residual* of a whole call on hardware,
where it competes with worker lifecycle noise measured in seconds. This times the same
arithmetic directly, so the two numbers can be checked against each other and so we know
what porting it to the chips would actually save.

The two forward pieces call the shipped functions. The backward blocks are inline in
``SimplerZeCo.backward`` and cannot be called in isolation, so they are **re-implemented
here** — line-for-line from impl.py's Phase A/C/D. That is a copy, and it can drift; it is
a measurement aid, not a second source of truth. Cross-check against the residual.
"""

from __future__ import annotations

import sys
import time

import torch

from gla.common import make_gla_inputs
from gla.implementations.simpler.impl import _S_total, _shift_snaps

CONFIGS = [(2, 128, 32, 32), (4, 128, 32, 32), (2, 256, 32, 32),
           (4, 256, 32, 32), (2, 128, 32, 64), (4, 128, 32, 64)]
REPS = 20


def _time(fn, reps=REPS):
    fn()                                       # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1e3 / reps


def forward_glue(P, L, C, dk, dv):
    """Per-CALL cost of the forward's host arithmetic (summed over the P ranks)."""
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    N = L // C
    g_cs = torch.randn(L, dk).cumsum(0) * -1e-3
    s_snap = torch.randn(N, dk, dv)
    S_recv = torch.randn(dk, dv)

    log_ms = _time(lambda: [torch.log(A[p]).contiguous() for p in range(P)])
    stot_ms = _time(lambda: [_S_total(s_snap, g_cs, K[p], V[p], L, C) for p in range(P)])
    shift_ms = _time(lambda: [_shift_snaps(s_snap, A[p], S_recv, L, C, dk) for p in range(P)])
    gam_ms = _time(lambda: torch.stack(
        [A[p].reshape(-1, dk).prod(dim=0).reshape(dk, 1) for p in range(P)]))
    return {"log": log_ms, "S_total": stot_ms, "shift_snaps": shift_ms, "gammas": gam_ms}


def backward_glue(P, L, C, dk, dv):
    """Per-CALL cost of the backward's host arithmetic (summed over the P ranks).

    Re-implemented from impl.py Phase A/C/D — see the module docstring.
    """
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    N = L // C
    g_cs = torch.randn(L, dk).cumsum(0) * -1e-3
    s_snap = torch.randn(N, dk, dv)
    S_recv = torch.randn(dk, dv)
    dH = torch.randn(N, dk, dv)
    dQt = torch.randn(L, dk); dKin = torch.randn(L, dk)
    dKstate = torch.randn(L, dk)
    dS_tot = torch.randn(dk, dv); dgam = torch.randn(dk)

    def decay_block():                       # impl.py:662-670
        for _ in range(P):
            g_last = g_cs.reshape(N, C, dk)[:, -1, :]
            gam = torch.exp(g_last)
            c = torch.ones(N, dk)
            if N > 1:
                c[1:] = torch.cumprod(gam, dim=0)[:-1]

    g_last = g_cs.reshape(N, C, dk)[:, -1, :]
    gam_n = torch.exp(g_last)
    cprev = torch.ones(N, dk)
    if N > 1:
        cprev[1:] = torch.cumprod(gam_n, dim=0)[:-1]

    def gate_o_block():                      # impl.py:691-695
        for p in range(P):
            e = torch.exp(g_cs); ei = torch.exp(-g_cs)
            dqo = dQt * e
            dko = dKin * ei
            _ = dqo * Q[p] - dko * K[p]

    def dcp_loop():                          # impl.py:700-706  (python loop over chunks)
        for _ in range(P):
            dcp = torch.zeros(N, dk)
            acc = torch.zeros(dk, dv)
            for n in range(N):
                dcp[n] = (dH[n] * S_recv).sum(dim=1)
                acc += cprev[n].unsqueeze(1) * dH[n]

    def reverse_recurrence():                # impl.py:723-733  (python loop over chunks)
        for _ in range(P):
            dSloc = torch.zeros(N, dk, dv)
            dcvec = torch.zeros(N, dk)
            cur_S = dS_tot.clone(); cur_c = dgam.clone()
            for m in reversed(range(N)):
                dSloc[m] = cur_S; dcvec[m] = cur_c
                if m > 0:
                    cur_S = gam_n[m].unsqueeze(1) * cur_S + dH[m]
                    cur_c = gam_n[m] * cur_c + dH[m][:, 0]

    dSloc0 = torch.randn(N, dk, dv); dcvec0 = torch.randn(N, dk)

    def gate_h_loop():                       # impl.py:736-751  (python loop over chunks)
        for p in range(P):
            dgcs_p = torch.randn(L, dk)
            dk_h = torch.zeros(L, dk)
            g_cs_ch = g_cs.reshape(N, C, dk)
            for n in range(N):
                lo, hi = n * C, (n + 1) * C
                gtot = g_cs_ch[n, -1, :]
                dkh = dKstate[lo:hi] * torch.exp(gtot.unsqueeze(0) - g_cs[lo:hi])
                dk_h[lo:hi] = dkh
                dgcs_p[lo:hi] += -dkh * K[p][lo:hi]
                dgs = (dSloc0[n] * s_snap[n]).sum(dim=1)
                dgc = dcvec0[n] * cprev[n]
                dgcs_p[hi - 1] += (dgs + dgc) * gam_n[n] + (dkh * K[p][lo:hi]).sum(dim=0)

    fwd = forward_glue(P, L, C, dk, dv)      # backward also runs stage1 + shift_snaps
    return {
        "stage1_log+S_total": fwd["log"] + fwd["S_total"],
        "shift_snaps": fwd["shift_snaps"],
        "gammas": fwd["gammas"],
        "decay_block": _time(decay_block),
        "gate_o_block": _time(gate_o_block),
        "dcp_loop": _time(dcp_loop),
        "reverse_recurrence": _time(reverse_recurrence),
        "gate_h_loop": _time(gate_h_loop),
    }


def main() -> int:
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    print(f"CPU-only micro-benchmark of simpler's host glue "
          f"(torch threads={torch.get_num_threads()}, {REPS} reps)\n")
    for direction, fn in (("forward", forward_glue), ("backward", backward_glue)):
        print(f"--- {direction} (ms per CALL, summed over ranks) ---")
        for (P, L, C, D) in CONFIGS:
            parts = fn(P, L, C, D, D)
            tot = sum(parts.values())
            det = "  ".join(f"{k}={v:.3f}" for k, v in parts.items())
            print(f"P={P} L={L} C={C} D={D}  TOTAL={tot:.3f}ms   {det}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
