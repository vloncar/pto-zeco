#!/usr/bin/env python3
"""B4 pre-flight: torch emulation of the *exact* op sequence the pypto backward kernels emit.

`gla.common.gla_chunk_backward` already validates the backward MATH. What it does not
validate is the **restructuring** the DSL forces on us, which is where a silent error would
come from:

  * `gamma` is a `[dk,1]` row-scalar, so `k * (gamma/b)` (a *column*-broadcast over `[C,dk]`)
    is not directly expressible. Every use is refactored to push `gamma` onto the `[dk,dv]`
    state instead, exactly as the forward's F3.1 rewrite did:
        dV_h    = (k/b) @ (gamma*dSloc)          instead of  (k*gamma/b) @ dSloc
        dK_h    = (v @ (gamma*dSloc)^T) / b      instead of  (v @ dSloc^T) * (gamma/b)
    and the three `dgamma` terms are formed already-multiplied by gamma.
  * `db[n][-1] += dgamma_n` (a single-row update) becomes a whole-tile `col_expand_add`
    AFTER the reverse cumulative sum: row `C-1` is `>= t` for every `t`, so its contribution
    to `reverse_cumsum` is the same constant in every row.
  * the reverse cumsum itself is `triu @ dgcs` (a matmul), not a scan.
  * the gate gradient is carried in the LOG domain (`dgcs = db * b`), so no `db` ever exists.
  * the reverse-ring message is `dS_recv[p] + gamma[p]*d[p]`, which IS `d[p-1]` — so unlike
    the standalone AllScan backward, no host-side `g_out` shuffle is needed and `out[p-1]`
    is device-local (it is this rank's own `S_recv`).

Run: python3 devtools/b4_math_check.py
"""

from __future__ import annotations

import sys

import torch

from gla.common import expected_gla_backward, flatten_seq, make_gla_inputs


# ---------------------------------------------------------------------------
# Kernel A — forward recompute: per-chunk state/decay snapshots + S_total.
# ---------------------------------------------------------------------------

def k_recompute(Kp, Vp, Ap, tril, C):
    """InCore #1. Snapshots are stored [N*dk, dv] / [N*dk, 1] (a tile is 2D)."""
    L, dk = Kp.shape
    dv = Vp.shape[1]
    N = L // C
    Ssnap = torch.zeros(N * dk, dv)
    Cprev = torch.zeros(N * dk, 1)
    s_run = torch.zeros(dk, dv)          # pl.tile.full([DK,DV], 0.0)
    c_run = torch.ones(dk, 1)            # pl.tile.full([DK,1], 1.0)
    for n in range(N):
        off, soff = n * C, n * dk
        k, v, a = Kp[off:off + C], Vp[off:off + C], Ap[off:off + C]
        la = torch.log(a)
        b = torch.exp(tril @ la)                                   # [C,dk]
        gamma = torch.exp(la.sum(0).reshape(dk, 1))                # col_sum -> reshape
        Ssnap[soff:soff + dk] = s_run                              # BEFORE the update
        Cprev[soff:soff + dk] = c_run
        kb = k / b
        kv = kb.t() @ v
        s_run = (s_run + kv) * gamma                               # row_expand_mul
        c_run = c_run * gamma
    return Ssnap, Cprev, s_run


# ---------------------------------------------------------------------------
# Kernel B — grad_o: output-stage adjoints, per chunk, forward order.
# ---------------------------------------------------------------------------

def k_grad_o(Qp, Kp, Vp, Ap, dOp, tril, C, Ssnap, Cprev, Srecv):
    """InCore #2. Carries only the `dS_recv` accumulator across chunks."""
    L, dk = Qp.shape
    dv = Vp.shape[1]
    N = L // C
    dQ = torch.zeros(L, dk)
    dKo = torch.zeros(L, dk)
    dVo = torch.zeros(L, dv)
    dgcso = torch.zeros(L, dk)
    dH = torch.zeros(N * dk, dv)
    dcprev = torch.zeros(N * dk, 1)
    acc = torch.zeros(dk, dv)            # pl.tile.full
    for n in range(N):
        off, soff = n * C, n * dk
        q, k, v, a = Qp[off:off + C], Kp[off:off + C], Vp[off:off + C], Ap[off:off + C]
        do = dOp[off:off + C]
        la = torch.log(a)
        b = torch.exp(tril @ la)
        qt = q * b
        kb = k / b
        scores = (qt @ kb.t()) * tril
        Sprev = Ssnap[soff:soff + dk]
        cprev = Cprev[soff:soff + dk]
        H = Sprev + Srecv * cprev                                  # row_expand_mul
        dQt = do @ H.t()
        dH_n = qt.t() @ do
        dH[soff:soff + dk] = dH_n
        dcprev[soff:soff + dk] = (dH_n * Srecv).sum(dim=1, keepdim=True)   # row_sum
        acc = acc + dH_n * cprev                                   # row_expand_mul
        dsc = (do @ v.t()) * tril
        dVo[off:off + C] = scores.t() @ do
        dQt = dQt + dsc @ kb
        dKintra = dsc.t() @ qt
        dQ_n = dQt * b
        dKo_n = dKintra / b
        dQ[off:off + C] = dQ_n
        dKo[off:off + C] = dKo_n
        dgcso[off:off + C] = dQ_n * q - dKo_n * k                  # log-domain gate grad
    return dQ, dKo, dVo, dgcso, dH, dcprev, acc


# ---------------------------------------------------------------------------
# Kernel C — grad_h + gate: reverse chunk recurrence, then dA.
# ---------------------------------------------------------------------------

def k_grad_h(Kp, Vp, Ap, tril, triu, C, Ssnap, Cprev, dH, dcprev,
             dKo, dVo, dgcso, dStot, dgam):
    """InCore #3. Reverse loop is `for m in range(N): n = N-1-m` (pl.range is forward-only)."""
    L, dk = Kp.shape
    dv = Vp.shape[1]
    N = L // C
    dK = torch.zeros(L, dk)
    dV = torch.zeros(L, dv)
    dA = torch.zeros(L, dk)
    dSloc = dStot.clone()
    dcvec = dgam.clone()
    for m in range(N):
        n = N - 1 - m
        off, soff = n * C, n * dk
        k, v, a = Kp[off:off + C], Vp[off:off + C], Ap[off:off + C]
        la = torch.log(a)
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0).reshape(dk, 1))
        Sprev = Ssnap[soff:soff + dk]
        cprev = Cprev[soff:soff + dk]

        # Push gamma onto the [dk,dv] state ONCE; every gamma-scaled quantity below
        # is then a plain matmul/elementwise op with no column broadcast.
        dSloc_p = dSloc * gamma                                    # row_expand_mul
        dcvec_p = dcvec * gamma
        kb = k / b
        dV_h = kb @ dSloc_p                                        # == Kstate @ dSloc
        dK_h = (v @ dSloc_p.t()) / b                               # == dKstate * gamma/b
        dgcs_h = -(dK_h * k)                                       # == -dKstate*Kstate

        # The three dgamma terms, each already carrying its gamma factor.
        c1 = (dSloc_p * Sprev).sum(dim=1, keepdim=True).reshape(1, dk)   # row_sum
        c2 = (dcvec_p * cprev).reshape(1, dk)
        c3 = (dK_h * k).sum(dim=0, keepdim=True)                   # col_sum -> [1,dk]
        corr = c1 + c2 + c3

        dgcs = dgcso[off:off + C] + dgcs_h
        rcs = triu @ dgcs                                          # reverse cumulative sum
        dA[off:off + C] = (rcs + corr) / a                         # col_expand_add
        dK[off:off + C] = dKo[off:off + C] + dK_h
        dV[off:off + C] = dVo[off:off + C] + dV_h

        # Advance. At n == 0 the result is dead, so the `n > 0` guard is dropped
        # (a conditional inside the InCore loop would be far more expensive).
        dSloc = dSloc_p + dH[soff:soff + dk]
        dcvec = dcvec_p + dcprev[soff:soff + dk]
    return dK, dV, dA


# ---------------------------------------------------------------------------
# The reverse ring (device-local `out[p-1]`, fused message).
# ---------------------------------------------------------------------------

def ring_backward(dSrecv, gammas, Srecvs, P, dk, dv):
    """d[p] = dS_recv[p+1] + gamma[p+1]*d[p+1]; the message from p to p-1 IS d[p-1]."""
    dStot = torch.zeros(P, dk, dv)
    dgam = torch.zeros(P, dk, 1)
    msg = None
    for p in reversed(range(P)):
        if p == P - 1:                       # source: d = 0, nothing downstream of out[P-1]
            d_p = torch.zeros(dk, dv)
            dStot[p] = d_p
            dgam[p] = 0.0
        else:
            d_p = msg
            dStot[p] = d_p
            dgam[p] = (d_p * Srecvs[p]).sum(dim=1, keepdim=True)   # out[p-1] == S_recv[p]
        if p == 0:
            dgam[p] = 0.0                    # gamma[0] is unused
            break
        msg = dSrecv[p] + d_p * gammas[p]    # == d[p-1]
    return dStot, dgam


def run(P, L, C, dk, dv, seed):
    Q, K, V, A = make_gla_inputs(P, L, dk, dv, seed=seed)
    torch.manual_seed(seed + 991)
    dO = torch.randn(P, L, dv)
    tril = torch.tril(torch.ones(C, C))
    triu = torch.triu(torch.ones(C, C))

    # --- phase 1: recompute (all ranks) ---
    snaps = [k_recompute(K[p], V[p], A[p], tril, C) for p in range(P)]
    S_tot = [s[2] for s in snaps]
    gammas = [A[p].prod(dim=0).reshape(dk, 1) for p in range(P)]

    # --- phase 2: forward ring -> S_recv per rank ---
    Srecvs, run_out = [], torch.zeros(dk, dv)
    for p in range(P):
        Srecvs.append(torch.zeros(dk, dv) if p == 0 else run_out.clone())
        run_out = S_tot[p] + gammas[p] * run_out

    # --- phase 3: grad_o (all ranks) ---
    go = [k_grad_o(Q[p], K[p], V[p], A[p], dO[p], tril, C,
                   snaps[p][0], snaps[p][1], Srecvs[p]) for p in range(P)]
    dSrecv = torch.stack([g[6] for g in go])

    # --- phase 4: reverse ring ---
    dStot, dgam = ring_backward(dSrecv, gammas, Srecvs, P, dk, dv)

    # --- phase 5: grad_h + gate ---
    dQ = torch.zeros(P, L, dk)
    dKf = torch.zeros(P, L, dk)
    dVf = torch.zeros(P, L, dv)
    dAf = torch.zeros(P, L, dk)
    for p in range(P):
        dQp, dKo, dVo, dgcso, dH, dcprev, _ = go[p]
        dKp, dVp, dAp = k_grad_h(K[p], V[p], A[p], tril, triu, C,
                                 snaps[p][0], snaps[p][1], dH, dcprev,
                                 dKo, dVo, dgcso, dStot[p], dgam[p])
        dQ[p], dKf[p], dVf[p], dAf[p] = dQp, dKp, dVp, dAp

    gQ, gK, gV, gA = expected_gla_backward(
        flatten_seq(Q), flatten_seq(K), flatten_seq(V), flatten_seq(A), flatten_seq(dO))
    errs = [(flatten_seq(dQ) - gQ).abs().max().item(),
            (flatten_seq(dKf) - gK).abs().max().item(),
            (flatten_seq(dVf) - gV).abs().max().item(),
            (flatten_seq(dAf) - gA).abs().max().item()]
    scale = max(g.abs().max().item() for g in (gQ, gK, gV, gA))
    return errs, scale


def main():
    cases = [(1, 64, 16, 16, 16), (1, 128, 32, 32, 32), (2, 64, 16, 16, 16),
             (2, 128, 32, 32, 32), (2, 128, 32, 64, 32), (2, 128, 32, 32, 64),
             (2, 128, 64, 64, 64), (4, 128, 32, 32, 32), (4, 256, 32, 32, 32),
             (4, 128, 16, 32, 16), (3, 96, 32, 32, 32)]
    bad = 0
    for (P, L, C, dk, dv) in cases:
        errs, scale = run(P, L, C, dk, dv, seed=1234)
        ok = max(errs) < 1e-3 * max(scale, 1.0)
        bad += not ok
        print(f"  {'ok   ' if ok else 'WRONG'} P={P} L={L:4d} C={C:3d} dk={dk:3d} dv={dv:3d}"
              f"  dQ {errs[0]:.2e}  dK {errs[1]:.2e}  dV {errs[2]:.2e}  dA {errs[3]:.2e}"
              f"   (|g|max {scale:.1f})")
    print(f"\n{len(cases) - bad}/{len(cases)} cases match expected_gla_backward")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
