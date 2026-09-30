#!/usr/bin/env python3
"""Can stage2 drop its state CARRY entirely, the way pto-kernels' chunk_o does?

Three copies of the [dk,dv] state -- the chunk carry, its detached copy, and the block
assembly target -- are 60% of the vector budget at dk=dv=128, and blocking the head dim
touches none of them. pto-kernels' KDA chunk_o has no carry at all: "chunks within a work item
are fully independent (each reads its own s_snapshots entry)".

The recurrence is LINEAR in the state:

    S_n = gamma_n * (S_{n-1} + kv_n)

so starting from a boundary B instead of 0 just adds a decayed copy of B:

    S_n(B) = G_n * B + S_n(0)        with  G_n = prod_{m<=n} gamma_m

stage1 already walks the chunks from zero. If it also stored its per-chunk snapshot
``S_n(0)`` and the running decay ``G_n``, stage2 could RECONSTRUCT the state it needs for each
chunk from the boundary it received -- no carry, and the chunks become independent.

This is the same trick the cross-device AllScan uses, applied one level down.

Checks that identity in double precision across shapes before any DSL work.
"""
import itertools
import torch


def carried(k, v, a, tril, B):
    """What stage2 does today: carry the state across chunks from the boundary."""
    C = tril.shape[0]
    N = a.shape[0] // C
    S = B.clone()
    states = []
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        states.append(S.clone())          # the state chunk n's OUTPUT uses
        S = gamma * (S + (k[s] / b).T @ v[s])
    return states, S


def reconstructed(k, v, a, tril, B):
    """What stage1 would store, and what stage2 would rebuild from it."""
    C = tril.shape[0]
    N = a.shape[0] // C
    S0 = torch.zeros_like(B)              # stage1 walks from zero
    G = torch.ones(B.shape[0], 1, dtype=B.dtype)
    snaps, decays = [], []
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        snaps.append(S0.clone())          # S_n(0)
        decays.append(G.clone())          # G_n
        S0 = gamma * (S0 + (k[s] / b).T @ v[s])
        G = G * gamma
    # stage2 rebuilds each chunk's state from the boundary, with no carry
    return [d * B + sn for sn, d in zip(snaps, decays)], G * B + S0


def main():
    torch.manual_seed(3)
    worst = 0.0
    rows = []
    for C, dk, dv, N in itertools.product([16, 64], [32, 128], [32, 128], [1, 4, 7]):
        L = N * C
        k = torch.randn(L, dk, dtype=torch.float64)
        v = torch.randn(L, dv, dtype=torch.float64)
        a = torch.rand(L, dk, dtype=torch.float64) * 0.4 + 0.6
        tril = torch.tril(torch.ones(C, C, dtype=torch.float64))
        B = torch.randn(dk, dv, dtype=torch.float64) * 0.1
        sc, fc = carried(k, v, a, tril, B)
        sr, fr = reconstructed(k, v, a, tril, B)
        e = max(max((x - y).abs().max().item() for x, y in zip(sc, sr)),
                (fc - fr).abs().max().item())
        worst = max(worst, e)
        rows.append((C, dk, dv, N, e))
    for C, dk, dv, N, e in rows:
        flag = "" if e < 1e-10 else "   <-- MISMATCH"
        print(f"C={C:>3} dk={dk:>3} dv={dv:>3} N={N}  max|diff| = {e:.3e}{flag}")
    print(f"\n{len(rows)} cases, worst {worst:.3e}")
    return 0 if worst < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
