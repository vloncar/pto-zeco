#!/usr/bin/env python3
"""Task 5 pre-flight: is the DK-blocked GLA chunk scan the same arithmetic?

Before writing a line of DSL, check the decomposition in torch. Two matmuls contract over
the head dim (`qt @ S` and `qt @ kb^T`), so blocking DK turns each into a sum of partials;
everything else is column-independent. If this does not hold in torch it cannot hold on a
device, and a kernel mismatch later would be ambiguous between "blocking is wrong" and
"the DSL did something unexpected".
"""
import itertools
import torch


def unblocked(q, k, v, a, tril, S):
    N, C = a.shape[0] // tril.shape[0], tril.shape[0]
    O = torch.zeros(a.shape[0], v.shape[1], dtype=torch.float64)
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        qt, kb = q[s] * b, k[s] / b
        scores = (qt @ kb.T) * tril
        O[s] = qt @ S + scores @ v[s]
        S = gamma * (S + kb.T @ v[s])
    return O, S


def blocked(q, k, v, a, tril, S, NB):
    N, C = a.shape[0] // tril.shape[0], tril.shape[0]
    DK = a.shape[1]
    BK = DK // NB
    O = torch.zeros(a.shape[0], v.shape[1], dtype=torch.float64)
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        scores_acc = torch.zeros(C, C, dtype=torch.float64)
        o_inter = torch.zeros(C, v.shape[1], dtype=torch.float64)
        S_new = S.clone()
        for j in range(NB):
            d = slice(j * BK, (j + 1) * BK)
            la_b = torch.log(a[s, d])
            b_b = torch.exp(tril @ la_b)
            gamma_b = torch.exp(la_b.sum(0)).reshape(-1, 1)
            qt_b, kb_b = q[s, d] * b_b, k[s, d] / b_b
            scores_acc += qt_b @ kb_b.T          # [C,C] partial, summed in the vector unit
            o_inter += qt_b @ S[d, :]            # [C,DV] partial, likewise
            S_new[d, :] = gamma_b * (S[d, :] + kb_b.T @ v[s])
        O[s] = o_inter + (scores_acc * tril) @ v[s]
        S = S_new
    return O, S


def main():
    torch.manual_seed(11)
    worst = 0.0
    rows = []
    for C, DK, DV, N, NB in itertools.product([16, 64], [32, 128], [32, 128], [1, 3], [2, 4]):
        if DK % NB:
            continue
        L = N * C
        q = torch.randn(L, DK, dtype=torch.float64)
        k = torch.randn(L, DK, dtype=torch.float64)
        v = torch.randn(L, DV, dtype=torch.float64)
        a = torch.rand(L, DK, dtype=torch.float64) * 0.5 + 0.5   # decays in (0.5, 1)
        tril = torch.tril(torch.ones(C, C, dtype=torch.float64))
        S0 = torch.randn(DK, DV, dtype=torch.float64) * 0.1
        Ou, Su = unblocked(q, k, v, a, tril, S0.clone())
        Ob, Sb = blocked(q, k, v, a, tril, S0.clone(), NB)
        e = max((Ou - Ob).abs().max().item(), (Su - Sb).abs().max().item())
        worst = max(worst, e)
        rows.append((C, DK, DV, N, NB, e))
    for C, DK, DV, N, NB, e in rows:
        flag = "" if e < 1e-10 else "   <-- MISMATCH"
        print(f"C={C:>3} DK={DK:>3} DV={DV:>3} N={N} NB={NB}  max|diff| = {e:.3e}{flag}")
    print(f"\n{len(rows)} cases, worst {worst:.3e}")
    return 0 if worst < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
