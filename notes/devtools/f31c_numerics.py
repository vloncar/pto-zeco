#!/usr/bin/env python3
"""F3.1c: is the `C > D` error a CONDITIONING problem rather than a codegen bug?

No device, no pypto — pure torch. The chunk kernel reconstructs the intra-chunk term as

    qt = q * b,  kb = k / b,  scores = (qt @ kb^T) * tril

where ``b`` is the within-chunk cumulative decay product. ``b`` shrinks monotonically down
the chunk, so ``k / b`` grows, and the two only cancel in exact arithmetic. The dynamic
range of that cancellation grows with **C**, not with D — at C=64 the exponent span is
twice the C=32 one.

Two things this settles that the hardware runs cannot:

  1. **Is the reference itself unstable at this shape?** Evaluating the identical chunk math
     in fp32 and in fp64 and comparing tells us how much of any observed error the
     algorithm owns before a kernel is involved.
  2. **Are the shapes even comparable?** ``make_gla_inputs`` seeds by shape, so C=64/dk=32
     and C=64/dk=64 do NOT see the same numbers. A shape that passes may simply have drawn
     a gentler decay. This prints the per-shape conditioning so that "C=64,D=64 passes"
     can be read as evidence rather than assumed to be one.

Usage: python3 devtools/f31c_numerics.py
"""

from __future__ import annotations

import torch

from gla.common import expected_gla, flatten_seq, make_gla_inputs

CASES = [
    (128, 64, 32, 32, "FAILS on HW"),
    (128, 64, 64, 64, "passes"),
    (128, 64, 64, 32, "?"),
    (128, 64, 32, 64, "?"),
    (128, 32, 64, 64, "passes"),
    (128, 32, 32, 32, "passes"),
    (64, 64, 32, 32, "N=1"),
]


def chunk_forward(Q, K, V, A, C, dtype):
    """The kernel's own chunk recurrence, evaluated in `dtype`."""
    q_, k_, v_, a_ = (t.to(dtype) for t in (Q, K, V, A))
    L, dk = q_.shape
    dv = v_.shape[1]
    tril = torch.tril(torch.ones(C, C, dtype=dtype))
    s = torch.zeros(dk, dv, dtype=dtype)
    out = torch.zeros(L, dv, dtype=dtype)
    stats = {"b_min": float("inf"), "kb_max": 0.0, "scores_max": 0.0}
    for n in range(L // C):
        sl = slice(n * C, (n + 1) * C)
        q, k, v, a = q_[sl], k_[sl], v_[sl], a_[sl]
        la = torch.log(a)
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(dim=0)).reshape(dk, 1)
        qt, kb = q * b, k / b
        scores = (qt @ kb.t()) * tril
        out[sl] = qt @ s + scores @ v
        s = (s + kb.t() @ v) * gamma
        stats["b_min"] = min(stats["b_min"], b.abs().min().item())
        stats["kb_max"] = max(stats["kb_max"], kb.abs().max().item())
        stats["scores_max"] = max(stats["scores_max"], scores.abs().max().item())
    return out, stats


def main() -> int:
    print(f"{'shape':<26} {'HW':<12} {'fp32-vs-fp64':<14} {'vs golden':<12} "
          f"{'min b':<10} {'max k/b':<10} {'max |O|'}")
    print("-" * 104)
    for (L, C, dk, dv, verdict) in CASES:
        Qa, Ka, Va, Aa = make_gla_inputs(1, L, dk, dv)
        Q, K, V, A = Qa[0], Ka[0], Va[0], Aa[0]

        o32, st = chunk_forward(Q, K, V, A, C, torch.float32)
        o64, _ = chunk_forward(Q, K, V, A, C, torch.float64)
        golden = expected_gla(flatten_seq(Qa), flatten_seq(Ka),
                              flatten_seq(Va), flatten_seq(Aa)).reshape(L, dv)

        self_err = (o32 - o64.to(torch.float32)).abs().max().item()
        gold_err = (o32 - golden).abs().max().item()
        tag = f"L={L} C={C} dk={dk} dv={dv}"
        print(f"{tag:<26} {verdict:<12} {self_err:<14.3e} {gold_err:<12.3e} "
              f"{st['b_min']:<10.2e} {st['kb_max']:<10.2e} {o32.abs().max().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
