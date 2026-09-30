#!/usr/bin/env python3
"""A3: which matmul operands can GLA afford in fp16 (or bf16), and which cannot?

The L0 operand buffers are 64 KB each, so a [128,128] fp32 tile IS the whole buffer and
`C=128` cannot double-buffer. Narrower operands fix that -- but only for operands whose
DYNAMIC RANGE fits. GLA divides by the within-chunk cumulative decay `b`, and `b` is an
exponential of a running sum, so `k/b` can be astronomically large while `q*b` underflows.
Those are not flash-attention's operands and they do not get flash-attention's answer.

This does the arithmetic in torch first (fp64 golden, fp32 baseline, then each matmul's
operands rounded to the narrow type before the product) so the kernel work is aimed at the
casts that are actually safe. Reports the error each cast costs AND the range it needs.

Usage: a3_precision_study.py [C dk dv N] [--dtype fp16|bf16]
"""
from __future__ import annotations

import sys

import torch

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.104e-5


def chunked_gla(q, k, v, a, C, *, narrow=(), dt=torch.float16, ranges=None):
    """The kernel's algorithm, with `narrow` naming which matmul operands to round.

    Names match the kernel: 'tril'/'la' (decay), 'qt'/'kbt' (scores), 'qt2'/'s' (state read),
    'scores'/'v' (within-chunk output), 'kbt2'/'v2' (stage1 state update).
    """
    L, dk = q.shape
    dv = v.shape[1]
    N = L // C
    tril = torch.tril(torch.ones(C, C, dtype=q.dtype))

    def rnd(x, name):
        if ranges is not None:
            finite = x[torch.isfinite(x)].abs()
            nz = finite[finite > 0]
            lo = nz.min().item() if nz.numel() else 0.0
            ranges.setdefault(name, [float("inf"), 0.0])
            ranges[name][0] = min(ranges[name][0], lo)
            ranges[name][1] = max(ranges[name][1], finite.max().item() if finite.numel() else 0.0)
        return x.to(dt).to(x.dtype) if name in narrow else x

    O = torch.zeros(L, dv, dtype=q.dtype)
    S = torch.zeros(dk, dv, dtype=q.dtype)
    for n in range(N):
        s = slice(n * C, (n + 1) * C)
        la = torch.log(a[s])
        b = torch.exp(rnd(tril, "tril") @ rnd(la, "la"))
        gamma = torch.exp(la.sum(0)).reshape(-1, 1)
        qt, kb = q[s] * b, k[s] / b
        scores = (rnd(qt, "qt") @ rnd(kb.T, "kbt")) * tril
        O[s] = rnd(qt, "qt2") @ rnd(S, "s") + rnd(scores, "scores") @ rnd(v[s], "v")
        S = gamma * (S + rnd(kb.T, "kbt2") @ rnd(v[s], "v2"))
    return O


CASTS = [
    ("baseline fp32",              ()),
    ("tril only",                  ("tril",)),
    ("decay matmul (tril+la)",     ("tril", "la")),
    ("output matmul (scores+v)",   ("scores", "v")),
    ("score matmul (qt+kbt)",      ("qt", "kbt")),
    ("state read (qt2+s)",         ("qt2", "s")),
    ("stage1 update (kbt2+v2)",    ("kbt2", "v2")),
    ("SAFE SET: tril+la+scores+v", ("tril", "la", "scores", "v")),
    ("everything",                 ("tril", "la", "qt", "kbt", "qt2", "s",
                                    "scores", "v", "kbt2", "v2")),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dt = torch.bfloat16 if "--dtype" in sys.argv and "bf16" in sys.argv else torch.float16
    C, dk, dv, N = (int(x) for x in args[:4]) if len(args) >= 4 else (64, 128, 128, 4)
    L = N * C
    torch.manual_seed(11)
    q = torch.randn(L, dk, dtype=torch.float64)
    k = torch.randn(L, dk, dtype=torch.float64)
    v = torch.randn(L, dv, dtype=torch.float64)
    a = torch.rand(L, dk, dtype=torch.float64) * 0.4 + 0.6

    gold = chunked_gla(q, k, v, a, C)                      # fp64, no rounding
    q32, k32, v32, a32 = (t.float() for t in (q, k, v, a))

    ranges: dict[str, list[float]] = {}
    base = chunked_gla(q32, k32, v32, a32, C, ranges=ranges)
    scale = gold.abs().max().item()
    print(f"C={C} dk={dk} dv={dv} N={N}  narrow type = "
          f"{'bf16' if dt is torch.bfloat16 else 'fp16'}   |O|max = {scale:.3g}")
    print()
    print(f"  {'what is rounded':<30} {'max |err|':>11}  {'rel':>9}")
    for name, narrow in CASTS:
        out = chunked_gla(q32, k32, v32, a32, C, narrow=narrow, dt=dt)
        e = (out - gold).abs().max().item()
        print(f"  {name:<30} {e:>11.3e}  {e / scale:>9.2e}"
              + ("   <-- fp32 reference" if not narrow else ""))
    print()
    print("  operand dynamic range in fp32 (min non-zero .. max |x|); fp16 holds 6.1e-05 .. 6.6e+04")
    for nm in ("tril", "la", "qt", "kbt", "s", "scores", "v"):
        if nm in ranges:
            lo, hi = ranges[nm]
            bad = []
            if hi > FP16_MAX:
                bad.append("OVERFLOWS fp16")
            if 0 < lo < FP16_MIN_NORMAL:
                bad.append("underflows fp16 normals")
            print(f"    {nm:<8} {lo:>10.2e} .. {hi:<10.2e} {'  ** ' + ', '.join(bad) if bad else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
