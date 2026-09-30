#!/usr/bin/env python3
"""F3.1c: find WHICH value in gla_stage2 first goes wrong at C=64, dk=dv=32 on hardware.

Runs a standalone copy of the stage2 chunk kernel (no ring, no distribution, no stage1 —
P=1 already isolates the failure to stage2) and stores ONE named intermediate to a GM
output alongside the real output, then compares that intermediate against a torch
computation of the same quantity.

**One tap per build, on purpose.** Storing every intermediate at once changes the tile
liveness and hence the whole allocation, so the probe would no longer be running the
program under investigation. Each tap adds a single extra store to the original kernel.
Even so, treat a tap that "fixes" the output as information, not noise: it means the value
is allocation-sensitive.

Taps run in dataflow order, so the first one that disagrees with torch is the first wrong
value:

    la -> b -> gamma -> qt -> kb -> kbt -> scores -> o_intra -> o_inter -> kv -> s_new

Usage: python3 devtools/f31c_stage2_probe.py <device> <platform> <L> <C> <dk> <dv> [tap ...]
       (no tap list = run them all, in order, stopping at the first mismatch)
"""

from __future__ import annotations

import sys

import torch

import pypto.language as pl

TAPS = ["la", "b", "gamma", "qt", "kb", "kbt", "scores", "o_intra", "o_inter", "kv", "s_new"]


def tap_shape(tap: str, C: int, dk: int, dv: int) -> list[int]:
    return {
        "la": [C, dk], "b": [C, dk], "gamma": [dk, 1], "qt": [C, dk], "kb": [C, dk],
        "kbt": [dk, C], "scores": [C, C], "o_intra": [C, dv], "o_inter": [C, dv],
        "kv": [dk, dv], "s_new": [dk, dv],
    }[tap]


_KERNEL_SRC = '''
@pl.program
class Stage2Probe:
    @pl.function(type=pl.FunctionType.InCore)
    def stage2(
        self,
        Q: pl.Tensor[[{L}, {DK}], pl.FP32],
        Kmat: pl.Tensor[[{L}, {DK}], pl.FP32],
        Vmat: pl.Tensor[[{L}, {DV}], pl.FP32],
        A: pl.Tensor[[{L}, {DK}], pl.FP32],
        tril: pl.Tensor[[{C}, {C}], pl.FP32],
        Srecv: pl.Tensor[[{DK}, {DV}], pl.FP32],
        O: pl.Out[pl.Tensor[[{L}, {DV}], pl.FP32]],
        T: pl.Out[pl.Tensor[[{TROWS}, {TC}], pl.FP32]],
    ) -> pl.Tensor[[{L}, {DV}], pl.FP32]:
        tril_t = pl.load(tril, [0, 0], [{C}, {C}])
        s_init = pl.load(Srecv, [0, 0], [{DK}, {DV}])
        out = O
        tapped = T
        for n, (s_run,) in pl.range(0, {N}, init_values=(s_init,)):
            off = n * {C}
            q = pl.load(Q, [off, 0], [{C}, {DK}])
            k = pl.load(Kmat, [off, 0], [{C}, {DK}])
            v = pl.load(Vmat, [off, 0], [{C}, {DV}])
            a = pl.load(A, [off, 0], [{C}, {DK}])
            la = pl.log(a)
            b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
            gamma = pl.exp(pl.tile.reshape(pl.tile.col_sum(la), [{DK}, 1]))
            qt = pl.mul(q, b)
            kb = pl.div(k, b)
            kbt = pl.transpose(kb, 0, 1)
            scores = pl.mul(pl.matmul(qt, kbt, out_dtype=pl.FP32), tril_t)
            o_intra = pl.matmul(scores, v, out_dtype=pl.FP32)
            s_run_v = pl.mul(s_run, 1.0)
            o_inter = pl.matmul(qt, s_run_v, out_dtype=pl.FP32)
            o_n = pl.add(o_inter, o_intra)
            out = pl.store(o_n, [off, 0], out)
            kv = pl.matmul(kbt, v, out_dtype=pl.FP32)
            s_new = pl.tile.row_expand_mul(pl.add(s_run, kv), gamma)
            tapped = pl.store({TAP}, [n * {TR}, 0], tapped)
            s_run = pl.yield_(s_new)
        return out
'''


def build_probe(L: int, C: int, dk: int, dv: int, tap: str):
    """stage2, plus a store of `tap` (every chunk, stacked) into its own output tensor.

    The kernel source is generated per tap rather than indexing `locals()`, because pypto's
    frontend PARSES the function source — it is not a Python tracer. `locals()[tap]` compiles
    as far as the parser and then fails there ("1 iteration arguments but 0 return
    variables"), which reads like a loop-structure error and is not one.
    """
    N = L // C
    TR, TC = tap_shape(tap, C, dk, dv)
    # Every chunk's value is stored, stacked down the rows. A conditional store ("tap only
    # chunk 0") is not expressible here: it makes the tapped tensor conditionally defined
    # inside the loop, and pypto rejects the body with "1 iteration arguments but 0 return
    # variables". Per-chunk data is more useful anyway — it shows whether the corruption
    # starts at chunk 0 or only after the carry has gone round once.
    src = _KERNEL_SRC.format(L=L, C=C, DK=dk, DV=dv, N=N, TR=TR, TC=TC, TROWS=TR * N, TAP=tap)
    return pl.parse(src)


def torch_ref(tap, Q, K, V, A, tril, C, dk, dv):
    """The tapped quantity for every chunk, stacked down the rows like the kernel stores it."""
    L = Q.shape[0]
    s_run = torch.zeros(dk, dv)
    per_chunk = []
    for n in range(L // C):
        sl = slice(n * C, (n + 1) * C)
        q, k, v, a = Q[sl], K[sl], V[sl], A[sl]
        la = torch.log(a)
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(dim=0)).reshape(dk, 1)
        qt = q * b
        kb = k / b
        kbt = kb.t()
        scores = (qt @ kbt) * tril
        o_intra = scores @ v
        o_inter = qt @ s_run
        kv = kbt @ v
        s_new = (s_run + kv) * gamma
        per_chunk.append(dict(
            la=la, b=b, gamma=gamma, qt=qt, kb=kb, kbt=kbt, scores=scores,
            o_intra=o_intra, o_inter=o_inter, kv=kv, s_new=s_new)[tap])
        s_run = s_new
    return torch.cat(per_chunk, dim=0)


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2]
    L, C, dk, dv = (int(v) for v in sys.argv[3:7])
    taps = sys.argv[7:] or TAPS

    from pypto import ir
    from pypto.runtime.runner import RunConfig

    from gla.common import make_gla_inputs

    torch.manual_seed(1234)
    Qa, Ka, Va, Aa = make_gla_inputs(1, L, dk, dv)
    Q, K, V, A = Qa[0], Ka[0], Va[0], Aa[0]
    tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
    zero = torch.zeros(dk, dv, dtype=torch.float32)

    N = L // C
    golden_O = None
    print(f"=== L={L} C={C} dk={dk} dv={dv} (N={N}) on {platform} dev {device}")
    print(f"{'tap':<9} {'shape':<10} {'rel err':<11} {'per chunk':<26} out-vs-torch")
    print("-" * 78)
    worst = None
    for tap in taps:
        TR, TC = tap_shape(tap, C, dk, dv)
        try:
            compiled = ir.compile(build_probe(L, C, dk, dv, tap), platform=platform)
            O = torch.zeros(L, dv, dtype=torch.float32)
            T = torch.zeros(TR * N, TC, dtype=torch.float32)
            compiled(Q, K, V, A, tril, zero, O, T,
                     config=RunConfig(platform=platform, device_id=device))
        except Exception as exc:  # noqa: BLE001 - a build/run failure is a result here
            print(f"  {tap:<8} ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
            continue

        ref = torch_ref(tap, Q, K, V, A, tril, C, dk, dv)
        scale = max(ref.abs().max().item(), 1e-30)
        rel = (T - ref).abs().max().item() / scale
        per_chunk = " ".join(
            f"{(T[n * TR:(n + 1) * TR] - ref[n * TR:(n + 1) * TR]).abs().max().item() / scale:.1e}"
            for n in range(N))

        # The kernel's real output, on the same build. If a tap changes whether O itself is
        # wrong, that tap perturbed the allocation — say so rather than trusting the tap.
        if golden_O is None:
            golden_O = torch_ref_out(Q, K, V, A, tril, C, dk, dv)
        o_rel = (O - golden_O).abs().max().item() / max(golden_O.abs().max().item(), 1e-30)

        flag = ""
        if rel >= 1e-3 and worst is None:
            worst, flag = tap, "  <<< FIRST WRONG"
        elif rel >= 1e-3:
            flag = "  (wrong)"
        print(f"{tap:<9} [{TR},{TC}]{'':<3} {rel:<11.2e} {per_chunk:<26} "
              f"{o_rel:.2e}{'  O OK' if o_rel < 1e-3 else '  O WRONG'}{flag}")

    print(f"\nfirst wrong value: {worst or 'none — every tapped value matches torch'}")
    return 0


def torch_ref_out(Q, K, V, A, tril, C, dk, dv):
    """The kernel's real output O, in torch."""
    L = Q.shape[0]
    s_run = torch.zeros(dk, dv)
    out = torch.zeros(L, V.shape[1])
    for n in range(L // C):
        sl = slice(n * C, (n + 1) * C)
        q, k, v, a = Q[sl], K[sl], V[sl], A[sl]
        la = torch.log(a)
        b = torch.exp(tril @ la)
        gamma = torch.exp(la.sum(dim=0)).reshape(dk, 1)
        qt, kb = q * b, k / b
        out[sl] = qt @ s_run + ((qt @ kb.t()) * tril) @ v
        s_run = (s_run + kb.t() @ v) * gamma
    return out


if __name__ == "__main__":
    raise SystemExit(main())
