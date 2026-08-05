"""F2 Phase 2 step (a) — split the end-to-end error into stage1 / ring / stage2.

Runs the *instrumented* fused forward (``f2_diag_prog``) which exposes S_local (stage1's
output), S_recv (the boundary the ring handed stage2) and S_out, then compares each against
the torch reference:

    err_S_local[p]  stage1 wrong?                 S_local_dev[p]  vs  gla_chunk_scan(...).S_total
    err_S_recv[p]   ring wrong?                   S_recv_dev[p]   vs  exclusive AllScan prefix
    err_O_mixed[p]  stage2 wrong GIVEN its input? O_dev[p]        vs  reconstruct(S_recv_DEV[p])
    err_O[p]        end-to-end                    O_dev[p]        vs  reconstruct(S_recv_ref[p])

At P=2 the ring degenerates to a pure copy (out[0] == S_local[0]), so S_recv_dev[1] must equal
S_local_dev[0] *bit-for-bit* — a zero-tolerance check that isolates comm from compute.

Usage: python3 scratchpad/f2_diag.py <device_csv> [C] [P] [N]
"""
from __future__ import annotations

import sys

import torch

from gla.common import gla_chunk_scan, gla_reconstruct, make_gla_inputs

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from f2_diag_prog import build_fused_forward_diag_program  # noqa: E402


def run_diag(P: int, N: int, C: int, devices: list[int], platform: str = "a2a3"):
    dk = dv = C
    L = N * C
    Q, K, V, A = make_gla_inputs(P, L, dk, dv)
    gammas = A.prod(dim=1).reshape(P, dk, 1)

    from pypto import ir
    from pypto.ir.distributed_compiled_program import DistributedConfig

    program = build_fused_forward_diag_program(L, C, dk, dv, 1, P)
    dist_cfg = DistributedConfig(device_ids=devices[:P], num_sub_workers=0)
    compiled = ir.compile(program, platform=platform, distributed_config=dist_cfg)
    assert hasattr(compiled, "prepare"), "diag program must compile distributed"

    # Every shared buffer must exist BEFORE prepare() forks the chip workers.
    def sm(t):
        return t.clone().share_memory_()

    h_Q, h_K, h_V, h_A, h_g = sm(Q), sm(K), sm(V), sm(A), sm(gammas)
    h_tril = sm(torch.tril(torch.ones(C, C)))
    h_mask = sm(torch.tril(torch.ones(C, C)))
    h_ones_cc = sm(torch.ones(C, C))
    h_ones_cdv = sm(torch.ones(C, dv))
    h_zero = sm(torch.zeros(dk, dv))
    h_O = sm(torch.zeros(P, L, dv))
    h_Sloc = sm(torch.zeros(P, dk, dv))
    h_Srecv = sm(torch.zeros(P, dk, dv))
    h_Sout = sm(torch.zeros(P, dk, dv))

    rt = compiled.prepare()
    try:
        rt(h_Q, h_K, h_V, h_A, h_g, h_tril, h_mask, h_ones_cc, h_ones_cdv, h_zero,
           h_O, h_Sloc, h_Srecv, h_Sout)
    finally:
        rt.close()

    return dict(Q=Q, K=K, V=V, A=A, O=h_O.clone(), S_local=h_Sloc.clone(),
                S_recv=h_Srecv.clone(), S_out=h_Sout.clone())


def analyse(res, P, N, C):
    Q, K, V, A = res["Q"], res["K"], res["V"], res["A"]
    dk = dv = C

    # --- torch reference, per rank ---
    scans = [gla_chunk_scan(Q[p], K[p], V[p], A[p], C) for p in range(P)]
    S_total_ref = torch.stack([s[3] for s in scans])          # [P, dk, dv]
    g_total_ref = torch.stack([s[4] for s in scans])          # [P, dk]

    # AllScan: out[p] = S_local[p] + gamma[p] * out[p-1];  S_recv[p] = out[p-1]
    S_recv_ref = torch.zeros(P, dk, dv)
    out_prev = torch.zeros(dk, dv)
    for p in range(P):
        S_recv_ref[p] = out_prev
        out_prev = S_total_ref[p] + g_total_ref[p].unsqueeze(1) * out_prev

    def mx(a, b):
        return (a - b).abs().max().item()

    print(f"\n=== F2 diag: P={P} N={N} C=dk=dv={C} L={N*C} ===")
    print(f"{'rank':>4} {'err_S_local':>12} {'err_S_recv':>12} {'err_O_mixed':>12} {'err_O':>12}")
    rows = []
    for p in range(P):
        S_prev, c_prev, O_intra, _, _ = scans[p]
        srecv_dev = res["S_recv"][p] if p > 0 else torch.zeros(dk, dv)
        O_mixed = gla_reconstruct(Q[p], A[p], C, S_prev, c_prev, srecv_dev, O_intra)
        O_ref = gla_reconstruct(Q[p], A[p], C, S_prev, c_prev, S_recv_ref[p], O_intra)
        e_sl = mx(res["S_local"][p], S_total_ref[p])
        e_sr = mx(srecv_dev, S_recv_ref[p])
        e_om = mx(res["O"][p], O_mixed)
        e_o = mx(res["O"][p], O_ref)
        rows.append((p, e_sl, e_sr, e_om, e_o))
        print(f"{p:>4} {e_sl:>12.3e} {e_sr:>12.3e} {e_om:>12.3e} {e_o:>12.3e}")

    # Per-chunk profile of stage2's own error (O vs reconstruct-from-actual-boundary):
    # a flat/growing profile implicates stage2's loop; a decaying one implicates a bad
    # initial boundary (which err_S_recv would already show).
    print("\nper-chunk |O_dev - O_mixed|max  (stage2 given its ACTUAL boundary):")
    for p in range(P):
        S_prev, c_prev, O_intra, _, _ = scans[p]
        srecv_dev = res["S_recv"][p] if p > 0 else torch.zeros(dk, dv)
        O_mixed = gla_reconstruct(Q[p], A[p], C, S_prev, c_prev, srecv_dev, O_intra)
        prof = [f"{(res['O'][p][n*C:(n+1)*C] - O_mixed[n*C:(n+1)*C]).abs().max().item():.2e}"
                for n in range(N)]
        print(f"  rank {p}: " + " ".join(prof))

    if P == 2:
        # The P=2 ring is a pure copy: out[0] == S_local[0] -> S_recv[1] must match bitwise.
        d = mx(res["S_recv"][1], res["S_local"][0])
        same = torch.equal(res["S_recv"][1], res["S_local"][0])
        print(f"\nP=2 ring copy check: S_recv[1] vs S_local[0] -> max_diff={d:.3e} "
              f"bitwise_equal={same}")

    return rows


if __name__ == "__main__":
    devices = [int(x) for x in sys.argv[1].split(",")]
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    res = run_diag(P, N, C, devices)
    analyse(res, P, N, C)
