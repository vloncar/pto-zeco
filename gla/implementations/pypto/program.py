"""PyPTO DSL programs for the ZeCO / GLA forward (chunk-recurrent InCore form).

The per-device GLA compute is the true chunk-recurrent scan (O(L·C)), expressed as
InCore tile kernels with a ``pl.range`` loop-carry over the ``N = L // C`` chunks —
NOT the earlier O(L²) quadratic form. It composes with the AllScan boundary exchange
at the host level (see :mod:`.impl`):

* ``stage1`` — the local end-of-slice state that feeds AllScan. Carries ``S`` across
  chunks from zero: ``S <- (S ⊙ gamma_n) + (K_n * (gamma_n / b_n))^T @ V_n``, returns
  the final ``S`` (= ``S_total``, local-only). ``b_n = exp(tril @ log A_n)`` is the
  within-chunk cumulative decay; ``gamma_n = prod_t A_n`` the chunk total decay.
* ``stage2`` — the output, given the received boundary state ``S_recv``. Runs the
  SAME recurrence but **initialised from ``S_recv``** (zero for rank 0), so the carried
  ``S`` before chunk ``n`` equals ``S_prev[n] + c_prev[n] ⊙ S_recv`` exactly. Per chunk:
  ``O[n] = (Q_n*b_n) @ S_run  +  ((Q_n*b_n) @ (K_n/b_n)^T ⊙ mask) @ V_n``, then advances
  ``S_run``. Storing ``O[n]`` per chunk and advancing the state fuses reconstruction
  into one pass.

Design notes (all learned on-device):

* **Loop-carry**: a Tile iter_arg carried across ``pl.range(init_values=)`` works when
  every yielded value is an ``add``/vector result (plain layout) — GLA's ``S`` update
  ends in ``add`` so it reconciles with the plain init. A loop-carried tile used as a
  matmul operand must first be detached via ``pl.mul(s, 1.0)`` (a raw iter_arg tile
  stays in ``vec`` and fails ``tmatmul``'s address-space check).
* **No 1-column tiles**: tile cols must be a multiple of 16, so ``gamma`` is broadcast
  two ways via all-ones matmuls — ``g_row_full[C,dk] = exp(ones[C,C] @ log A)`` and
  ``g_full[dk,dv] = exp(log(A)^T @ ones[C,dv])`` — instead of ``[*,1]`` vectors +
  ``row/col_expand_mul``.
* **Simulator vs hardware**: this kernel's per-chunk body is a wide matmul DAG that
  once **deadlocked the a2a3sim scheduler** (rc=-100) while running fine on a2a3
  hardware. That was an upstream CPU-sim bug in the cross-core cube↔vector pipe model
  and has since been **fixed upstream**, so the kernel now runs on both a2a3sim and
  a2a3. (The O(L²) quadratic form kept in git history is no longer needed for sim/CI.)

``@pl.jit`` needs shapes as compile-time constants, so — as before — we source-gen the
kernels into a temp module with ``L``/``C``/``dk``/``dv``/``N`` baked in and import them
back. Each stage is an ``@pl.jit.incore`` tile kernel plus a thin ``@pl.jit`` entry that
dispatches to it (the entry is what :mod:`.impl` calls with a ``RunConfig``).
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

_TEMPLATE = '''
import pypto.language as pl

L = {L}
C = {C}
N = {N}
DK = {DK}
DV = {DV}


@pl.jit.incore
def _stage1_kernel(
    A: pl.Tensor[[L, DK], pl.FP32],
    K: pl.Tensor[[L, DK], pl.FP32],
    V: pl.Tensor[[L, DV], pl.FP32],
    tril: pl.Tensor[[C, C], pl.FP32],
    ones_cc: pl.Tensor[[C, C], pl.FP32],
    ones_cdv: pl.Tensor[[C, DV], pl.FP32],
    zero: pl.Tensor[[DK, DV], pl.FP32],
    Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
):
    """Local end-of-slice state S_total (from S=0), carried over N chunks."""
    tril_t: pl.Tile[[C, C], pl.FP32] = pl.load(tril, [0, 0], [C, C])
    ones_cc_t: pl.Tile[[C, C], pl.FP32] = pl.load(ones_cc, [0, 0], [C, C])
    ones_cdv_t: pl.Tile[[C, DV], pl.FP32] = pl.load(ones_cdv, [0, 0], [C, DV])
    s_init: pl.Tile[[DK, DV], pl.FP32] = pl.load(zero, [0, 0], [DK, DV])
    for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
        off = n * C
        k: pl.Tile[[C, DK], pl.FP32] = pl.load(K, [off, 0], [C, DK])
        v: pl.Tile[[C, DV], pl.FP32] = pl.load(V, [off, 0], [C, DV])
        a: pl.Tile[[C, DK], pl.FP32] = pl.load(A, [off, 0], [C, DK])
        la: pl.Tile[[C, DK], pl.FP32] = pl.log(a)
        b: pl.Tile[[C, DK], pl.FP32] = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
        g_row_full: pl.Tile[[C, DK], pl.FP32] = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
        g_full: pl.Tile[[DK, DV], pl.FP32] = pl.exp(
            pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
        kb: pl.Tile[[C, DK], pl.FP32] = pl.div(k, b)
        kbar: pl.Tile[[C, DK], pl.FP32] = pl.mul(kb, g_row_full)
        kv: pl.Tile[[DK, DV], pl.FP32] = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
        s_scaled: pl.Tile[[DK, DV], pl.FP32] = pl.mul(s_run, g_full)
        s_new: pl.Tile[[DK, DV], pl.FP32] = pl.add(s_scaled, kv)
        s_fin = pl.yield_(s_new)
    out: pl.Tensor[[DK, DV], pl.FP32] = pl.store(s_fin, [0, 0], Stot)
    return out


@pl.jit
def stage1(
    A: pl.Tensor[[L, DK], pl.FP32],
    K: pl.Tensor[[L, DK], pl.FP32],
    V: pl.Tensor[[L, DV], pl.FP32],
    tril: pl.Tensor[[C, C], pl.FP32],
    ones_cc: pl.Tensor[[C, C], pl.FP32],
    ones_cdv: pl.Tensor[[C, DV], pl.FP32],
    zero: pl.Tensor[[DK, DV], pl.FP32],
    Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
):
    Stot = _stage1_kernel(A, K, V, tril, ones_cc, ones_cdv, zero, Stot)
    return Stot


@pl.jit.incore
def _stage2_kernel(
    Q: pl.Tensor[[L, DK], pl.FP32],
    K: pl.Tensor[[L, DK], pl.FP32],
    V: pl.Tensor[[L, DV], pl.FP32],
    A: pl.Tensor[[L, DK], pl.FP32],
    tril: pl.Tensor[[C, C], pl.FP32],
    mask: pl.Tensor[[C, C], pl.FP32],
    ones_cc: pl.Tensor[[C, C], pl.FP32],
    ones_cdv: pl.Tensor[[C, DV], pl.FP32],
    Srecv: pl.Tensor[[DK, DV], pl.FP32],
    O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
):
    """Output per chunk given the boundary state; carry S_run from S_recv."""
    tril_t: pl.Tile[[C, C], pl.FP32] = pl.load(tril, [0, 0], [C, C])
    mask_t: pl.Tile[[C, C], pl.FP32] = pl.load(mask, [0, 0], [C, C])
    ones_cc_t: pl.Tile[[C, C], pl.FP32] = pl.load(ones_cc, [0, 0], [C, C])
    ones_cdv_t: pl.Tile[[C, DV], pl.FP32] = pl.load(ones_cdv, [0, 0], [C, DV])
    s_init: pl.Tile[[DK, DV], pl.FP32] = pl.load(Srecv, [0, 0], [DK, DV])
    out = O
    for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
        off = n * C
        q: pl.Tile[[C, DK], pl.FP32] = pl.load(Q, [off, 0], [C, DK])
        k: pl.Tile[[C, DK], pl.FP32] = pl.load(K, [off, 0], [C, DK])
        v: pl.Tile[[C, DV], pl.FP32] = pl.load(V, [off, 0], [C, DV])
        a: pl.Tile[[C, DK], pl.FP32] = pl.load(A, [off, 0], [C, DK])
        la: pl.Tile[[C, DK], pl.FP32] = pl.log(a)
        b: pl.Tile[[C, DK], pl.FP32] = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
        g_row_full: pl.Tile[[C, DK], pl.FP32] = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
        g_full: pl.Tile[[DK, DV], pl.FP32] = pl.exp(
            pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
        qt: pl.Tile[[C, DK], pl.FP32] = pl.mul(q, b)
        kb: pl.Tile[[C, DK], pl.FP32] = pl.div(k, b)
        # intra-chunk causal attention
        scores: pl.Tile[[C, C], pl.FP32] = pl.mul(
            pl.matmul(qt, pl.transpose(kb, 0, 1), out_dtype=pl.FP32), mask_t)
        o_intra: pl.Tile[[C, DV], pl.FP32] = pl.matmul(scores, v, out_dtype=pl.FP32)
        # inter (history) term: (Q*b) @ S_run  (S_run already folds S_recv + local hist)
        s_run_v: pl.Tile[[DK, DV], pl.FP32] = pl.mul(s_run, 1.0)   # detach carry for matmul
        o_inter: pl.Tile[[C, DV], pl.FP32] = pl.matmul(qt, s_run_v, out_dtype=pl.FP32)
        o_n: pl.Tile[[C, DV], pl.FP32] = pl.add(o_inter, o_intra)
        out = pl.store(o_n, [off, 0], out)
        # advance state: S <- (S ⊙ gamma) + (k*(gamma/b))^T @ v
        kbar: pl.Tile[[C, DK], pl.FP32] = pl.mul(kb, g_row_full)
        kv: pl.Tile[[DK, DV], pl.FP32] = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
        s_scaled: pl.Tile[[DK, DV], pl.FP32] = pl.mul(s_run, g_full)
        s_new: pl.Tile[[DK, DV], pl.FP32] = pl.add(s_scaled, kv)
        s_run = pl.yield_(s_new)
    return out


@pl.jit
def stage2(
    Q: pl.Tensor[[L, DK], pl.FP32],
    K: pl.Tensor[[L, DK], pl.FP32],
    V: pl.Tensor[[L, DV], pl.FP32],
    A: pl.Tensor[[L, DK], pl.FP32],
    tril: pl.Tensor[[C, C], pl.FP32],
    mask: pl.Tensor[[C, C], pl.FP32],
    ones_cc: pl.Tensor[[C, C], pl.FP32],
    ones_cdv: pl.Tensor[[C, DV], pl.FP32],
    Srecv: pl.Tensor[[DK, DV], pl.FP32],
    O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
):
    O = _stage2_kernel(Q, K, V, A, tril, mask, ones_cc, ones_cdv, Srecv, O)
    return O
'''


def make_zeco_jits(L: int, C: int, dk: int, dv: int):
    """Build the ``(stage1, stage2)`` chunk-recurrent InCore kernels for these shapes.

    Args:
        L: Tokens per device.
        C: Chunk size (``L`` must be divisible by ``C``); ``N = L // C`` chunks.
        dk: Key/query dimension.
        dv: Value dimension.

    Returns:
        ``(stage1, stage2)`` ``@pl.jit`` entry callables specialised for these shapes.
    """
    assert L % C == 0, f"L={L} not divisible by C={C}"
    src = _TEMPLATE.format(L=L, C=C, N=L // C, DK=dk, DV=dv)
    fd, path = tempfile.mkstemp(prefix=f"zeco_chunk_L{L}c{C}_{dk}x{dv}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"zeco_chunk_L{L}c{C}_{dk}x{dv}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stage1, mod.stage2
