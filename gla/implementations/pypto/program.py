"""PyPTO DSL programs for the ZeCO / GLA forward (quadratic, unchunked form).

The per-device GLA compute is split into two ``@pl.jit`` kernels so it composes
with the AllScan boundary exchange at the host level (see :mod:`.impl`):

* ``stage1`` — the local-only state and total decay that feed AllScan:
  ``S_total = (K * (g / b))^T @ V``, ``g = b[L-1]`` where ``b = exp(tril @ log A)``
  is the device-global cumulative decay.
* ``stage2`` — the output, given the received boundary state ``S_recv``:
  ``O = ((Q*b) @ (K/b)^T ⊙ mask) @ V  +  (Q*b) @ S_recv``.

We use the QUADRATIC (whole-device-as-one-block, ``C = L``) form deliberately:
the chunk-recurrent form needs a matmul-fed state carried across a loop, which
this pypto build cannot schedule (silent no-accumulate / scheduler deadlock).
The quadratic form is ``O(L^2)`` but uses only single matmuls — the pattern that
compiles and runs correctly. Every matmul result that feeds another matmul or is
added to another matmul result is first normalized to a vector tile via
``pl.mul(x, 1.0)`` (else codegen ``tmov acc->right`` errors or a scheduler
deadlock when combining two cube-resident tiles).

``@pl.jit`` needs the shapes as compile-time constants in the type annotations,
so — as with :mod:`allscan.implementations.pypto.batched_program` — we generate
the two kernels into a temp module with ``L``/``dk``/``dv`` baked in and import
them back (so the DSL parser's ``inspect.getsource`` works).
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

_TEMPLATE = '''
import pypto.language as pl

L = {L}
DK = {DK}
DV = {DV}


@pl.jit
def stage1(
    Q: pl.Tensor[[L, DK], pl.FP32],
    K: pl.Tensor[[L, DK], pl.FP32],
    V: pl.Tensor[[L, DV], pl.FP32],
    A: pl.Tensor[[L, DK], pl.FP32],
    tril: pl.Tensor[[L, L], pl.FP32],
    Stot: pl.Out[pl.Tensor[[DK, DV], pl.FP32]],
):
    """Local-only end-of-slice state Stot (for AllScan's S_local).

    The device total decay ``g = prod(A)`` fed to AllScan as ``gamma`` is computed
    host-side from ``A`` (identical to ``b[L-1]``), so it is not returned here.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zeco_stage1"):
        b = pl.exp(pl.matmul(tril, pl.log(A)))          # [L,DK] device-global cumprod
        kb = pl.div(K, b)                                # [L,DK] = K / b
        g_row = pl.slice(b, [1, DK], [L - 1, 0])         # [1,DK] device total decay
        Kbar_state = pl.col_expand_mul(kb, g_row)        # [L,DK] = K * (g / b)
        Stot[:, :] = pl.matmul(pl.transpose(Kbar_state, 0, 1), V)   # [DK,DV]
    return Stot


@pl.jit
def stage2(
    Q: pl.Tensor[[L, DK], pl.FP32],
    K: pl.Tensor[[L, DK], pl.FP32],
    V: pl.Tensor[[L, DV], pl.FP32],
    A: pl.Tensor[[L, DK], pl.FP32],
    tril: pl.Tensor[[L, L], pl.FP32],
    mask: pl.Tensor[[L, L], pl.FP32],
    Srecv: pl.Tensor[[DK, DV], pl.FP32],
    O: pl.Out[pl.Tensor[[L, DV], pl.FP32]],
):
    """Output: local GLA over the whole slice + the cross-device prefix term."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zeco_stage2"):
        b = pl.exp(pl.matmul(tril, pl.log(A)))          # [L,DK]
        Qt = pl.mul(Q, b)                                # [L,DK]
        kb = pl.div(K, b)                                # [L,DK]
        scores = pl.mul(pl.matmul(Qt, pl.transpose(kb, 0, 1)), mask)   # [L,L] causal
        O_local = pl.mul(pl.matmul(scores, V), 1.0)      # [L,DV] -> vector tile
        O_cross = pl.mul(pl.matmul(Qt, Srecv), 1.0)      # [L,DV] -> vector tile
        O[:, :] = pl.add(O_local, O_cross)
    return O
'''


def make_zeco_jits(L: int, dk: int, dv: int):
    """Build the ``(stage1, stage2)`` ``@pl.jit`` kernels for the given shapes.

    Args:
        L: Tokens per device (whole device treated as one block, ``C = L``).
        dk: Key/query dimension.
        dv: Value dimension.

    Returns:
        ``(stage1, stage2)`` JIT callables specialised for these shapes.
    """
    src = _TEMPLATE.format(L=L, DK=dk, DV=dv)
    fd, path = tempfile.mkstemp(prefix=f"zeco_jit_L{L}_{dk}x{dv}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"zeco_jit_L{L}_{dk}x{dv}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stage1, mod.stage2
