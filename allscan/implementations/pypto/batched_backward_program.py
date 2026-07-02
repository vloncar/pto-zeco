"""Generate a *batched* variant of the AllScan backward program for fair timing.

Backward analogue of :mod:`implementations.pypto.batched_program`. The single-ring
:func:`program_backward.build_allscan_backward_program` pays a full comm-domain
alloc/free + drain round-trip per dispatch, so its per-call latency is dominated
by that fixed overhead. To compare the marginal kernel+comm cost against the
simpler backend on equal footing, we dispatch ``B`` independent backward passes
inside ONE dispatch (one comm domain), each ring on its own disjoint window
buffers + output slice — exactly like simpler's batched ``measure``.

The DSL cannot express ``B`` disjoint window buffers in a loop (``alloc_window_buffer``
names are parser-injected from the LHS and must be globally unique; ``pld.window``
has no offset), so we *generate* the host orchestrator with ``B`` explicitly-named
buffer pairs and splice it into the pristine ``program_backward.py`` source (the
InCore/Orchestration kernels stay the single source of truth), then import the
result from a real temp file so the DSL parser's ``inspect.getsource`` works.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

_HOST_ORCH_MARKER = "        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)"
_TAIL = "\n    return AllScanBackwardProgram\n"


def _gen_host_orch(B: int) -> str:
    """Emit a host_orch that runs B independent backward rings, each on its own
    window buffers and disjoint output slices ``dS[b]`` / ``dgamma[b]``.

    Args:
        B: Number of independent backward rings / output slices to emit.

    Returns:
        The generated ``host_orch`` method source as a string.
    """
    lines = [
        "        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)",
        "        def host_orch(",
        "            self,",
        "            g_outs: pl.Tensor[[P, dk, dv], pl.FP32],",
        "            gammas: pl.Tensor[[P, dk, 1], pl.FP32],",
        "            out_prevs: pl.Tensor[[P, dk, dv], pl.FP32],",
        f"            dS: pl.Out[pl.Tensor[[{B}, P, dk, dv], pl.FP32]],",
        f"            dgamma: pl.Out[pl.Tensor[[{B}, P, dk, 1], pl.FP32]],",
        "        ):",
        "",
    ]
    for b in range(B):
        lines += [
            f"            dst_buf_{b} = pld.alloc_window_buffer(dk * dv * 4)",
            f"            signal_buf_{b} = pld.alloc_window_buffer(K * 4)",
            f"            dS_{b} = dS[{b}]",
            f"            dgamma_{b} = dgamma[{b}]",
            "            for r in pl.range(P):",
            "                g_out_r = g_outs[r]",
            "                gamma_r = gammas[r]",
            "                out_prev_r = out_prevs[r]",
            f"                dS_r = dS_{b}[r]",
            f"                dgamma_r = dgamma_{b}[r]",
            f"                dst = pld.window(dst_buf_{b}, [dk, dv], dtype=pl.FP32)",
            f"                signal = pld.window(signal_buf_{b}, [K, 1], dtype=pl.INT32)",
            "                if r == P - 1:",
            "                    self.chip_orch_bwd_source(g_out_r, gamma_r, out_prev_r, dS_r, dgamma_r, dst, signal, r - 1, device=r)",
            "                elif r == 0:",
            "                    self.chip_orch_bwd_terminal(g_out_r, gamma_r, out_prev_r, dS_r, dgamma_r, dst, signal, device=r)",
            "                else:",
            "                    self.chip_orch_bwd_middle(g_out_r, gamma_r, out_prev_r, dS_r, dgamma_r, dst, signal, r - 1, device=r)",
        ]
    lines += ["            return dS, dgamma"]
    return "\n".join(lines)


def make_batched_backward_builder(B: int):
    """Return ``build(dk, dv, K, P) -> AllScanBackwardProgram`` for a B-ring
    batched backward program, with the same closure-var contract as
    :func:`program_backward.build_allscan_backward_program`.

    Args:
        B: Number of independent backward rings per dispatch (batch size).

    Returns:
        A ``build(dk, dv, K, P)`` callable that constructs the batched backward
        program.
    """
    prog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program_backward.py")
    with open(prog_path) as f:
        src = f.read()

    head, sep, _ = src.partition(_HOST_ORCH_MARKER)
    if not sep:
        raise RuntimeError("could not locate host_orch marker in program_backward.py")
    batched_src = head + _gen_host_orch(B) + _TAIL

    fd, path = tempfile.mkstemp(prefix=f"allscan_bwd_batched_B{B}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(batched_src)

    spec = importlib.util.spec_from_file_location(f"allscan_bwd_batched_B{B}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_allscan_backward_program
