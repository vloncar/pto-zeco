"""Generate a *batched* variant of the AllScan DSL program for fair timing.

The single-ring :func:`program.build_allscan_program` pays a full comm-domain
alloc/free + drain round-trip per dispatch, so its per-call latency is dominated
by that fixed overhead. To compare the marginal kernel+comm cost against the
simpler backend on equal footing, we need to dispatch ``B`` independent AllScans
inside ONE dispatch (one comm domain), exactly like simpler's batched ``measure``.

That requires ``B`` disjoint window buffers (each ring's recv + signal region
must not alias another's, or the AtomicAdd signals collide). The DSL cannot
express this in a loop: ``pld.alloc_window_buffer`` names are parser-injected
from the assignment LHS and must be globally unique, and ``pld.window`` has no
offset to sub-slice one buffer. So we *generate* the host orchestrator with ``B``
explicitly-named buffer pairs (``dst_buf_0``, ``signal_buf_0``, ...), splicing it
into the pristine ``program.py`` source (the InCore/Orchestration kernels — and
the race fix in them — stay the single source of truth), then import the result
from a real temp file so the DSL parser's ``inspect.getsource`` works.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

_HOST_ORCH_MARKER = "        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)"
_TAIL = "\n    return AllScanProgram\n"


def _gen_host_orch(B: int) -> str:
    """Emit a host_orch method that runs B independent rings, each on its own
    window buffers and disjoint output slice ``outputs[b]``.

    Args:
        B: Number of independent rings / output slices to emit.

    Returns:
        The generated ``host_orch`` method source as a string.
    """
    lines = [
        "        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)",
        "        def host_orch(",
        "            self,",
        "            S_locals: pl.Tensor[[P, dk, dv], pl.FP32],",
        "            gammas: pl.Tensor[[P, dk, 1], pl.FP32],",
        f"            outputs: pl.Out[pl.Tensor[[{B}, P, dk, dv], pl.FP32]],",
        f"        ) -> pl.Tensor[[{B}, P, dk, dv], pl.FP32]:",
        "",
    ]
    for b in range(B):
        lines += [
            f"            dst_buf_{b} = pld.alloc_window_buffer(dk * dv * 4)",
            f"            signal_buf_{b} = pld.alloc_window_buffer(K * 4)",
            f"            out_{b} = outputs[{b}]",
            "            for r in pl.range(P):",
            "                S_local_r = S_locals[r]",
            "                gamma_r = gammas[r]",
            f"                output_r = out_{b}[r]",
            f"                dst = pld.window(dst_buf_{b}, [dk, dv], dtype=pl.FP32)",
            f"                signal = pld.window(signal_buf_{b}, [K, 1], dtype=pl.INT32)",
            "                if r == 0:",
            "                    self.chip_orch_first(S_local_r, output_r, dst, signal, r + 1, device=r)",
            "                elif r == P - 1:",
            "                    self.chip_orch_last(S_local_r, gamma_r, output_r, dst, signal, device=r)",
            "                else:",
            "                    self.chip_orch_middle(S_local_r, gamma_r, output_r, dst, signal, r + 1, device=r)",
        ]
    lines += ["            return outputs"]
    return "\n".join(lines)


def make_batched_builder(B: int):
    """Return ``build(dk, dv, K, P) -> AllScanProgram`` for a B-ring batched
    program. The returned builder has the same closure-var contract as
    :func:`program.build_allscan_program`, so it plugs straight into
    ``ir.compile``.

    Args:
        B: Number of independent rings per dispatch (batch size).

    Returns:
        A ``build(dk, dv, K, P)`` callable that constructs the batched program.
    """
    prog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program.py")
    with open(prog_path) as f:
        src = f.read()

    head, sep, _ = src.partition(_HOST_ORCH_MARKER)
    if not sep:
        raise RuntimeError("could not locate host_orch marker in program.py")
    batched_src = head + _gen_host_orch(B) + _TAIL

    # Write to a real file so the DSL parser (inspect.getsource / linecache) can
    # read the generated host_orch back during ir.compile.
    fd, path = tempfile.mkstemp(prefix=f"allscan_batched_B{B}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(batched_src)

    spec = importlib.util.spec_from_file_location(f"allscan_batched_B{B}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_allscan_program
