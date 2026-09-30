#!/usr/bin/env python3
# Compile-only dump of the two-ring wait variants -- NO NPU, NO device.
#
# `sendfirst` deadlocks on hardware even though it has no cyclic dependency at all
# (both ranks send at dispatch 2 with nothing in front of the send, then both wait at
# dispatch 4). A program that cannot deadlock for ordering reasons but deadlocks anyway
# is polling an address the notify did not write, so the question is what the codegen
# actually emitted -- not what the source says.
#
# Dumps raw MLIR (skip_ptoas=True) for a failing and a passing variant so the emitted
# remote_store / notify / wait addressing can be compared directly.
#
# Usage: python3 devtools/b4_dump_variants.py <variant> [outdir]

from __future__ import annotations

import sys
from pathlib import Path

import pypto.language as pl  # noqa: F401  (import side effects)
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_tworing_wait_probe as probe  # noqa: E402


def main() -> int:
    variant = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/b4dump/{variant}"
    # Deliberately NOT a local copy of the variant table -- see probe.select_program.
    program = probe.select_program(variant)
    compiled = ir.compile(
        program,
        output_dir=outdir,
        platform="a2a3",
        skip_ptoas=True,
        distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    )
    print(f"{variant}: compiled -> {compiled.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
