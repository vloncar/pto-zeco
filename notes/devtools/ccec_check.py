"""Build a generated pypto kernel with the AICore compiler, off-queue, no card.

`ir.compile` and both simulators compile with g++, so a kernel that only the
AICore rejects passes all three. This runs the real compiler on the generated
vector and cube sources, which is the gate the queue was spending grants on.

Usage: ccec_check.py <build_dir_or_glob> [...]
"""
import glob
import os
import sys

from pypto.runtime.kernel_compiler import KernelCompiler
try:
    from simpler_setup.pto_isa import ensure_pto_isa_root
    ISA = ensure_pto_isa_root()
except Exception:
    ISA = "/opt/pto-isa"


def check(build_dir):
    kc = KernelCompiler()
    extra = [f"{ISA}/include", f"{ISA}/include/pto"]
    try:
        extra += list(kc.get_incore_include_dirs())
    except Exception:
        pass
    os.makedirs(f"{build_dir}/ccec_check", exist_ok=True)
    ok = True
    for core in ("aiv", "aic"):
        for src in sorted(glob.glob(f"{build_dir}/kernels/{core}/*.cpp")):
            try:
                kc.compile_incore(src, core_type=core,
                                  pto_isa_root=ISA,
                                  runtime_name="tensormap_and_ringbuffer",
                                  extra_include_dirs=extra,
                                  build_dir=f"{build_dir}/ccec_check")
                print(f"  {core}: OK   {src.split('/')[-1]}")
            except Exception as e:
                ok = False
                msg = str(e)
                lines = [l for l in msg.splitlines() if "error:" in l]
                print(f"  {core}: FAIL {src.split('/')[-1]}")
                for l in (lines or msg.splitlines())[:6]:
                    print(f"        {l.strip()[:200]}")
    return ok


if __name__ == "__main__":
    bad = 0
    for pat in sys.argv[1:]:
        for d in sorted(glob.glob(pat)):
            print(f"=== {d}")
            if not check(d):
                bad += 1
    raise SystemExit(1 if bad else 0)
