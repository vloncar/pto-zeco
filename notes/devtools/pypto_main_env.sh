#!/usr/bin/env bash
# Source this to work against CURRENT pypto main, isolated from our pinned tree.
#
# Our own environment is simpler 3165cc89 + pto-isa 83d01313 with two carried patches the
# dk<C / dv<C shapes depend on, and pypto main needs simpler 1f27a157 + pto-isa f51c92f6.
# Nothing under /opt is modified: the newer runtime lives in its own venv and the newer
# pto-isa in its own checkout.
#
#   source devtools/pypto_main_env.sh
#   python3 examples/distributed/04_barrier.py -p a2a3sim -d 0,1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1

# set_env.sh prepends /opt/pypto/runtime/python; a PYTHONPATH entry beats venv
# site-packages, so the old simpler would shadow the new one unless it is dropped.
export PYTHONPATH=$(python3 - <<'PY'
import os
keep = [p for p in os.environ.get("PYTHONPATH", "").split(":")
        if p and not p.startswith("/opt/pypto/runtime")]
print(":".join(keep))
PY
)
source /tmp/pypto-pr/runtime/.venv/bin/activate
export PYTHONPATH=/tmp/pypto-pr/python:$PYTHONPATH
export PTO_ISA_ROOT=/tmp/pto-isa-new
echo "pypto main env: simpler=$(python3 -c 'import simpler,os;print(os.path.dirname(simpler.__file__))')"
