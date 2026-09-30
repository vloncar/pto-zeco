#!/usr/bin/env bash
# Env shim for the MOVED pin: pypto main + simpler 799640e6 + pto-isa cd4a3d3f.
#
# Same contract as tq_env.sh (run it under a task-submit grant; the granted cards arrive as
# $TASK_DEVICE) but pointed at the newer toolchain instead of /opt/pypto. Nothing under /opt
# is modified, so rolling back is a matter of using tq_env.sh again.
set +e

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
# set_env.sh prepends /opt/pypto/runtime/python, and a PYTHONPATH entry beats venv
# site-packages -- so the OLD simpler would shadow the new one unless it is dropped.
export PYTHONPATH=$(python3 - <<'PY'
import os
keep = [p for p in os.environ.get("PYTHONPATH", "").split(":")
        if p and not p.startswith("/opt/pypto/runtime")]
print(":".join(keep))
PY
)
source /tmp/pypto-pr/runtime/.venv/bin/activate
export PYTHONPATH="/root/workspace/allscan/pto-zeco:/tmp/pypto-pr/python:${PYTHONPATH}"
export PTOAS_ROOT=/opt/ptoas-bin
export PTO_ISA_ROOT=/tmp/pto-isa-new
export PATH="/opt/ptoas-bin/bin:${PATH}"
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so

if ! [[ "${TASK_DEVICE:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  n=$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES:-0}")
  TASK_DEVICE=$(seq -s, 0 $((n - 1)))
  export TASK_DEVICE
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
cd /root/workspace/allscan/pto-zeco || exit 1
echo "[tq_env_main] pypto=$(python3 -c 'import pypto,os;print(os.path.dirname(pypto.__file__))')" \
     "simpler=$(python3 -c 'import simpler,os;print(os.path.dirname(simpler.__file__))')" \
     "pto_isa=${PTO_ISA_ROOT} TASK_DEVICE=${TASK_DEVICE:-<unset>}"
exec "$@"
