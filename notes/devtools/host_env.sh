#!/usr/bin/env bash
# Host-only pypto env (compile / sim; no device, no HCCL, no rendezvous cleanup).
# Use for anything that does not touch an NPU:  bash devtools/host_env.sh <cmd> [args...]
#
# set_env.sh resets PYTHONPATH, so the repo root must be prepended AFTER sourcing it.
set +e
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
export PTOAS_ROOT=/opt/ptoas-bin
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-/opt/pto-isa}"
export PATH="/opt/ptoas-bin/bin:${PATH}"
cd /root/workspace/allscan/pto-zeco || exit 1
exec "$@"
