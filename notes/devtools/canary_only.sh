#!/usr/bin/env bash
# Canary only, FULL output. The per-pair "BAD (a,b) <detail>" line names the actual failure
# (comm_alloc_domain_windows / comm_init / halMemCtl = environment; anything else = our code).
# The fix-validation runner used `tail -3`, which cut precisely that line off.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1}"

# Self-contained: this used to rely on being wrapped in tq_env.sh, and a bare
# invocation died with ModuleNotFoundError: allscan -- which reads like a dead card.
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PYTHONPATH="$ROOT/pto-zeco:${PYTHONPATH}"
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
cd "$ROOT/pto-zeco" || exit 1
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
timeout 300 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1 \
    | grep -vE "TIMING|STRACE" | tail -40
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
