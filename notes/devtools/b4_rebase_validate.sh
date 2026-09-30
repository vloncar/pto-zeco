#!/usr/bin/env bash
# Validate PR #2398 (rebased) on hardware: the fused GLA backward at P=1/2/4.
#
# On stock pypto every P>1 backward deadlocks; this is the end-to-end proof that the
# comm-ordering fix still works after the rebase onto current main. PYTHONPATH puts the
# rebased worktree AHEAD of the installed pypto so the freshly built pypto_core is used.
set +e
ROOT=/root/workspace/allscan
PR=/tmp/pypto-pr
DEV="${TASK_DEVICE:-0,1,2,3}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PYTHONPATH="$ROOT/pto-zeco:$PR/python:${PYTHONPATH}"
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so

# A ring must stay inside one HCCS group (0-3 | 4-7).
vis="${ASCEND_RT_VISIBLE_DEVICES:-}"
if [[ "$vis" == *,* ]]; then
    lo=0; hi=0
    for d in ${vis//,/ }; do (( d < 4 )) && lo=1 || hi=1; done
    if (( lo && hi )); then
        echo "=== ABORT: card set $vis straddles the HCCS boundary; resubmit ==="; exit 99
    fi
fi
echo "=== cards VIS=$vis logical=$DEV ==="
python3 -c "import pypto, os; print('pypto from:', os.path.dirname(pypto.__file__))"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

echo "=== comm canary over $DEV ==="
canary=$(timeout 900 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
echo "$canary" | grep -vE "TIMING|STRACE|perf_hint" | tail -12
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: no comm-capable pair on $DEV ==="; exit 3
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

cd "$ROOT/pto-zeco" || exit 1
echo "=== pypto GLA backward P=1/2/4 on the REBASED build ==="
timeout 2700 python3 -u -m pytest gla/tests/test_pypto_gla_backward.py \
    -k "test_pypto_zeco_backward and not sizes" \
    --platform a2a3 --device "$DEV" -q 2>&1 | grep -vE "TIMING|STRACE|^\[chip_process"
echo "pytest rc=${PIPESTATUS[0]}"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
