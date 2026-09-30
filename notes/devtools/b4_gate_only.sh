#!/usr/bin/env bash
# B4 gate only: the original target, on the patched pypto. Canary-guarded.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1}"
cd "$ROOT/pto-zeco" || exit 1
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
canary=$(timeout 400 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
echo "$canary" | grep -vE "TIMING|STRACE" | tail -4
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: no comm-capable pair on $DEV ==="; exit 3
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
echo "=== pytest B4 backward (P=1, P=2; P=4 skips on 2 cards) ==="
timeout 1500 python3 -m pytest -x -q gla/tests/test_pypto_gla_backward.py \
    -k "test_pypto_zeco_backward and not sizes and not module and not repeat" 2>&1 \
    | grep -vE "TIMING|STRACE" | tail -14
rc=${PIPESTATUS[0]}
case "$rc" in
    4|5) echo "B4 GATE DID NOT RUN (rc=$rc)" ;;
    *)   echo "B4 pytest rc=$rc" ;;
esac
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
