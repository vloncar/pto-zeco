#!/usr/bin/env bash
# Does padding every comm window buffer to a cache line clear the P>1 stall?
# Control (BASE, unpadded) re-run in the same grant so the comparison is same-cards, same-session.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1}"
cd "$ROOT/pto-zeco" || exit 1

echo "=== comm canary on $DEV ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
timeout 400 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1 | tail -3
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

arm () {
    local tag="$1"; shift
    echo "=== arm $tag ==="
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
    timeout 500 python3 -u "$@" 2>&1 | tail -12 | sed "s/^/[$tag] /"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = "124" ] && echo "[$tag] VERDICT: TIMED OUT (stalled)" || echo "[$tag] VERDICT: rc=$rc"
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
}

arm PADDED   "$ROOT/devtools/b4_pad_probe.py" "$DEV" a2a3
arm CONTROL  "$ROOT/devtools/b4_ring_diag.py" "$DEV" a2a3 2
echo "=== pad test done ==="
