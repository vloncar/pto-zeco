#!/usr/bin/env bash
# Is the trigger the CYCLIC WAIT, or the CROSS-DISPATCH SPLIT of a rank's send and wait?
#   both       both ranks wait; each rank's send and wait in SEPARATE dispatches -> known deadlock
#   fusedcomm  both ranks wait; each rank's send and wait in ONE kernel (allscan/EP shape)
#   samedir    rank 0 wait-free (control that already passes)
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1}"
cd "$ROOT/pto-zeco" || exit 1

echo "=== comm canary on $DEV ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
timeout 400 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1 | tail -3
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

run () {
    echo "=== arm $2 ==="
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
    timeout 360 python3 -u "$ROOT/devtools/b4_tworing_wait_probe.py" "$1" "$DEV" a2a3 2>&1 \
        | tail -5 | sed "s/^/[$2] /"
    rc=${PIPESTATUS[0]}
    [ "$rc" = "124" ] && echo "[$2] VERDICT: TIMED OUT (DEADLOCK)" || echo "[$2] VERDICT: rc=$rc"
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
}

run fusedcomm fusedcomm-1
run fusedcomm fusedcomm-2
run fusedcomm fusedcomm-3
run both      both-recheck
run samedir   samedir-control
echo "=== split test done ==="
