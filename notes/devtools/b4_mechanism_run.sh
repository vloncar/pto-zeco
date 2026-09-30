#!/usr/bin/env bash
# MECHANISM TEST for the B4 P>1 deadlock.
#
# Diagnosis, read out of the runtime's own contract rather than guessed:
#   * pypto lowers each chip dispatch to `orch.submit_next_level`, and
#     `runtime/docs/orchestrator.md` says deps come ONLY from tensor tags. A comm window
#     carries no edge, so program order in host_orch is DISCARDED.
#   * `runtime/docs/scheduler.md`: a task is routed to the per-worker FIFO once its producers
#     complete, and "both immediately-ready submissions and dependency-released consumers use
#     the same routing operation" -- so the FIFO is ordered by when a task becomes READY.
#   * a `c_recv` dispatch takes only Out + the two windows, so NOTHING produces its inputs:
#     fan-in 0, READY at submission, routed AHEAD of the rank's own still-PENDING `c_send`.
#   * one task per worker => the spin-wait owns the core and the send never runs.
#
# The `*dep` arms are byte-identical programs plus ONE artificial RAW edge (the send's Out
# tensor threaded into the recv as an extra INPUT), which is exactly the fan-in the diagnosis
# says is missing. Payload-neutral: recv stores `dst + (dep - dep)`, so the existing
# correctness checks still apply and a pass cannot be a silent no-op.
#
#   dep arms PASS + controls DEADLOCK -> mechanism CONFIRMED; the fix is to give a rank's
#                                        comm dispatches program-order edges
#   dep arms DEADLOCK                 -> fan-in story is wrong; re-open the mechanism
#   controls PASS                     -> baseline moved; the whole comparison is void
#
# Fast arms first so a truncated grant still yields the decisive result.
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
    local out
    out=$(timeout "${3:-360}" python3 -u "$ROOT/devtools/b4_tworing_wait_probe.py" "$1" "$DEV" a2a3 2>&1)
    rc=$?
    echo "$out" | grep -vE "^\[.*\] \[(info|debug)\]" | tail -6 | sed "s/^/[$2] /"
    # A stall shows up TWO ways: the wall-clock `timeout` (rc=124) if we lose the race, or
    # -- more often -- the device scheduler's own timeout firing first and raising
    # `finalize_native_run failed with code -100` (PTO2_ERROR_SCHEDULER_TIMEOUT = 100), which
    # exits rc=1. Reporting that as a bare "rc=1" reads like an ordinary assertion failure and
    # hides the very symptom under test, so classify it explicitly.
    if [ "$rc" = "124" ]; then
        echo "[$2] VERDICT: WALL-CLOCK TIMEOUT (stalled)"
    elif echo "$out" | grep -q "code -100"; then
        echo "[$2] VERDICT: SCHEDULER_TIMEOUT -100 (stalled)"
    elif [ "$rc" = "0" ]; then
        echo "[$2] VERDICT: completed rc=0"
    else
        echo "[$2] VERDICT: FAILED rc=$rc (not a stall -- read the trace above)"
    fi
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
}

# The fix candidates -- expected to COMPLETE, and fast if they do.
run sendfirstdep sendfirstdep-1 200
run bothdep      bothdep-1      200
run sendfirstdep sendfirstdep-2 200
run bothdep      bothdep-2      200

# Controls -- expected to hang, so they cost a full timeout each. Run last.
run sendfirst    sendfirst-control
run both         both-control

echo "=== mechanism test done ==="
