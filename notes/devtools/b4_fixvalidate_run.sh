#!/usr/bin/env bash
# Validate the pypto codegen fix: per-rank comm ordering token.
#
# The fix makes each comm dispatch take a per-rank int32 token as INOUT, so a rank's comm
# dispatches form a WAW chain in program order and a fan-in-free `wait` can no longer be
# routed to the worker FIFO ahead of that rank's own `send`. Generated code verified:
#   __comm_d0_ord = torch.zeros((max(world_size,1),1), ...).share_memory_()
#   _ta_N.add_tensor(make_tensor_arg(__comm_d0_ord[r, 0:1]), TensorArgType.INOUT)
# appended LAST on comm dispatches only (never on compute ones).
#
# Arms, in order of what they would cost if the fix is wrong (cheap disproof first):
#   sendfirst/both  the two arms that deadlock on stock -- now expected to PASS with the
#                   STOCK DSL (no artificial dependency). This is the fix working.
#   fwd/samedir/fusedcomm  arms that already passed -- must STILL pass (no regression).
#   allscan pypto forward + backward   the validated P>=2 programs; the codegen change
#                   touches every distributed program, so these are the real regression gate.
#   B4 P=2          the original target.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1}"
cd "$ROOT/pto-zeco" || exit 1

echo "=== comm canary on $DEV ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
canary=$(timeout 400 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
# Print the WHOLE canary, not `tail -3`: the per-pair "BAD (a,b) <detail>" line is the only
# thing that says whether this is the box or our code, and tail -3 cuts precisely it off.
echo "$canary" | grep -vE "TIMING|STRACE"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
# ABORT if the pair cannot do comm at all. Without this the arms below still run and every
# one of them fails for an environmental reason, which reads exactly like the fix not working
# -- that already burned one grant. A dead canary means the run has no information in it.
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: canary reports no comm-capable pair on $DEV; results would be meaningless ==="
    exit 3
fi

run () {
    echo "=== arm $2 ==="
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
    local out
    out=$(timeout "${3:-240}" python3 -u "$ROOT/devtools/b4_tworing_wait_probe.py" "$1" "$DEV" a2a3 2>&1)
    local rc=$?
    echo "$out" | grep -vE "^\[.*\] \[(info|debug)\]|TIMING|STRACE" | tail -4 | sed "s/^/[$2] /"
    if [ "$rc" = "124" ]; then
        echo "[$2] VERDICT: WALL-CLOCK TIMEOUT (stalled)"
    elif echo "$out" | grep -q "code -100"; then
        echo "[$2] VERDICT: SCHEDULER_TIMEOUT -100 (stalled)"
    elif [ "$rc" = "0" ]; then
        echo "[$2] VERDICT: completed rc=0"
    else
        echo "[$2] VERDICT: FAILED rc=$rc"
    fi
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
}

# --- the fix itself, on the STOCK programs ---
run sendfirst  FIX-sendfirst
run both       FIX-both

# --- must not regress ---
run fwd        REG-fwd
run samedir    REG-samedir
run fusedcomm  REG-fusedcomm

# pytest args are passed as SEPARATE arguments, never as one string: an unquoted "$2" word-
# splits, so a quoted -k expression turns into literal argv entries and pytest exits rc=4
# ("file or directory not found: and") having run nothing. That reads like a failing gate.
# Usage: pytest_arm <label> <timeout> <pytest args...>
pytest_arm () {
    local label="$1"; local tmo="$2"; shift 2
    echo "=== pytest $label ==="
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
    timeout "$tmo" python3 -m pytest -x -q "$@" 2>&1 \
        | grep -vE "TIMING|STRACE" | tail -12 | sed "s/^/[$label] /"
    local rc=${PIPESTATUS[0]}
    # rc=4 (usage) and rc=5 (nothing collected) mean the gate never ran -- call that out
    # rather than letting it read as a pass or a failure.
    case "$rc" in
        4|5) echo "[$label] pytest rc=$rc -- GATE DID NOT RUN (usage / no tests collected)" ;;
        *)   echo "[$label] pytest rc=$rc" ;;
    esac
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
}

# --- regression gate: the HW-validated distributed programs ---
pytest_arm allscan-fwd 900 allscan/tests/test_pypto.py
pytest_arm allscan-bwd 900 allscan/tests/test_pypto_backward.py

# --- the original target ---
pytest_arm B4-bwd 900 gla/tests/test_pypto_gla_backward.py \
    -k "test_pypto_zeco_backward and not sizes and not module and not repeat"

echo "=== fix validation done ==="
