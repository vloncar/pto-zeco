#!/usr/bin/env bash
# B4 issue A, phase 1: does the P>1 stall belong to the forward ring, the reverse ring, or
# only to both together?
#
# Everything about DELIVERY is now proven correct (b4_reread: 36/36 arrive, 0 never arrive),
# so the wait is blocking on something that never comes rather than on lost data. This
# bisects which half owns it, ON THE REAL PROGRAM WITH THE WAITS INTACT -- the previous
# probe family removed the waits and therefore measured its own missing synchronisation.
#
# Arms, cheapest information first, each bounded so one hang cannot consume the grant:
#   base  full fused backward           -- is the baseline still reproducible on ptoas 0.57?
#   fwd   reverse ring stubbed out      -- forward ring + recompute/grad_o chain alone
#   rev   forward ring stubbed out      -- reverse ring alone
#
#   base stalls, fwd completes, rev completes -> neither ring alone; the TWO-RING/two-window
#                                                interaction owns it. (Note the old
#                                                b4_tworing_probe "exoneration" of two windows
#                                                was a wait-less probe and does not count.)
#   fwd stalls                                -> forward ring / recompute->grad_o chain
#   rev stalls                                -> reverse ring kernels
#   base COMPLETES                            -> baseline moved under ptoas 0.57; re-scope
#
# Numbers are wrong by construction in the stubbed arms; the only question is COMPLETION.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1}"
cd "$ROOT/pto-zeco" || exit 1

echo "=== comm canary on $DEV ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
timeout 400 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1 | tail -4
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

arm () {
    local tag="$1"; shift
    echo "=== arm $tag ==="
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
    timeout "${ARM_TIMEOUT:-500}" python3 -u "$@" 2>&1 \
        | grep -vE "^\[.*\] \[(info|debug)\]" | tail -25 | sed "s/^/[$tag] /"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = "124" ]; then
        echo "[$tag] VERDICT: TIMED OUT (stalled)"
    else
        echo "[$tag] VERDICT: completed rc=$rc"
    fi
    rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
}

# Baseline first: if this no longer stalls, the whole phase needs re-scoping.
arm BASE "$ROOT/devtools/b4_ring_diag.py" "$DEV" a2a3 2
arm FWD  "$ROOT/devtools/b4_bisect.py" "$DEV" a2a3
arm REV  "$ROOT/devtools/b4_bisect.py" "$DEV" a2a3 --stub-forward

echo "=== phase1 done ==="
