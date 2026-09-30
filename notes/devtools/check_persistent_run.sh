#!/usr/bin/env bash
# Held-worker vs per-kernel-worker check for the simpler backend, BOTH directions.
# Needs up to four comm-capable cards in one HCCS group (0-3) because the P>=2 configs
# run the real boundary AllScan.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1,2,3}"
REPEATS="${REPEATS:-2}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PYTHONPATH="$ROOT/pto-zeco:${PYTHONPATH}"
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so

cd "$ROOT/pto-zeco" || exit 1

# The box has two HCCS groups (0-3, 4-7) and `--device auto --device-num 4` will happily
# grant a straddling set (seen: 2,3,4,5). A ring across the boundary fails or hangs with no
# error text, so bail out immediately and let the caller resubmit rather than burn the slot.
vis="${ASCEND_RT_VISIBLE_DEVICES:-}"
if [[ "$vis" == *,* ]]; then
    lo=0; hi=0
    for d in ${vis//,/ }; do (( d < 4 )) && lo=1 || hi=1; done
    if (( lo && hi )); then
        echo "=== ABORT: card set $vis straddles the HCCS boundary (0-3 | 4-7); resubmit ==="
        exit 99
    fi
fi
echo "=== cards VIS=$vis logical=$DEV ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

# The canary IS a gate for this workload. `comm_alloc_domain_windows` is not
# pypto-specific: the simpler AllScan boundary reaches it too, via
# orch.allocate_domain -> Worker._allocate_domain -> _dispatch_control_domain. Confirmed
# the hard way on VIS=2,3, where the canary said "comm-capable pairs: NONE" and the run
# then died 8 minutes later inside _boundary_on with exactly that error.
echo "=== comm canary over $DEV (all in-group pairs) ==="
canary=$(timeout 900 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
echo "$canary" | grep -vE "TIMING|STRACE|perf_hint" | tail -25
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: no comm-capable pair on $DEV (VIS=$vis); resubmit elsewhere ==="; exit 3
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

echo "=== check_persistent.py $DEV a2a3 $REPEATS ==="
timeout 3000 python3 -u "$ROOT/devtools/check_persistent.py" "$DEV" a2a3 "$REPEATS" 2>&1 \
    | grep -vE "TIMING|STRACE|^\[chip_process"
rc=${PIPESTATUS[0]}
echo "=== check_persistent rc=${rc}"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
exit $rc
