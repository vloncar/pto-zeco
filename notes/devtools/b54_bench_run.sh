#!/usr/bin/env bash
# B5.4 — fair simpler-vs-pypto numbers for BOTH directions.
#
# Split by impl (IMPLS) and direction (DIRECTION), because --max-time is silently clamped
# to 3600 s and the full 24-row sweep does not fit. bench.py loops direction OUTERMOST, so a
# --direction both job that runs out of time yields ALL forward rows and NO backward rows --
# run the direction you actually need first. Each job writes its own JSON; merge afterwards.
#   IMPLS=simpler bash devtools/b54_bench_run.sh
#   IMPLS=pypto   bash devtools/b54_bench_run.sh
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1,2,3}"
IMPLS="${IMPLS:-simpler pypto}"
DIRECTION="${DIRECTION:-both}"
CONFIGS="${CONFIGS:-}"          # e.g. "2,256,32,32 4,256,32,32"; empty = default sweep
TAG="${IMPLS// /_}_${DIRECTION}${CONFIGS:+_part}"
JSON="${B54_JSON:-$ROOT/devtools/b54_results_${TAG}.json}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PYTHONPATH="$ROOT/pto-zeco:${PYTHONPATH}"
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so

cd "$ROOT/pto-zeco" || exit 1

# A ring must stay inside one HCCS group (0-3 | 4-7); `--device auto` grants straddling
# sets happily and they fail with no error text.
vis="${ASCEND_RT_VISIBLE_DEVICES:-}"
if [[ "$vis" == *,* ]]; then
    lo=0; hi=0
    for d in ${vis//,/ }; do (( d < 4 )) && lo=1 || hi=1; done
    if (( lo && hi )); then
        echo "=== ABORT: card set $vis straddles the HCCS boundary (0-3 | 4-7); resubmit ==="
        exit 99
    fi
fi
echo "=== cards VIS=$vis logical=$DEV impls='$IMPLS' dir=$DIRECTION configs='${CONFIGS:-default}' json=$JSON ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

# Gate, not advisory: the simpler boundary reaches comm_alloc_domain_windows too, so a
# comm-dead grant produces a table of failures that reads like a backend problem.
echo "=== comm canary over $DEV ==="
canary=$(timeout 900 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
echo "$canary" | grep -vE "TIMING|STRACE|perf_hint" | tail -20
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: no comm-capable pair on $DEV (VIS=$vis) ==="; exit 3
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

echo "=== B5.4 bench: both directions, impls '$IMPLS', devices $DEV ==="
timeout 3300 python3 -u gla/bench.py \
    --platform a2a3 --device "$DEV" \
    --direction "$DIRECTION" --impl $IMPLS \
    ${CONFIGS:+--configs $CONFIGS} \
    --iters 10 --warmup 3 \
    --json "$JSON" 2>&1 | grep -vE "TIMING|STRACE|^\[chip_process"
echo "bench rc=${PIPESTATUS[0]}"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
