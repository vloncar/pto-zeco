#!/usr/bin/env bash
# Fair comparison #2 — the COLLECTIVE on its own, simpler vs pypto, forward + backward.
#
# Symmetric by construction: both backends set amortized_timing=True and amortize the same
# way -- B independent rings dispatched under ONE built worker, divided by B (pypto's
# impl.py docstring: "mirroring simpler's batched measure exactly"). So the fixed comm-domain
# setup that ruins the operator-level comparison is paid once per batch on BOTH sides, and
# what is reported is marginal kernel+comm cost.
#
# This re-runs the 2026-07-01 measurement on the CURRENT stack (pypto main / ptoas 0.57 /
# pto-isa 83d01313 + 2 carried fixes). That matters: the July gap at P=2 was attributed to
# pypto's coarse pld.system.fence per block, which F1's upstream fix has since replaced.
#
# Forward and backward run SEQUENTIALLY -- only one distributed worker may be prepared per
# device set at a time.
#
#   task-submit --device auto --device-num 4 --max-time 3300 --run 'bash devtools/tq_env.sh bash devtools/allscan_fair_run.sh'
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1,2,3}"
FJSON="${AS_FWD_JSON:-$ROOT/devtools/allscan_fair_forward.json}"
BJSON="${AS_BWD_JSON:-$ROOT/devtools/allscan_fair_backward.json}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PYTHONPATH="$ROOT/pto-zeco:${PYTHONPATH}"
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so

cd "$ROOT/pto-zeco" || exit 1

# A ring must stay inside one HCCS group (0-3 | 4-7); a straddling grant fails with no text.
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

# Gate, not advisory: a comm-dead grant yields a table of failures that reads like a backend
# problem. Seen on the whole 4-7 group.
echo "=== comm canary over $DEV ==="
canary=$(timeout 900 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
echo "$canary" | grep -vE "TIMING|STRACE|perf_hint" | tail -20
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: no comm-capable pair on $DEV (VIS=$vis) ==="; exit 3
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

echo "=== AllScan FORWARD (amortized, simpler + pypto), devices $DEV ==="
timeout 1400 python3 -u allscan/bench.py \
    --platform a2a3 --device "$DEV" --impl simpler pypto \
    --iters 20 --warmup 5 --json "$FJSON" 2>&1 | grep -vE "TIMING|STRACE|^\[chip_process"
echo "forward rc=${PIPESTATUS[0]}"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

echo "=== AllScan BACKWARD (amortized, simpler + pypto), devices $DEV ==="
timeout 1400 python3 -u allscan/bench_backward.py \
    --platform a2a3 --device "$DEV" --impl simpler pypto \
    --iters 20 --warmup 5 --json "$BJSON" 2>&1 | grep -vE "TIMING|STRACE|^\[chip_process"
echo "backward rc=${PIPESTATUS[0]}"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
