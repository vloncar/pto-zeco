#!/usr/bin/env bash
# Fair comparison #1 — SINGLE-CHIP operator latency, simpler vs pypto, both directions.
#
# Why P=1 is the fair point: at one rank the simpler backend short-circuits the boundary
# collective entirely (gla/implementations/simpler/impl.py:583, :673, :709) -- no HCCL
# worker is built, no devices are released, nothing is stood back up. The 26.6-27.1 s
# per-phase worker build that dominates (and disqualifies) every P>=2 row simply does not
# happen. Both backends then run pure operator compute on one card, both steady-state,
# both correctness-verified, both under the same stopwatch (F6.6 step 1).
#
# This is an END-TO-END operator number, not a kernel number: a dispatch round trip costs
# ~33 ms (fwd) / ~39 ms (bwd) flat, simpler makes 3/5 of them and pypto makes 1.
#
#   task-submit --device auto --max-time 2400 --run 'bash devtools/tq_env.sh bash devtools/p1_bench_run.sh'
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0}"
JSON="${P1_JSON:-$ROOT/devtools/p1_results.json}"
CONFIGS="${CONFIGS:-1,128,32,32 1,256,32,32 1,128,32,64}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PYTHONPATH="$ROOT/pto-zeco:${PYTHONPATH}"
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so

cd "$ROOT/pto-zeco" || exit 1

# No comm canary here: P=1 allocates no comm domain and needs no peer, so there is no pair
# to probe. Cards pass P=1 work even when their comm windows are dead.
echo "=== cards VIS=${ASCEND_RT_VISIBLE_DEVICES:-<unset>} logical=$DEV configs='$CONFIGS' json=$JSON ==="
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

timeout 2300 python3 -u gla/bench.py \
    --platform a2a3 --device "$DEV" \
    --direction both --impl simpler pypto \
    --configs $CONFIGS \
    --iters 10 --warmup 3 \
    --json "$JSON" 2>&1 | grep -vE "TIMING|STRACE|^\[chip_process"
echo "bench rc=${PIPESTATUS[0]}"
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
