#!/usr/bin/env bash
# Run the L3 callable-isolation ST on real a2a3 hardware (single card, no collectives).
# The existing dynamic_register cases are simulator-only, so this is the first hardware
# coverage of multi-callable dispatch on one worker.
set +e
DEV="${TASK_DEVICE:-0}"
DEV="${DEV%%,*}"
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
cd /opt/pypto/runtime || exit 1
echo "=== VIS=${ASCEND_RT_VISIBLE_DEVICES:-unset} logical device=$DEV ==="
timeout 2400 python3 -m pytest \
    tests/st/a2a3/tensormap_and_ringbuffer/test_l3_callable_isolation.py \
    --platform a2a3 --device "$DEV" -q 2>&1 | grep -vE "TIMING|STRACE|^\[chip_process"
echo "pytest rc=${PIPESTATUS[0]}"
