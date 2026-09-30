#!/usr/bin/env bash
# B4 driver: one card grant, maximum information.
#
# Card contention on this box is high, so a single-device grant runs BOTH the DSL op probe
# (which isolates the five ops the backward needs and the forward never used) and the
# backward correctness suite. If the probe fails, the test failures that follow are
# explained by it rather than by the operator.
#
# The suite skips its own P>1 cases when fewer devices are granted, so the device list is
# the only knob: `--device 0` runs the P=1 path (all three compute kernels, no rings),
# `--device 0,1` adds both rings.
#
# Usage: bash devtools/b4_run.sh [pytest args...]      (TASK_DEVICE picks the devices)
set +e
DEV="${TASK_DEVICE:-0}"
PLATFORM="${B4_PLATFORM:-a2a3}"
ROOT=/root/workspace/allscan
cd "$ROOT/pto-zeco" || exit 1

rc=0
echo "===== DSL op probe (${PLATFORM}, device $(echo "$DEV" | cut -d, -f1)) ====="
python3 -u "$ROOT/devtools/b4_op_probe.py" "$PLATFORM" "$(echo "$DEV" | cut -d, -f1)"
probe=$?
echo "op probe exit=$probe"
[ $probe -ne 0 ] && rc=1

echo "===== backward suite (devices ${DEV}) ====="
python3 -u -m pytest gla/tests/test_pypto_gla_backward.py -v \
    --platform "$PLATFORM" --device "$DEV" "$@"
[ $? -ne 0 ] && rc=1

echo "===== b4_run.sh rc=$rc ====="
exit $rc
