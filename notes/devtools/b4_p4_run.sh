#!/usr/bin/env bash
# B4 at P=4. Needs FOUR comm-capable cards in one HCCS group (0-3; the 4-7 group has failed
# every P>=2 run -- see the comm-domain card map).
#
# The canary tests PAIRS, so a green canary is necessary but NOT sufficient for P=4: a 4-rank
# domain can still fail where all six pairs pass. Its value here is telling an environmental
# failure apart from a code one, which has already cost several misread runs.
set +e
ROOT=/root/workspace/allscan
DEV="${TASK_DEVICE:-0,1,2,3}"
cd "$ROOT/pto-zeco" || exit 1
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
echo "=== comm canary over $DEV (all in-group pairs) ==="
canary=$(timeout 900 python3 -u "$ROOT/devtools/comm_canary.py" "$DEV" a2a3 2>&1)
echo "$canary" | grep -vE "TIMING|STRACE|perf_hint"
if echo "$canary" | grep -q "comm-capable pairs: NONE"; then
    echo "=== ABORT: no comm-capable pair on $DEV ==="; exit 3
fi
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

# `--device` MUST be passed: conftest's device_ids fixture reads ONLY the pytest CLI option
# (default "0,1") and ignores ASCEND_RT_VISIBLE_DEVICES / TASK_DEVICE entirely. Omitting it on a
# 4-card grant silently leaves device_ids=[0,1], so `test_pypto_zeco_backward[4]` hits its
# `len(device_ids) < P` guard and SKIPS — which prints "2 passed, 1 skipped", exactly what a
# 2-card run prints, so the gate looks like it ran when it did not.
echo "=== pytest B4 backward P=4 (--device $DEV) ==="
timeout 1800 python3 -m pytest -x -q --device "$DEV" gla/tests/test_pypto_gla_backward.py \
    -k "test_pypto_zeco_backward and not sizes and not module and not repeat" 2>&1 \
    | grep -vE "TIMING|STRACE" | tail -18
rc=${PIPESTATUS[0]}
case "$rc" in
    4|5) echo "B4 P=4 GATE DID NOT RUN (rc=$rc -- usage / nothing collected)" ;;
    *)   echo "B4 P=4 pytest rc=$rc" ;;
esac
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null
