#!/usr/bin/env bash
# A6 regression: the forward suite (which the shared plan-search refactor touches) plus one
# backward spot check. Run inside a TaskQueue grant -- $TASK_DEVICE names the granted cards.
set +e
DEV="${TASK_DEVICE:-0}"
cd /root/workspace/allscan/pto-zeco || exit 1
rc=0
echo "===== forward suite (devices ${DEV}) ====="
python3 -u -m pytest gla/tests/test_pypto_gla.py -q --platform a2a3 --device "$DEV"
[ $? -ne 0 ] && rc=1
echo "===== backward spot check C=64 dk=dv=128 ====="
python3 -u /root/workspace/allscan/devtools/a6_case.py a2a3 1 256 64 128 128 "${DEV%%,*}" 2
[ $? -ne 0 ] && rc=1
echo "===== a6_regress rc=$rc ====="
exit $rc
