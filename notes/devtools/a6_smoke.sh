#!/usr/bin/env bash
# Quick both-directions smoke test. Run inside a TaskQueue grant ($TASK_DEVICE names the cards).
set +e
DEV="${TASK_DEVICE:-0}"
D1="${DEV%%,*}"
cd /root/workspace/allscan/pto-zeco || exit 1
rc=0
echo "===== forward P=1 ====="
python3 -u /root/workspace/allscan/devtools/a1_case.py a2a3 1 128 32 32 32 "$D1" 2 || rc=1
echo "===== backward P=1 ====="
python3 -u /root/workspace/allscan/devtools/a6_case.py a2a3 1 128 32 32 32 "$D1" 2 || rc=1
if [ "$DEV" != "$D1" ]; then
  echo "===== forward P=2 ====="
  python3 -u /root/workspace/allscan/devtools/a1_case.py a2a3 2 128 32 32 32 "$DEV" 2 || rc=1
  echo "===== backward P=2 ====="
  python3 -u /root/workspace/allscan/devtools/a6_case.py a2a3 2 128 32 32 32 "$DEV" 2 || rc=1
fi
echo "===== a6_smoke rc=$rc ====="
exit $rc
