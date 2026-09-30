#!/usr/bin/env bash
# A6: static cube/vector placement check across several forced blockings (no NPU needed).
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
SHAPE=${SHAPE:-"128 32 32 32"}
for plan in "$@"; do
  out=$(timeout 900 python3 -u /root/workspace/allscan/devtools/a6_split_check.py "$plan" $SHAPE 2>&1 \
        | grep -E "TOTAL|BAD|Error" | tail -3)
  printf '%-14s %s\n' "$plan" "${out:-ERR}"
done
echo PLANDONE
