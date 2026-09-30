#!/usr/bin/env bash
# A6: run the backward at several shapes with the search choosing the blocking.
# Usage: a6_shapes.sh "<P L C dk dv devs>" "<P L C dk dv devs>" ...
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
for case in "$@"; do
  rm -f /tmp/barrier_pto_multi_comm_*
  out=$(timeout 3600 python3 -u /root/workspace/allscan/devtools/a6_case.py a2a3 $case 2 2>&1 \
        | grep -E "PASS|FAIL|Error|error:" | tail -2)
  printf '%-26s %s\n' "$case" "${out:-ERR}"
done
echo A6DONE
