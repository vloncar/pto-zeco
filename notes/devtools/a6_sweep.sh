#!/usr/bin/env bash
# A6: run one backward shape at several FORCED blockings. A split that a shape does not need
# must reproduce the unsplit answer -- that is the bit-identity check, not merely "it fits".
# Usage: a6_sweep.sh <P> <L> <C> <dk> <dv> [devs] [plans...]
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
rm -f /tmp/barrier_pto_multi_comm_*
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
P=$1; L=$2; C=$3; DK=$4; DV=$5; DEVS=${6:-0}
shift 6
for plan in "$@"; do
  out=$(ZECO_FORCE_PLAN="$plan" timeout 2400 python3 -u /root/workspace/allscan/devtools/a6_case.py \
        a2a3 "$P" "$L" "$C" "$DK" "$DV" "$DEVS" 2 2>&1 | grep -E "PASS|FAIL|Error|error:" | tail -2)
  printf '%-16s %s\n' "$plan" "${out:-ERR}"
done
echo A6DONE
