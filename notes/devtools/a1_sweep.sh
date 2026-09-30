#!/usr/bin/env bash
# Sweep A1 probes over shapes/blockings; one process each so an overflow doesn't kill the rest.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
PROBE="$1"; shift
DEV="${DEV:-0}"
while read -r NB CC DK DV NN; do
  [ -z "$NB" ] && continue
  out=$(timeout 900 python3 -u "$PROBE" a2a3 "$DEV" "$NB" "$CC" "$DK" "$DV" "$NN" 2>&1 \
        | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint|DeprecationWarning|@pl.function")
  res=$(echo "$out" | grep -E "PASS|FAIL|SKIP" | tail -1)
  why=$(echo "$out" | grep -oE "(Vec|Mat|Left|Right|Acc) buffer usage \([0-9]+ bytes\)" | tail -1)
  err=$(echo "$out" | grep -oE "max\|.*" | tail -1)
  printf "C=%-4s dk=%-4s dv=%-4s N=%-3s NB=%-3s  %s %s %s\n" \
    "$CC" "$DK" "$DV" "$NN" "$NB" "${res:-ERR}" "$why" "$err"
done
echo DONE
