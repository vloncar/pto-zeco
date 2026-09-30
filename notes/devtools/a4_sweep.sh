#!/usr/bin/env bash
# Sweep the A4 key-row probe over "NB NC C dk dv N" lines, one process each.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
DEV="${DEV:-0}"
while read -r NB NC NV CC DK DV NN; do
  [ -z "$NB" ] && continue
  out=$(timeout 1200 python3 -u /root/workspace/allscan/devtools/a4_keyrow_probe.py \
        a2a3 "$DEV" "$NB" "$NC" "$NV" "$CC" "$DK" "$DV" "$NN" 2>&1 \
        | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint|DeprecationWarning|@pl.function")
  res=$(echo "$out" | grep -E "PASS|FAIL|SKIP" | tail -1)
  why=$(echo "$out" | grep -oE "(Vec|Mat|Left|Right|Acc) buffer usage \([0-9]+ bytes\)" | tail -1)
  err=$(echo "$out" | grep -oE "max\|O - golden\| = .*" | tail -1)
  [ -z "$res$why$err" ] && why=$(echo "$out" | grep -oE "Error: .*" | head -1 | cut -c1-90)
  printf "C=%-4s dk=%-4s dv=%-4s N=%-2s NB=%-3s NC=%-3s NV=%-3s  %s %s %s\n" \
    "$CC" "$DK" "$DV" "$NN" "$NB" "$NC" "$NV" "${res:-ERR}" "$why" "$err"
done
echo DONE
