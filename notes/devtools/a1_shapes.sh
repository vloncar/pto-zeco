#!/usr/bin/env bash
# Run a list of "P L C dk dv" shapes through the real forward, one process each.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
while read -r P L C DK DV; do
  [ -z "$P" ] && continue
  rm -f /tmp/barrier_pto_multi_comm_*
  devs=$(seq -s, 0 $((P-1)))
  out=$(timeout 1200 python3 -u /root/workspace/allscan/devtools/a1_case.py a2a3 "$P" "$L" "$C" "$DK" "$DV" "$devs" 3 2>&1 \
        | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint")
  line=$(echo "$out" | grep -E "PASS|FAIL" | tail -1)
  if [ -z "$line" ]; then
    line="ERR: $(echo "$out" | grep -oE "(no blocking plan fits|error: .*|Error: .*|[A-Za-z]+ buffer usage \([0-9]+ bytes\))" | tail -1)"
  fi
  printf "P=%s L=%-5s C=%-4s dk=%-4s dv=%-4s  %s\n" "$P" "$L" "$C" "$DK" "$DV" "$line"
done
echo DONE
