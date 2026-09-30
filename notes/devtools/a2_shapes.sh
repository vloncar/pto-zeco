#!/usr/bin/env bash
# Run "P L C dk dv [forced-plan]" shapes through the real forward, one process each.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
while read -r P L C DK DV PLAN; do
  [ -z "$P" ] && continue
  rm -f /tmp/barrier_pto_multi_comm_*
  devs=$(seq -s, 0 $((P-1)))
  if [ -n "$PLAN" ]; then export ZECO_FORCE_PLAN="$PLAN"; else unset ZECO_FORCE_PLAN; fi
  out=$(timeout 1800 python3 -u /root/workspace/allscan/devtools/a1_case.py a2a3 "$P" "$L" "$C" "$DK" "$DV" "$devs" 2 2>&1 \
        | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint")
  line=$(echo "$out" | grep -E "PASS|FAIL" | tail -1)
  if [ -z "$line" ]; then
    line="ERR: $(echo "$out" | grep -oE "(no blocking plan fits|Error: [^|]*|error: .*|[A-Za-z]+ buffer usage \([0-9]+ bytes\))" | tail -1)"
  fi
  printf "P=%s L=%-5s C=%-4s dk=%-4s dv=%-4s plan=%-8s  %s\n" "$P" "$L" "$C" "$DK" "$DV" "${PLAN:-auto}" "$line"
done
echo DONE
