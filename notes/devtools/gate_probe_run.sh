#!/usr/bin/env bash
set +e
D=/tmp/claude-0/-root-workspace-allscan/7ebeee77-7eb5-4eea-bee4-c0b033fc3dcd/scratchpad/s7
cd /root/workspace/allscan/pypto-lib || exit 1
for S in nosplit split; do
  pass=0; fail=0; note=""
  for i in 1 2 3 4 5 6; do
    out=$(python "$D/p_gate_${S}.py" -p a2a3 -d 0 2>&1)
    if echo "$out" | grep -q "^\[RUN\] PASS"; then pass=$((pass+1)); else
      fail=$((fail+1))
      [ -z "$note" ] && note=$(echo "$out" | grep -oE "Max diff[^ ]* [0-9.e+-]+|max abs diff [0-9.]+|error: .{0,70}|Error: .{0,70}" | head -1)
    fi
  done
  printf "gate-broadcast %-8s %d pass / %d FAIL of 6   %s\n" "$S" "$pass" "$fail" "$note"
done
