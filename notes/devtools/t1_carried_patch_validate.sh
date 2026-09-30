#!/usr/bin/env bash
# Validate the carried !1457 patch against the exact shapes it is supposed to repair.
#
# These four were REFUSED by build() until the patch landed. Before it:
#   dv < C  -> 20/20 dispatches wrong
#   dk < C  -> ~1/20 dispatches wrong
# The dk pair is why R is high here: at a 5% rate, R=40 still misses a regression 13% of the
# time, and the suite's default 3 repeats would miss it 86% of the time.
set +e
R=${R:-40}
PROBE=/root/workspace/allscan/devtools/t2_predicate_sweep.py
echo "=== carried-patch validation, R=$R dispatches per shape ==="
for cfg in "64 32 32" "32 16 16" "64 32 64" "32 16 32"; do
  set -- $cfg
  python3 "$PROBE" "$TASK_DEVICE" a2a3 "$R" 1 128 "$1" "$2" "$3" 2>&1 | grep -E "^SWEEP"
done
