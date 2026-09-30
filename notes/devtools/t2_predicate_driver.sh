#!/usr/bin/env bash
# Task 2: drive the predicate sweep, one config per fresh process.
#
# Tests whether the trigger is exactly "dk < C" (mirroring the known "dv < C"), by walking dk
# below / at / above C at two chunk sizes, and by isolating the dv side under the build guard's
# ZECO_ALLOW_TALL bypass. P=1 throughout -- the failure is not P-dependent and P=1 is cheapest.
set +e
N=${N:-20}
PROBE=/root/workspace/allscan/devtools/t2_predicate_sweep.py

# P  L    C   dk   dv   allow_tall
CONFIGS="
1 128  64   16   64  0
1 128  64   32   64  0
1 128  64   64   64  0
1 128  32   16   32  0
1 128  32   32   32  0
1 128  32   64   32  0
1 128  32   16   64  0
1 128  32   32   64  0
1 128  64   64   32  1
1 128  32   32   16  1
"

echo "=== predicate sweep, N=$N dispatches per config, fresh process each ==="
echo "$CONFIGS" | while read -r P L C dk dv tall; do
  [ -z "$P" ] && continue
  if [ "$tall" = 1 ]; then
    ZECO_ALLOW_TALL=1 python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" "$P" "$L" "$C" "$dk" "$dv" 2>&1 | grep -E "^SWEEP"
  else
    python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" "$P" "$L" "$C" "$dk" "$dv" 2>&1 | grep -E "^SWEEP"
  fi
done
