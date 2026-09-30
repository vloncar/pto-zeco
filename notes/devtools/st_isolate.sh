#!/usr/bin/env bash
# solve_tril at FRACTAL=128 PASSES a2a3sim but fails a2a3 hardware with
# `prepare_native_run failed with code 13`. The same kernel body ran on hardware
# earlier as devtools/../d1/d1_f128_timing.py, so the break is in what changed
# around it: an unused `masks` ARGUMENT was dropped, and a dead `ident` slice was
# dropped, leaving `consts` read only from offset 128.
#
# Isolation matrix rather than a guess, with the known-good kernel as the control.
set +e
DT=/root/workspace/allscan/devtools
X=/tmp/claude-0/-root-workspace-allscan/7ebeee77-7eb5-4eea-bee4-c0b033fc3dcd/scratchpad/xstage
D1=/tmp/claude-0/-root-workspace-allscan/7ebeee77-7eb5-4eea-bee4-c0b033fc3dcd/scratchpad/d1
bash $DT/tq_pypto.sh bash -c '
set +e
cd /root/workspace/allscan/pypto-lib || exit 1
export PYTHONPATH=/root/workspace/allscan/pypto-lib:$PYTHONPATH
X='"$X"'; D1='"$D1"'
for V in "control: d1_f128 (ran on HW before):$D1/d1_f128_timing.py" \
         "shipped (fails):models/gdn/solve_tril.py" \
         "+ unused masks argument back:$X/st_masks.py" \
         "+ dead ident slice back:$X/st_ident.py" \
         "consts = -I only, slice at offset 0:$X/st_negonly.py"; do
  printf "\n### %s ###\n" "${V%%:*}"
  python "${V#*:}" -p a2a3 -d 0 2>&1 \
    | grep -E "^\[stats\]|^\[RUN\] (PASS|FAIL)|RuntimeError|Error|error:" | head -3
done'
