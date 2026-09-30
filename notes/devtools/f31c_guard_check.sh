#!/usr/bin/env bash
# Decisive A/B for the P=2 C=64 dk=32 dv=64 failure: is it the SAME pto-isa FIFO defect as
# F3.1c (in which case the `dv >= C` guard is incomplete and `dk` matters after all), or a
# separate P=2 problem?
#
# Runs the one failing shape against the STOCK pinned pto-isa, then against a tree carrying
# the local-slot-stride fix. Same program, same data, only the header differs.
set +e
FIXED_ISA=/tmp/claude-0/-root-workspace-allscan/9f70ad70-920a-4d23-905e-77956a96d143/scratchpad/gitcode/pto-isa

run_half() {
  echo
  echo "############### $1 ###############"
  echo "[guard] PTO_ISA_ROOT=$2"
  PTO_ISA_ROOT="$2" python3 -m pytest gla/tests/test_pypto_gla.py \
      -q --platform a2a3 --device "$TASK_DEVICE" \
      -k "128-64-32-64" 2>&1 | tail -6
}

run_half "STOCK pinned pto-isa (/opt/pto-isa)" /opt/pto-isa
run_half "WITH the F3.1c local-slot fix"       "$FIXED_ISA"
