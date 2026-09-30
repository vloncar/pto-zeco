#!/usr/bin/env bash
# Run one A6 backward case through the TaskQueue instead of taking a card behind its back.
#
# The queue is how this box arbitrates its eight NPUs: `task-submit` takes an npu-lock flock on
# the cards it grants, so a job started outside it can land on a card another tenant already
# holds. That does not surface as a queue conflict -- it surfaces as unexplained slowness, or
# as a device bring-up error, in whichever of the two jobs is unlucky. `tq_env.sh` then sets
# the env the queue does not (PYTHONPATH, PTO_ISA_ROOT, the HCCL LD_PRELOAD) and hands the
# granted cards over as $TASK_DEVICE, already re-indexed to 0..N-1.
#
# Usage: a6_tq.sh <P> <L> <C> <dk> <dv> [repeats] [max-seconds]
set +e
P=$1; L=$2; C=$3; DK=$4; DV=$5; R=${6:-2}; MAXT=${7:-3600}
# --ptoas none: tq_env.sh points PTOAS_ROOT at /opt/ptoas-bin, which is the toolchain every
# result so far was measured against; letting the queue inject "the latest under
# /usr/local/ptoas" instead would quietly change it.
task-submit --device auto --device-num "$P" --max-time "$MAXT" --timeout "$MAXT" --ptoas none --run \
  "bash /root/workspace/allscan/devtools/tq_env.sh python3 -u /root/workspace/allscan/devtools/a6_case.py a2a3 $P $L $C $DK $DV \$TASK_DEVICE $R" \
  2>&1 | grep -vE "warning|TIMING|STRACE|^\[chip_process|perf_hint|INFO_V9|^\[20" | tail -12
