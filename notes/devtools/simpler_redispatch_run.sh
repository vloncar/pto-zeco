#!/usr/bin/env bash
# L3 callable-redispatch probe on real HW. Single device, no HCCL -> no canary, no LD_PRELOAD.
#
# Usage (via taskqueue):
#   task-submit ... --run 'bash devtools/simpler_redispatch_run.sh [<simpler_tree>]'
#
# With no argument it probes the installed runtime (/opt/pypto/runtime, our pinned
# checkout). With a path it probes THAT tree instead, by prepending <tree>/python and
# <tree> to PYTHONPATH -- used to compare our pin against origin/main.
set +e
ROOT=/root/workspace/allscan
# Env var, NOT $1: the task supervisor appends its own flags to the --run command, so a
# positional read picked up "--device" (harmless here, but it silently mislabelled the
# "probing simpler tree:" line of the first run).
TREE="${SIMPLER_TREE:-}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH
export PTO_ISA_ROOT=/opt/pto-isa
if [[ -n "$TREE" ]]; then
    export PYTHONPATH="${TREE}/python:${TREE}:${PYTHONPATH}"
    echo "=== probing simpler tree: ${TREE}"
else
    echo "=== probing the installed runtime (/opt/pypto/runtime)"
fi

DEV="${TASK_DEVICE:-0}"
DEV="${DEV%%,*}"          # single card is enough
echo "=== device=${DEV} VIS=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
cd "$ROOT/allscan/issues/simpler-l3-callable-redispatch" || exit 1

timeout 2400 python3 -u repro.py "$DEV" a2a3 2>&1
rc=$?
echo "=== repro rc=${rc}"
exit $rc
