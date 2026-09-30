#!/usr/bin/env bash
# Env shim for task-submit-granted pypto-lib runs:
#   task-submit --device auto --ptoas none --max-time N \
#     --run 'bash devtools/tq_pypto.sh <cmd>'
#
# The queue pins the granted card via ASCEND_RT_VISIBLE_DEVICES and re-indexes it
# to 0..N-1, so scripts under this shim always pass -d 0. Submit with
# --ptoas none: this env carries its own ptoas inside /root/pypto-lib-env/.venv
# (0.65 since 2026-09-24) and the queue's injected PTOAS_ROOT would shadow it.
set +e
echo "[tq_pypto] VIS=${ASCEND_RT_VISIBLE_DEVICES:-<unset>} TASK_DEVICE=${TASK_DEVICE:-<unset>}"
exec bash /root/pypto-lib-env/env.sh "$@"
