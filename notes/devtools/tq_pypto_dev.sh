#!/usr/bin/env bash
# As tq_pypto.sh, but resolves pypto from the SOURCE tree at
# /root/pypto-lib-env/pypto/python (with its freshly built extension) instead of
# the venv install -- use it to run hardware jobs against local pypto changes.
set +e
echo "[tq_pypto_dev] VIS=${ASCEND_RT_VISIBLE_DEVICES:-<unset>} TASK_DEVICE=${TASK_DEVICE:-<unset>}"
exec bash /root/pypto-lib-env/devenv.sh "$@"
