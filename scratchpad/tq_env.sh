#!/usr/bin/env bash
# Env shim for TaskQueue-submitted pto-zeco runs:  task-submit ... --run 'bash scratchpad/tq_env.sh <cmd>'
#
# The queue pins the granted card(s) via ASCEND_RT_VISIBLE_DEVICES and re-indexes
# them to 0..N-1, exposing them as $TASK_DEVICE ("0" / "0,1" / ...). Never touch
# ASCEND_RT_VISIBLE_DEVICES here; pass $TASK_DEVICE to --device instead.
set +e

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1   # resets PYTHONPATH -> prepend after
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
export PTOAS_ROOT=/opt/ptoas-bin
export PTO_ISA_ROOT=/opt/pto-isa
export PATH="/opt/ptoas-bin/bin:${PATH}"
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so   # else HCCL hangs at rootinfo

# The broker occasionally hands the task through with TASK_DEVICE still literally "auto"
# (grant raced the assign). ASCEND_RT_VISIBLE_DEVICES has already pinned + re-indexed the
# card(s) to 0.., so derive the logical ids from it rather than failing the run.
if ! [[ "${TASK_DEVICE:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  n=$(awk -F, '{print NF}' <<<"${ASCEND_RT_VISIBLE_DEVICES:-0}")
  TASK_DEVICE=$(seq -s, 0 $((n - 1)))
  export TASK_DEVICE
  echo "[tq_env] WARNING: TASK_DEVICE was not numeric; derived ${TASK_DEVICE} from VIS=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
fi

# The box has two HCCS groups (cards 0-3 and 4-7). A distributed ring must stay inside one:
# `--device auto --device-num 4` happily grants e.g. 1,2,3,4, and then every distributed test
# fails or hangs with NO error text. Warn loudly; set TQ_REQUIRE_HCCS_GROUP=1 to hard-fail
# (exit 99) so a caller can retry for a different set.
vis="${ASCEND_RT_VISIBLE_DEVICES:-}"
if [[ "$vis" == *,* ]]; then
  lo=0; hi=0
  for d in ${vis//,/ }; do (( d < 4 )) && lo=1 || hi=1; done
  if (( lo && hi )); then
    echo "[tq_env] WARNING: card set $vis straddles the HCCS boundary (0-3 | 4-7);" \
         "distributed runs will fail or hang. Resubmit for a different set."
    [[ "${TQ_REQUIRE_HCCS_GROUP:-}" == "1" ]] && exit 99
  fi
fi

# Stale rendezvous files from a killed run make HCCL time out waiting for rootinfo.
# NOTE: this is global — do not run two distributed pto jobs of our own concurrently.
rm -f /tmp/barrier_pto_multi_comm_* 2>/dev/null

cd /root/workspace/allscan/pto-zeco || exit 1
echo "[tq_env] TASK_DEVICE=${TASK_DEVICE:-<unset>} VIS=${ASCEND_RT_VISIBLE_DEVICES:-<unset>} ptoas=${PTOAS_VERSION:-?}"
exec "$@"
