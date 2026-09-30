#!/usr/bin/env bash
# Submit an NPU job and retry on a DIFFERENT card if it lands on a bad one.
#
# Cards 1, 4, 5 and 6 fail every pypto program at device bring-up (halMemCtl rc=42) while
# npu-smi reports them healthy and the queue's block table stays empty, so `--device auto`
# keeps handing them out. A probe that exits 99 (see devtools/canary.py) is saying "this
# card is broken, not my code" — retry rather than believe the result.
#
# Usage: bash devtools/hw_retry.sh <attempts> <python-script> [args...]
set +e

ATTEMPTS="${1:-4}"
shift
SCRIPT="$1"
shift

# --device auto is NOT enough on its own: the broker takes the first FREE card in whitelist
# order, so when the healthy cards are busy it hands out the same bad card every time (five
# consecutive attempts drew 6,6,4,4,4). Walk explicit card numbers instead, skipping any this
# run has already proven bad.
CARDS="${HW_RETRY_CARDS:-0 7 2 3 1 4 5 6}"
tried=""

for i in $(seq 1 "$ATTEMPTS"); do
  log=$(mktemp /tmp/hw_retry_XXXX.log)
  next=""
  for c in $CARDS; do
    case " $tried " in *" $c "*) continue;; esac
    next="$c"; break
  done
  [ -z "$next" ] && { echo "no untried cards left"; exit 1; }
  tried="$tried $next"
  echo "=== attempt $i/$ATTEMPTS (card $next)"
  task-submit --device "$next" --max-time 2400 --timeout 2400 --run \
    "bash devtools/tq_env.sh python3 -u $SCRIPT \$TASK_DEVICE $*" > "$log" 2>&1

  card=$(grep -oE 'acquired lock on device [0-9]+' "$log" | head -1 | grep -oE '[0-9]+$')
  if grep -q "CANARY FAILED\|CANARY WRONG" "$log"; then
    echo "    card ${card:-?} is bad (canary); retrying on another card"
    continue
  fi
  echo "    ran on card ${card:-?}"
  grep -vE "STRACE|INFO_V9|chip_process|^\[20|perf_hint|dur=[0-9]+$" "$log"
  exit 0
done

echo "gave up after $ATTEMPTS attempts — every card tried was bad"
exit 1
