#!/usr/bin/env bash
set +e
DEV="${TASK_DEVICE:-0}"; DEV="${DEV%%,*}"
ENVSH=/root/workspace/allscan/devtools/tq_pypto.sh
PYPTO=/root/pypto-lib-env/pypto

echo "================= GDN, chained (each stage on the previous kernel's output) ================="
cd /root/workspace/allscan/pypto-lib || exit 1
timeout 2400 bash "$ENVSH" python models/gdn/test_gdn_stages.py -p a2a3 -d "$DEV" --chain 2>&1 | tail -11

echo
echo "================= pypto: advanced examples ================="
cd "$PYPTO" || exit 1
for e in examples/advanced/*.py; do
  case "$(basename $e)" in __init__.py) continue;; esac
  printf "  %-42s " "$(basename $e)"
  timeout 600 bash "$ENVSH" python "$e" >/tmp/v_adv.log 2>&1 && echo PASS \
    || echo "FAIL  $(grep -oE '[A-Za-z]*Error: .*' /tmp/v_adv.log | tail -1 | cut -c1-60)"
done

echo
echo "================= GDN benchmark on the new pypto (T=8192) ================="
cd /root/workspace/allscan/pypto-lib || exit 1
timeout 2400 bash "$ENVSH" python models/gdn/bench.py -p a2a3 -d "$DEV" \
  --seq-len 8192 --heads 16 2>&1 | tail -14
