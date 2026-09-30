#!/usr/bin/env bash
# Validate the rebuilt env: our GDN stages, plus untouched examples from
# pypto-lib and pypto. A failure in an untouched example is the environment;
# a failure only in GDN is ours.
set +e
DEV="${TASK_DEVICE:-0}"; DEV="${DEV%%,*}"
ENVSH=/root/workspace/allscan/devtools/tq_pypto.sh
PYPTO=/root/pypto-lib-env/pypto

echo "================= pypto-lib: GDN stages (ours) ================="
cd /root/workspace/allscan/pypto-lib || exit 1
timeout 2400 bash "$ENVSH" python models/gdn/test_gdn_stages.py -p a2a3 -d "$DEV" 2>&1 | tail -11

echo
echo "================= pypto-lib: untouched models ================="
for m in models/qwen3_14b/topk_select.py; do
  printf "  %-42s " "$(basename $m)"
  timeout 900 bash "$ENVSH" python "$m" -p a2a3 -d "$DEV" >/tmp/v_$(basename $m).log 2>&1
  grep -qE "^\[RUN\] PASS|PASS \(" /tmp/v_$(basename $m).log && echo PASS || echo "FAIL  $(grep -oE 'Error[^ ]*|RuntimeError: [^(]*' /tmp/v_$(basename $m).log | tail -1)"
done

echo
echo "================= pypto: repo examples ================="
cd "$PYPTO" || exit 1
for e in examples/beginner/*.py examples/intermediate/*.py; do
  [ -f "$e" ] || continue
  printf "  %-42s " "$(basename $e)"
  timeout 600 bash "$ENVSH" python "$e" >/tmp/v_ex.log 2>&1
  rc=$?
  if grep -qiE "PASS|success|OK$" /tmp/v_ex.log && [ $rc -eq 0 ]; then echo PASS
  elif [ $rc -eq 0 ]; then echo "ran (no verdict)"
  else echo "FAIL rc=$rc  $(grep -oE '[A-Za-z]*Error: .*' /tmp/v_ex.log | tail -1 | cut -c1-70)"
  fi
done
