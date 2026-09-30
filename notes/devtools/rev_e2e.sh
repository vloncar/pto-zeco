#!/usr/bin/env bash
# Device-chained pipeline, saved with the inputs that produced it, then scored
# end to end against megagdn-pto's own float64 reference.
set +e
DT=/root/workspace/allscan/devtools
OUT=$DT/gdn_bench
bash $DT/tq_pypto.sh bash -c '
cd /root/workspace/allscan/pypto-lib
export PYTHONPATH=/root/workspace/allscan/pypto-lib:$PYTHONPATH
timeout 3600 python models/gdn/test_gdn_stages.py -p a2a3 -d 0 --chain \
  --save-output '"$OUT"'/o_chained.pt 2>&1 \
  | grep -E "^====|^\[stats\]|^  (PASS|FAIL)|stages pass|saved|Error|error:|Traceback"'
echo
timeout 1800 bash $DT/mega_env.sh $DT/gdn_e2e_score.py "$OUT/o_chained.pt" 2>&1 | grep -vE "UserWarning|q, k, v ="
