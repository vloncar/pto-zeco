#!/usr/bin/env bash
# Re-validate and re-benchmark after the kernel style pass, which changed the
# generated code in four stages (chunk_o alone loses 25 ops). Same shapes, same
# rounds/warmup as the run before it, so the two are directly comparable.
set +e
DT=/root/workspace/allscan/devtools
OUT=$DT/gdn_bench
mkdir -p "$OUT"

echo "#### device occupancy before the run ####"
npu-smi info 2>/dev/null | sed -n '3,14p'

echo
echo "######################## 1. validation, reference-chained ########################"
bash $DT/tq_pypto.sh bash -c '
cd /root/workspace/allscan/pypto-lib
export PYTHONPATH=/root/workspace/allscan/pypto-lib:$PYTHONPATH
timeout 3600 python models/gdn/test_gdn_stages.py -p a2a3 -d 0 2>&1 \
  | grep -E "^====|^\[stats\]|^\[RUN\] (PASS|FAIL)|^  (PASS|FAIL)|stages pass|Error|error:|Traceback"'

echo
echo "######################## 2. validation, device-chained ########################"
bash $DT/tq_pypto.sh bash -c '
cd /root/workspace/allscan/pypto-lib
export PYTHONPATH=/root/workspace/allscan/pypto-lib:$PYTHONPATH
timeout 3600 python models/gdn/test_gdn_stages.py -p a2a3 -d 0 --chain \
  --save-output /root/workspace/allscan/devtools/gdn_bench/o_chained.pt 2>&1 \
  | grep -E "^====|^\[stats\]|^\[RUN\] (PASS|FAIL)|^  (PASS|FAIL)|stages pass|saved|Error|error:|Traceback"'

echo
echo "######################## 3. PyPTO benchmark sweep ########################"
bash $DT/tq_pypto.sh bash -c '
cd /root/workspace/allscan/pypto-lib
export PYTHONPATH=/root/workspace/allscan/pypto-lib:$PYTHONPATH
timeout 5400 python models/gdn/bench.py -p a2a3 -d 0 \
  --seq-len 4096,8192,16384 --heads 16 --rounds 50 --warmup 5 \
  --json '"$OUT"'/gdn_bench_pypto.json 2>&1 \
  | grep -E "^\[bench\]|^\||Error|error:|Traceback"'

echo
echo "######################## 4. end to end vs megagdn ref_gdn (float64) ########################"
timeout 1800 bash $DT/mega_env.sh $DT/gdn_e2e_score.py "$OUT/o_chained.pt" 2>&1 | tail -12
