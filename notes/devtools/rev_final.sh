#!/usr/bin/env bash
# Final grant: validate all six stages both ways, benchmark PyPTO, benchmark
# megagdn-pto at the same shapes, all on one card in one grant.
set +e
DT=/root/workspace/allscan/devtools
OUT=$DT/gdn_bench
mkdir -p "$OUT"

echo "#### device occupancy before the run ####"
npu-smi info 2>/dev/null | sed -n '3,30p'

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
timeout 3600 python models/gdn/test_gdn_stages.py -p a2a3 -d 0 --chain 2>&1 \
  | grep -E "^====|^\[stats\]|^\[RUN\] (PASS|FAIL)|^  (PASS|FAIL)|stages pass|Error|error:|Traceback"'

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
echo "######################## 4. megagdn-pto benchmark sweep ########################"
timeout 5400 bash $DT/mega_env.sh $DT/gdn_bench_mega.py \
  --seq-len 4096,8192,16384 --heads 16 --iters 50 --batch 20 -d 0 \
  --json "$OUT/gdn_bench_mega.json" 2>&1 | grep -E "^\[mega\]|Error|error:|Traceback"
