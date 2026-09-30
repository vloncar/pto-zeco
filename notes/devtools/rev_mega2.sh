#!/usr/bin/env bash
# Re-run the megagdn-pto sweep with per-batch synchronisation. The previous run
# queued all 50 batches and synchronised once at the end, which let early start
# events be recorded before the device had begun -- its minima were half its
# medians, which is not kernel variation.
set +e
DT=/root/workspace/allscan/devtools
timeout 5400 bash $DT/mega_env.sh $DT/gdn_bench_mega.py \
  --seq-len 4096,8192,16384 --heads 16 --iters 50 --batch 20 -d 0 \
  --json $DT/gdn_bench/gdn_bench_mega.json 2>&1 | grep -E "^\[mega\]|Error|error:|Traceback"
