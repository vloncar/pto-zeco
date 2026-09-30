#!/usr/bin/env bash
# Task 2: paired A/B, repeated, because reproduction varies PER PROCESS.
#
# Established the hard way: whether a given process reproduces the corruption is not predicted
# by input seeding or by in-process build history -- the same "modes" flipped between jobs. So
# a single A/B pair proves nothing. This alternates stock/fixed headers over several rounds in
# ONE job on ONE card, running the same three probe modes each time, and counts how many cells
# come out wrong per header.
#
# If the fix is real: stock accumulates wrong cells, fixed accumulates ~none.
# Phase 0 proves the header is live. MUST RUN ALONE -- mutates /opt/pto-isa.
set +e
HDR=/opt/pto-isa/include/pto/npu/a2a3/TPush.hpp
BAK=/tmp/t2p_TPush.orig.hpp
N=${N:-6}
ROUNDS=${ROUNDS:-3}
PROBE=/root/workspace/allscan/devtools/t2_confound_probe.py

cp "$HDR" "$BAK" || exit 1
trap 'cp "$BAK" "$HDR"; echo "[t2p] restored the pinned header"' EXIT

apply_fix() {
  cp "$BAK" "$HDR"
  sed -i '/LOCAL_SLOT_NUM/ s/\* ConsM \* ConsN \* sizeof(T);/* RingFiFo::SLOT_SIZE;/' "$HDR"
}

echo "################ PHASE 0 — prove the header is live ################"
sed -i '1i #error PTO_ISA_SENTINEL_HEADER_IS_LIVE' "$HDR"
if python3 "$PROBE" "$TASK_DEVICE" a2a3 1 seeded 2>&1 | grep -q "PTO_ISA_SENTINEL_HEADER_IS_LIVE"; then
  echo "[t2p] PROOF OK"
else
  echo "[t2p] PROOF FAILED — aborting."; exit 2
fi

for r in $(seq 1 "$ROUNDS"); do
  for phase in STOCK FIXED; do
    if [ "$phase" = FIXED ]; then apply_fix; else cp "$BAK" "$HDR"; fi
    for m in seeded random warm-canary; do
      line=$(python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" "$m" 2>&1 | grep -E "^MODE")
      echo "round $r  $phase  $line"
    done
  done
done
