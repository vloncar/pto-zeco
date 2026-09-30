#!/usr/bin/env bash
# Task 2, A/B against a reproducer that ACTUALLY REPRODUCES.
#
# The previous repeat-A/B used seed 1234 in a clean process -- the one configuration that is
# 0/12 wrong even on the stock header. Both halves came back clean and the comparison meant
# nothing. `random` mode (unseeded inputs, like the pytest harness) is 8/8 wrong on stock, so
# it can actually distinguish the two headers.
#
# Phase 0 proves the header is live. MUST RUN ALONE -- mutates /opt/pto-isa.
set +e
HDR=/opt/pto-isa/include/pto/npu/a2a3/TPush.hpp
BAK=/tmp/t2rr_TPush.orig.hpp
N=${N:-10}
PROBE=/root/workspace/allscan/devtools/t2_confound_probe.py

cp "$HDR" "$BAK" || exit 1
trap 'cp "$BAK" "$HDR"; echo "[t2rr] restored the pinned header"' EXIT

echo "################ PHASE 0 — prove the header is live ################"
sed -i '1i #error PTO_ISA_SENTINEL_HEADER_IS_LIVE' "$HDR"
if python3 "$PROBE" "$TASK_DEVICE" a2a3 1 random 2>&1 | grep -q "PTO_ISA_SENTINEL_HEADER_IS_LIVE"; then
  echo "[t2rr] PROOF OK: sentinel reached the compiler."
else
  echo "[t2rr] PROOF FAILED — aborting."; exit 2
fi

for phase in A B; do
  cp "$BAK" "$HDR"
  if [ "$phase" = B ]; then
    sed -i '/LOCAL_SLOT_NUM/ s/\* ConsM \* ConsN \* sizeof(T);/* RingFiFo::SLOT_SIZE;/' "$HDR"
    echo; echo "################ PHASE B — F3.1c fix (SLOT_SIZE stride) ################"
    echo "[t2rr] SLOT_SIZE-strided sites: $(grep -c 'LOCAL_SLOT_NUM) \* RingFiFo::SLOT_SIZE' "$HDR")"
  else
    echo; echo "################ PHASE A — stock (ConsM*ConsN stride) ################"
    echo "[t2rr] ConsM-strided sites: $(grep -c 'LOCAL_SLOT_NUM) \* ConsM' "$HDR")"
  fi
  for m in random warm-canary seeded; do
    python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" "$m" 2>&1 | grep -E "^MODE"
  done
done
