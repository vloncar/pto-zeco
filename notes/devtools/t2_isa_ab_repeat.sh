#!/usr/bin/env bash
# Task 2, decisive A/B WITH REPEATS: does the F3.1c local-slot fix eliminate the failure?
#
# The failure is nondeterministic (~2/3 of dispatches wrong), so the single-run A/B was
# inconclusive by construction. Each half here builds once and dispatches N times, and the
# comparison is failure RATE, not a single verdict.
#
# Patches the actual pinned tree, and phase 0 proves the header is live first.
# MUST RUN ALONE -- it mutates /opt/pto-isa, which every concurrent job compiles against.
set +e
HDR=/opt/pto-isa/include/pto/npu/a2a3/TPush.hpp
BAK=/tmp/t2r_TPush.orig.hpp
N=${N:-12}
PROBE=/root/workspace/allscan/devtools/t2_repeat_probe.py
# P=1 is enough: the failure is not P-dependent, and P=1 is the cheapest to dispatch.
CFG="1 128 64 32 64"
CTRL="1 128 64 64 64"

cp "$HDR" "$BAK" || exit 1
trap 'cp "$BAK" "$HDR"; echo "[t2r] restored the pinned header"' EXIT

echo "################ PHASE 0 — prove the header is live ################"
sed -i '1i #error PTO_ISA_SENTINEL_HEADER_IS_LIVE' "$HDR"
if python3 "$PROBE" "$TASK_DEVICE" a2a3 1 $CFG 2>&1 | grep -q "PTO_ISA_SENTINEL_HEADER_IS_LIVE"; then
  echo "[t2r] PROOF OK: sentinel reached the compiler."
else
  echo "[t2r] PROOF FAILED — aborting, the A/B would be meaningless."; exit 2
fi

echo
echo "################ PHASE A — stock (ConsM*ConsN stride) ################"
cp "$BAK" "$HDR"
python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" $CFG 2>&1 | grep -E "^RESULT|^  "
echo "--- control (dk == C), same header ---"
python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" $CTRL 2>&1 | grep -E "^RESULT"

echo
echo "################ PHASE B — F3.1c fix (SLOT_SIZE stride) ################"
cp "$BAK" "$HDR"
sed -i '/LOCAL_SLOT_NUM/ s/\* ConsM \* ConsN \* sizeof(T);/* RingFiFo::SLOT_SIZE;/' "$HDR"
echo "[t2r] SLOT_SIZE-strided sites: $(grep -c 'LOCAL_SLOT_NUM) \* RingFiFo::SLOT_SIZE' "$HDR")"
python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" $CFG 2>&1 | grep -E "^RESULT|^  "
echo "--- control (dk == C), fixed header ---"
python3 "$PROBE" "$TASK_DEVICE" a2a3 "$N" $CTRL 2>&1 | grep -E "^RESULT"
