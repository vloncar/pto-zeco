#!/usr/bin/env bash
# Task 2 -- is the P>=2 dk!=dv failure the same defect as F3.1c?
#
# Patches the ACTUAL pinned tree (/opt/pto-isa) rather than overriding PTOAS_ROOT, because a
# PTOAS_ROOT A/B on 2026-08-12 returned byte-identical output for both halves and could not be
# trusted. Phase 0 proves the header is live before any conclusion is drawn from A vs B.
#
#   phase 0  inject `#error` into TPush.hpp -> the build MUST fail with our sentinel.
#            If it does not, the tree is not being consumed and A/B means nothing.
#   phase A  stock pinned header (ConsM*ConsN stride)
#   phase B  F3.1c fix applied (SLOT_SIZE stride)
#
# Cases: the failing shape (P=1 and P=2) plus a passing control.
set +e
HDR=/opt/pto-isa/include/pto/npu/a2a3/TPush.hpp
BAK=/tmp/t2_TPush.orig.hpp
SEL="128-64-32-64 or 256-64-64-64"

cp "$HDR" "$BAK" || exit 1
restore() { cp "$BAK" "$HDR"; echo "[t2] restored the pinned header"; }
trap restore EXIT

run_tests() {
  python3 -m pytest gla/tests/test_pypto_gla.py -q --platform a2a3 \
      --device "$TASK_DEVICE" -k "$SEL" 2>&1
}

echo "################ PHASE 0 — prove the header is live ################"
cp "$BAK" "$HDR"
sed -i '1i #error PTO_ISA_SENTINEL_HEADER_IS_LIVE' "$HDR"
out=$(run_tests)
if grep -q "PTO_ISA_SENTINEL_HEADER_IS_LIVE" <<<"$out"; then
  echo "[t2] PROOF OK: the sentinel reached the compiler, so $HDR is the header in use."
else
  echo "[t2] PROOF FAILED: sentinel never appeared. The A/B below is meaningless."
  echo "$out" | tail -20
  exit 2
fi

echo
echo "################ PHASE A — stock pinned header ################"
cp "$BAK" "$HDR"
grep -c "LOCAL_SLOT_NUM) \* ConsM" "$HDR" | sed 's/^/[t2] ConsM-strided sites: /'
run_tests | tail -8

echo
echo "################ PHASE B — with the F3.1c SLOT_SIZE fix ################"
cp "$BAK" "$HDR"
sed -i '/LOCAL_SLOT_NUM/ s/\* ConsM \* ConsN \* sizeof(T);/* RingFiFo::SLOT_SIZE;/' "$HDR"
grep -c "LOCAL_SLOT_NUM) \* RingFiFo::SLOT_SIZE" "$HDR" | sed 's/^/[t2] SLOT_SIZE-strided sites: /'
run_tests | tail -8
