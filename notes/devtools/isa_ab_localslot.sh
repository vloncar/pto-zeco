#!/usr/bin/env bash
# A/B the consumer-local-slot-stride fix in ONE job: run the given ST cases with the fix
# applied, then with TPush.hpp reverted to upstream/master, then restore the fix.
#
# "stock" here means upstream/master, which ALREADY contains the merged DIR_BOTH ring-offset
# fix (69a81f3b) -- so the only difference between the two halves is the local slot stride.
#
# Usage: bash devtools/isa_ab_localslot.sh <repo> <case> [case ...]
set +e
REPO="$1"; shift
CASES=("$@")
HDR=include/pto/npu/a2a3/TPush.hpp

cd "$REPO" || exit 1
cp "$HDR" /tmp/TPush.fixed.hpp || exit 1

run_half() {
  local label="$1"
  echo
  echo "############### $label ###############"
  echo "[ab] SLOT_SIZE-strided local slots: $(grep -c 'LOCAL_SLOT_NUM) \*$' $HDR) split-line site(s)"
  grep -A1 "LOCAL_SLOT_NUM" "$HDR" | grep -cE "SLOT_SIZE" | sed 's/^/[ab] SLOT_SIZE occurrences after LOCAL_SLOT_NUM: /'
  bash /root/workspace/allscan/devtools/isa_st_suite.sh "$REPO" "${CASES[@]}" 2>&1 | sed -n '/SUMMARY/,$p'
}

run_half "WITH FIX (local slots strided by SLOT_SIZE)"

git checkout upstream/master -- "$HDR" || { echo "[ab] revert failed"; exit 2; }
run_half "STOCK upstream/master (local slots strided by ConsM*ConsN)"

cp /tmp/TPush.fixed.hpp "$HDR"
echo
echo "[ab] restored the fixed header"
