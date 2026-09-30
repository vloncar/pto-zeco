#!/usr/bin/env bash
# Build + run several pto-isa a2a3 ST testcases in one job and print a pass/fail summary.
#
# Used to regression-test a change to TPush.hpp against every testcase that exercises the
# cross-core pipe, before proposing it upstream. A stride change in the consumer ring could
# plausibly break cases that rely on same-size slots, and "my new case passes" is not
# evidence that the others still do.
#
# Usage: bash devtools/isa_st_suite.sh <repo> <case> [case ...]
set +e
REPO="$1"; shift
CASES=("$@")

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
cd "$REPO" || exit 1

echo "[suite] repo=$REPO"
echo "[suite] local slot stride: $(grep -c 'RingFiFo::LOCAL_SLOT_NUM) \*$' include/pto/npu/a2a3/TPush.hpp) site(s);" \
     "SLOT_SIZE-strided: $(grep -A1 'LOCAL_SLOT_NUM) \*$' include/pto/npu/a2a3/TPush.hpp | grep -c 'RingFiFo::SLOT_SIZE')"

declare -a RESULTS
for CASE in "${CASES[@]}"; do
  python3 tests/script/build_st.py -r npu -v a3 -t "$CASE" >/tmp/suite_build_$CASE.log 2>&1
  if [ $? -ne 0 ]; then
    RESULTS+=("$CASE: BUILD-FAIL")
    echo "[suite] $CASE  BUILD-FAIL"
    tail -5 /tmp/suite_build_$CASE.log
    continue
  fi
  # Hard cap: these kernels finish in milliseconds; anything past 180 s is a deadlock, and a
  # hung kernel holds the device grant. `timeout` only kills run_st.py, so reap its child.
  timeout 180 python3 tests/script/run_st.py -r npu -v a3 -t "$CASE" >/tmp/suite_run_$CASE.log 2>&1
  rc=$?
  pkill -9 -x "$CASE" 2>/dev/null
  if [ $rc -eq 124 ]; then
    RESULTS+=("$CASE: TIMEOUT")
  elif grep -q "FAILED TESTS\|\[  FAILED  \]" /tmp/suite_run_$CASE.log; then
    n=$(grep -c "\[  FAILED  \] [A-Za-z]" /tmp/suite_run_$CASE.log)
    RESULTS+=("$CASE: FAIL ($n)")
  elif grep -q "\[  PASSED  \]" /tmp/suite_run_$CASE.log; then
    n=$(grep -oE "\[  PASSED  \] [0-9]+" /tmp/suite_run_$CASE.log | tail -1 | grep -oE "[0-9]+")
    RESULTS+=("$CASE: PASS ($n)")
  else
    RESULTS+=("$CASE: NO-RESULT (rc=$rc)")
  fi
  echo "[suite] ${RESULTS[-1]}"
done

echo
echo "================ SUMMARY ================"
printf '%s\n' "${RESULTS[@]}"
bad=$(printf '%s\n' "${RESULTS[@]}" | grep -cvE ": PASS")
echo "========================================="
echo "$bad non-passing"
exit $([ "$bad" -eq 0 ] && echo 0 || echo 1)
