#!/usr/bin/env bash
# Measure how reliably tpushpop_dir_both_concurrent fails on stock and passes with the fix.
#
# A race does not have to fail every run, and an upstream report needs the real rate rather
# than a single observation. Build once per mode, then re-run the SAME gtest binary N times:
# the goldens are already generated, so each iteration is ~2 s instead of a full rebuild.
#
# Do NOT call run_st.py in the loop -- with -w it does `rm -rf build/T*`, deleting the
# golden directories the binary reads.
set +e
# Point at a GitCode pto-isa clone on branch fix/a2a3-dir-both-ring-offset. The default is a
# throwaway session path that will not exist in a later session -- override it:
#   PTO_ISA_REPO=/path/to/pto-isa bash repro_flakiness.sh
REPO="${PTO_ISA_REPO:-/tmp/claude-0/-root-workspace-allscan/9f70ad70-920a-4d23-905e-77956a96d143/scratchpad/gitcode/pto-isa}"
if [ ! -d "$REPO/.git" ]; then
  echo "ABORT: no pto-isa clone at $REPO -- set PTO_ISA_REPO"; exit 2
fi
N="${N:-20}"
CASE1=case1_float_dir_both_concurrent
CASE2=case2_float_dir_both_concurrent_left_right

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
cd "$REPO" || exit 1

run_mode() {
  local MODE="$1"
  if [ "$MODE" = stock ]; then
    git checkout "$(git rev-parse --verify -q upstream/master >/dev/null && echo upstream/master || echo origin/master)" -- include/pto/npu/a2a3/TPush.hpp
  else
    git checkout HEAD -- include/pto/npu/a2a3/TPush.hpp
  fi
  local NOFF
  NOFF=$(grep -c V2C_ENTRY_OFFSET include/pto/npu/a2a3/TPush.hpp)
  if { [ "$MODE" = stock ] && [ "$NOFF" != 0 ]; } || { [ "$MODE" = fixed ] && [ "$NOFF" = 0 ]; }; then
    echo "ABORT: header does not match mode=$MODE"; return 2
  fi
  echo "=== mode=$MODE (V2C_ENTRY_OFFSET x$NOFF) ==="

  python3 tests/script/build_st.py -r npu -v a3 -t tpushpop_dir_both_concurrent >/dev/null 2>&1 || {
    echo "build failed"; return 1; }
  # one full pass to generate the golden data
  timeout -k 10 180 python3 tests/script/run_st.py -r npu -w -v a3 -t tpushpop_dir_both_concurrent \
    >/dev/null 2>&1

  local f1=0 f2=0 i
  for i in $(seq 1 "$N"); do
    out=$(cd tests/npu/a2a3/src/st/build/bin && timeout -k 5 60 ./tpushpop_dir_both_concurrent 2>&1)
    grep -q "FAILED  \] TPushPopDirBothConcurrentTest.$CASE1" <<<"$out" && f1=$((f1 + 1))
    grep -q "FAILED  \] TPushPopDirBothConcurrentTest.$CASE2" <<<"$out" && f2=$((f2 + 1))
  done
  echo "mode=$MODE  runs=$N  case1 failed ${f1}/${N}  case2 failed ${f2}/${N}"
}

run_mode stock
run_mode fixed
git checkout HEAD -- include/pto/npu/a2a3/TPush.hpp
echo "header restored to HEAD (fix present: $(grep -c V2C_ENTRY_OFFSET include/pto/npu/a2a3/TPush.hpp))"
