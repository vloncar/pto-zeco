#!/usr/bin/env bash
# Build + run one pto-isa a2a3 ST testcase on NPU, with the DIR_BOTH ring-offset fix either
# applied or reverted, so the new concurrent testcase can be shown to fail before and pass
# after.
#
# Usage: isa_st_run.sh <repo> <testcase> <stock|fixed>
#
# "stock" reverts ONLY the TPush.hpp hunk (the ST buffer stays at two rings, which is
# harmless without the fix — both directions use slot 0 regardless).
set +e
REPO="$1"; CASE="$2"; MODE="${3:-fixed}"

source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
cd "$REPO" || exit 1

# NOTE: in this clone the fix is COMMITTED, not a working-tree edit, so `git stash` is a
# no-op here (that mistake produced a "stock" run that still had the fix in it). Take the
# header from origin/master instead, which is the true pre-fix state.
if [ "$MODE" = "stock" ]; then
  git checkout "$(git rev-parse --verify -q upstream/master >/dev/null && echo upstream/master || echo origin/master)" -- include/pto/npu/a2a3/TPush.hpp && echo "[isa_st] TPush.hpp taken from upstream master (stock)"
fi
N=$(grep -c V2C_ENTRY_OFFSET include/pto/npu/a2a3/TPush.hpp)
echo "[isa_st] mode=$MODE  V2C_ENTRY_OFFSET occurrences: $N"
if { [ "$MODE" = "stock" ] && [ "$N" != "0" ]; } || { [ "$MODE" = "fixed" ] && [ "$N" = "0" ]; }; then
  echo "[isa_st] ABORT: header state does not match requested mode"; exit 2
fi

# build_st.py ONLY builds -- it does not execute anything. A green exit here says nothing
# about the test result; run_st.py is what generates the data and runs the gtest binary.
python3 tests/script/build_st.py -r npu -v a3 -t "$CASE" 2>&1 | tail -12
rc=${PIPESTATUS[0]}
echo "[isa_st] build exit=$rc"
if [ "$rc" -eq 0 ]; then
  echo "[isa_st] ---------------- RUN ----------------"
  # HARD TIMEOUT. These kernels finish in microseconds; anything past a minute is a
  # deadlock, not slowness. Without this a hung kernel spins at 100% CPU and holds the
  # device grant until the queue's own 40-minute supervisor timeout. `timeout` only kills
  # run_st.py, so the gtest binary it spawned has to be reaped separately.
  # Keep the FULL run log on disk. Piping straight to tail loses whichever case fails
  # first, and the per-element "idx: ... [ERROR]" spam is thousands of lines long, so a
  # tail big enough to keep case1 is unreadable. Filtering it inline does not work either:
  # those lines start with a real ANSI escape byte, not the literal text "[1;31m".
  FULL_LOG="${ST_LOG:-/tmp/isa_st_${CASE}_${MODE}.log}"
  timeout -k 10 "${ST_TIMEOUT:-180}" python3 tests/script/run_st.py -r npu -w -v a3 -t "$CASE" \
    > "$FULL_LOG" 2>&1
  rc=$?          # capture BEFORE anything else runs -- a later pipeline overwrites $?
  echo "[isa_st] full log: $FULL_LOG ($(wc -l < "$FULL_LOG") lines)"
  grep -avE 'idx: 0x' "$FULL_LOG" | tail -40
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "[isa_st] *** TIMEOUT after ${ST_TIMEOUT:-180}s -- the kernel HUNG (mode=$MODE) ***"
    pkill -9 -x "$CASE" 2>/dev/null && echo "[isa_st] reaped stray $CASE binary"
  fi
  echo "[isa_st] run exit=$rc  (mode=$MODE)"
fi

if [ "$MODE" = "stock" ]; then
  git checkout HEAD -- include/pto/npu/a2a3/TPush.hpp && echo "[isa_st] TPush.hpp fix restored from HEAD"
fi
exit $rc
