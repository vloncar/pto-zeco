#!/usr/bin/env bash
# Is chunk_o's corruption a simulator artifact, or a latent race that hardware
# only usually hides? Same kernel, same pinned inputs, many repeats on each.
#
# Inputs and golden are computed ONCE and replayed from disk, so every repeat
# sees byte-identical input: any variation in the result is nondeterminism in
# the kernel/toolchain, not in the test data.
set +e
D=/tmp/claude-0/-root-workspace-allscan/7ebeee77-7eb5-4eea-bee4-c0b033fc3dcd/scratchpad/s7
SRC=/root/workspace/allscan/pypto-lib/models/gdn/chunk_o.py
WORK=$D/hwconf
cd /root/workspace/allscan/pypto-lib || exit 1
rm -rf "$WORK"; mkdir -p "$WORK"

sed 's/^T = 8192  /T = 4096  /' "$SRC" > "$D/hc.py"

echo "=== seeding pinned inputs + golden ==="
python "$D/hc.py" -p a2a3 -d 0 --save-data --runtime-dir "$WORK" 2>&1 \
  | grep -oE "^\[RUN\] (PASS|FAIL)|max abs diff [0-9.]+" | head -2
DATA="$WORK/data"
ls "$DATA/in" >/dev/null 2>&1 || { echo "FATAL: no pinned data at $DATA"; exit 1; }
echo "pinned data at $DATA"

run_n() {   # platform, count
  local P=$1 N=$2 pass=0 fail=0
  for i in $(seq 1 "$N"); do
    r=$(python "$D/hc.py" -p "$P" -d 0 --golden-data "$DATA" 2>&1 \
        | grep -oE "^\[RUN\] (PASS|FAIL)|max abs diff [0-9.]+" | head -2 | tr '\n' ' ')
    case "$r" in
      *PASS*) pass=$((pass+1)) ;;
      *)      fail=$((fail+1)); echo "    [$P run$i] $r" ;;
    esac
  done
  echo "=== $P: $pass pass / $fail FAIL out of $N ==="
}

run_n a2a3    40
run_n a2a3sim 12
