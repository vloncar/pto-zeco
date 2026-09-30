#!/usr/bin/env bash
# Does PTO_ISA_ROOT actually take effect? Two A/B attempts behaved as if it did not, and that
# was never explained. If an override is silently ignored, anyone validating a candidate
# pto-isa fix gets a false result -- worth filing. If it works, the earlier oddity was a
# log-capture artifact and there is nothing to file.
#
# Decisive: point PTO_ISA_ROOT at a COPY of the pinned tree carrying `#error`. The build must
# fail with that sentinel. Anything else means the override was ignored.
set +e
SRC=/opt/pto-isa
ALT=/tmp/isa_sentinel_copy
PROBE=/root/workspace/allscan/devtools/t2_predicate_sweep.py

rm -rf "$ALT"
cp -a "$SRC" "$ALT" || exit 1
sed -i '1i #error PTO_ISA_ROOT_OVERRIDE_REACHED_THIS_TREE' "$ALT/include/pto/npu/a2a3/TPush.hpp"

echo "=== control: default PTO_ISA_ROOT ($SRC), expect a normal run ==="
PTO_ISA_ROOT="$SRC" python3 "$PROBE" "$TASK_DEVICE" a2a3 1 1 128 32 32 32 2>&1 | tail -3

echo
echo "=== override: PTO_ISA_ROOT=$ALT (contains #error) ==="
out=$(PTO_ISA_ROOT="$ALT" python3 "$PROBE" "$TASK_DEVICE" a2a3 1 1 128 32 32 32 2>&1)
if grep -q "PTO_ISA_ROOT_OVERRIDE_REACHED_THIS_TREE" <<<"$out"; then
  echo "VERDICT: PTO_ISA_ROOT IS honoured — the sentinel reached the compiler. Nothing to file."
else
  echo "VERDICT: PTO_ISA_ROOT was IGNORED — the sentinel never reached the compiler. FILE THIS."
  echo "$out" | tail -5
fi
rm -rf "$ALT"
