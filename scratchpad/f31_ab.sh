#!/usr/bin/env bash
# F3.1 A/B on one granted card: pre-F3.1 (pristine HEAD worktree) vs the working tree.
#
# The first HW attempt at the F3.1 kernels failed with `halMemCtl rc=42` inside
# `init_aicore_register_addresses` — a device bring-up error, before any kernel math runs.
# That is not something a DSL change can cause, but "not caused by my change" is a claim
# that needs evidence, so run the SAME test both ways on the SAME card.
#
# Usage: f31_ab.sh <base_tree> <new_tree> <device_csv> [platform] [test_selector]
#
# The default selector is the P=1 case: it needs one card (so it schedules on a busy box)
# and still runs gla_stage2, which is where every op F3.1 introduced lives.
set +e
BASE="$1"; NEW="$2"; DEV="$3"; PLAT="${4:-a2a3}"
TEST="${5:-gla/tests/test_pypto_gla.py::test_pypto_zeco[1]}"

for tree in "$BASE" "$NEW"; do
  label=$([ "$tree" = "$BASE" ] && echo "BASE (pre-F3.1, HEAD)" || echo "NEW  (F3.1 kernels)")
  echo
  echo "================ $label ================"
  echo "tree: $tree"
  ( cd "$tree" && PYTHONPATH="$tree:${PYTHONPATH#*:}" \
      python3 -m pytest "$TEST" --platform "$PLAT" --device "$DEV" -q 2>&1 | tail -12 )
  echo "exit=$?"
done
