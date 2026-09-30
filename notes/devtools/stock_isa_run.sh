#!/usr/bin/env bash
# Run a command against a PRISTINE pto-isa checkout instead of the patched /opt/pto-isa.
#
# We carry one local pto-isa patch (the DIR_BOTH V2C ring offset, upstream MR !1438). Any
# "pypto computes X wrong" finding has to be shown to be independent of that patch before it
# is reported as an upstream bug in something else — otherwise we are reporting our own diff.
#
# Chain it after tq_env.sh, which sets PTO_ISA_ROOT=/opt/pto-isa and then execs "$@":
#   task-submit ... "bash devtools/tq_env.sh bash devtools/stock_isa_run.sh python3 <script> ..."
set +e

STOCK="${STOCK_ISA_ROOT:-/tmp/pto-isa-stock}"
if [ ! -d "$STOCK/include/pto" ]; then
  echo "[stock_isa] $STOCK is not a pto-isa checkout; create it with:"
  echo "  cp -a /opt/pto-isa $STOCK && git -C $STOCK checkout -- include/pto/npu/a2a3/TPush.hpp"
  exit 1
fi

export PTO_ISA_ROOT="$STOCK"
n=$(grep -c V2C_ENTRY_OFFSET "$STOCK/include/pto/npu/a2a3/TPush.hpp" 2>/dev/null)
echo "[stock_isa] PTO_ISA_ROOT=$PTO_ISA_ROOT (local TPush patch present: ${n:-?} occurrences; expect 0)"
exec "$@"
