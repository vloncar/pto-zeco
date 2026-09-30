#!/usr/bin/env bash
set +e
D=/tmp/claude-0/-root-workspace-allscan/7ebeee77-7eb5-4eea-bee4-c0b033fc3dcd/scratchpad/s7
cd /root/workspace/allscan/pypto-lib || exit 1
echo "############ lad_nosplit raw tail ############"
python "$D/lad_nosplit.py" -p a2a3 -d 0 2>&1 | tail -18
