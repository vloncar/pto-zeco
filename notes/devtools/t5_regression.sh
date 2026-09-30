#!/usr/bin/env bash
# Task 5 regression gate: every forward shape that worked BEFORE the blocking must still be
# correct after it. Runs the full existing sweep at P=1 and P=2 on real hardware.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
rm -f /tmp/barrier_pto_multi_comm_*
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
timeout 3000 python3 -u -m pytest -q \
  gla/tests/test_pypto_gla.py \
  2>&1 | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint" | tail -40
echo "pytest rc=${PIPESTATUS[0]}"
