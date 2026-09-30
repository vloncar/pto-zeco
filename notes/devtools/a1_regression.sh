#!/usr/bin/env bash
# A1 gate: the whole pypto forward sweep (P=1 and P=2) plus the backward, on real hardware.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
rm -f /tmp/barrier_pto_multi_comm_*
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
echo "===== FORWARD ====="
timeout 5400 python3 -u -m pytest -q gla/tests/test_pypto_gla.py \
  2>&1 | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint" | tail -45
echo "forward rc=${PIPESTATUS[0]}"
rm -f /tmp/barrier_pto_multi_comm_*
echo "===== BACKWARD ====="
timeout 3600 python3 -u -m pytest -q gla/tests/test_pypto_gla_backward.py \
  2>&1 | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint" | tail -30
echo "backward rc=${PIPESTATUS[0]}"
echo ALLDONE
