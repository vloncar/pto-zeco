#!/usr/bin/env bash
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
rm -f /tmp/barrier_pto_multi_comm_*
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export PTO_ISA_ROOT=/opt/pto-isa
export LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
timeout 1800 python3 -u -m pytest -q \
  "gla/tests/test_pypto_gla_backward.py::test_pypto_zeco_backward_sizes[2-128-32-32-32]" \
  2>&1 | grep -vE "TIMING|STRACE|^\[chip_process|perf_hint" | tail -20
echo "rc=${PIPESTATUS[0]}"
echo BWDONE
