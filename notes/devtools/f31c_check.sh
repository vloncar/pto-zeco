#!/usr/bin/env bash
# F3.1c: is `C > D` (C=64, DK=DV=32) correct on HW with the un-shared transposes?
# All four P=1 shapes, so a fix for C>D that breaks another shape is visible immediately.
set +e
cd /root/workspace/allscan/pto-zeco || exit 1
rm -f /tmp/barrier_pto_multi_comm_*
export PYTHONPATH="/root/workspace/allscan/pto-zeco:${PYTHONPATH}"
python3 -u -m pytest -v \
  "gla/tests/test_pypto_gla.py::test_pypto_zeco_sizes[1-128-64-32-32]" \
  "gla/tests/test_pypto_gla.py::test_pypto_zeco_sizes[1-128-32-32-32]" \
  "gla/tests/test_pypto_gla.py::test_pypto_zeco_sizes[1-128-32-64-64]" \
  "gla/tests/test_pypto_gla.py::test_pypto_zeco_sizes[1-256-64-64-64]" \
  2>&1 | tail -30
