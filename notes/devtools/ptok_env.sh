#!/usr/bin/env bash
# Env shim for pto-kernels builds and tests (mega-venv carries torch_npu).
set +e
source /usr/local/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann-9.0.0
export PTO_LIB_PATH=/opt/pto-isa
export PATH="/root/mega-venv/bin:${PATH}"
export PYTHONPATH="/root/workspace/allscan/pto-kernels/python:${PYTHONPATH}"
# the extension is linked with RUNPATH $ORIGIN/build/lib, which does not
# resolve from the copy inside the package; point the loader at it directly
# PTOK_LIB_OVERRIDE swaps in a saved kernel library (stock vs changed)
export LD_LIBRARY_PATH="${PTOK_LIB_OVERRIDE:-/root/workspace/allscan/pto-kernels/build/lib}:${LD_LIBRARY_PATH}"
exec "$@"
