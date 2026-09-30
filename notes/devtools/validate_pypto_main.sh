#!/usr/bin/env bash
# Local stand-in for CI's examples-tests + distributed STs, against current pypto main.
# Needs devtools/pypto_main_env.sh (isolated newer simpler + pto-isa). No hardware: a2a3sim.
#
#   PYPTO_TREE=/tmp/pypto-pr bash devtools/validate_pypto_main.sh
set +e
TREE="${PYPTO_TREE:-/tmp/pypto-pr}"
source /root/workspace/allscan/devtools/pypto_main_env.sh >/dev/null 2>&1
export PYTHONPATH="$TREE/python:$(echo "$PYTHONPATH" | sed 's#/tmp/pypto-pr/python:##')"
cd "$TREE" || exit 1
echo "=== tree=$TREE  pypto=$(python3 -c 'import pypto,os;print(os.path.dirname(pypto.__file__))')"

fail=0
run() {
    local desc="$1"; shift
    local out
    out=$(timeout 1200 "$@" 2>&1)
    if [ $? -eq 0 ]; then
        echo "PASS  $desc"
    else
        echo "FAIL  $desc"
        echo "$out" | grep -iE "AssertionError|Error|Traceback|mismatch|rc=-100" | tail -3 | sed 's/^/        /'
        fail=$((fail + 1))
    fi
}

# Plain invocations plus the flagged variants CI runs.
for f in examples/distributed/0*.py; do
    run "$(basename "$f")" python3 "$f" -p a2a3sim -d 0,1
done
run "04_barrier --use-builtin"      python3 examples/distributed/04_barrier.py -p a2a3sim -d 0,1 --use-builtin
run "05_remote_load_store --store"  python3 examples/distributed/05_remote_load_store.py -p a2a3sim -d 0,1 --mode store
run "06_put_get --get"              python3 examples/distributed/06_put_get.py -p a2a3sim -d 0,1 --mode get
run "07_dynamic_rank_count d=3"     python3 examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1,2
run "07_dynamic_rank_count d=4"     python3 examples/distributed/07_dynamic_rank_count.py -p a2a3sim -d 0,1,2,3

echo
echo "=== distributed system tests (a2a3sim) ==="
timeout 2400 python3 -m pytest tests/st/distributed -q --platform a2a3sim 2>&1 | tail -4

echo
echo "=== VERDICT: $fail example failure(s) ==="
exit $fail
