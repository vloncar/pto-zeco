# task-submit: `--env LD_PRELOAD=…` kills the task instantly (exit 137, no log)

**Symptom.** A task submitted with `--env LD_PRELOAD=<a real .so>` dies in ~1 s with
`exit=137` (SIGKILL), **no log file is ever created**, and `--list` shows the submitter as
`root@?` instead of `root@<container>`:

```
$ task-submit --timeout 120 --max-time 30 \
    --env LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so \
    --run "echo HELLO"
Task submitted: task_20260814_121705_23032031769
Tip: task-submit --help shows the card-allocation/log mechanisms and usage
=== task completed (exit=137) ===

$ task-submit --log task_20260814_121705_23032031769
Log does not exist (the task may not have started executing yet)
```

Note the missing `Waiting for task: …` line — a successful `--run` prints it, this never
gets that far. The command itself (`echo HELLO`) never runs.

`LD_PRELOAD=<libhccl.so>` is required for any HCCL/distributed workload on this box, and
`--help` advertises `--env` as the way to pass environment variables, so this is the obvious
thing to reach for. It cost us two NPU grants and a confusing debugging detour before we
noticed the correlation.

## Reproduce (no NPU needed, ~2 s)

```bash
# 1. control: --env works fine for an ordinary variable
task-submit --timeout 120 --max-time 30 --env FOO=bar --run 'echo OK; echo FOO=$FOO'
# -> OK / FOO=bar / exit=0

# 2. the leak, made visible with a library that cannot be loaded
task-submit --timeout 120 --max-time 30 --env LD_PRELOAD=/nonexistent.so --run 'echo OK'
# -> exit=0, but "ERROR: ld.so: object '/nonexistent.so' from LD_PRELOAD
#    cannot be preloaded ... ignored." is printed THREE TIMES

# 3. the failure, with a real library
task-submit --timeout 120 --max-time 30 \
  --env LD_PRELOAD=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so --run 'echo OK'
# -> exit=137 in ~1s, no log
```

## Root cause (partial): `--env` is applied to task-submit's own processes

Case 2 is the key: **one** `--env`, **three** loader complaints. The variable is not being
handed to the task alone — it is exported into the submit chain itself.

`apply_extra_env()` exports each `--env` entry into the **client process's own environment**
(**`task-submit:375`**, and the comment at **`:372`** states this is intentional):

```bash
# Apply --env / --env-file / --ptoas (via EXTRA_ENVS) to [the current client process's environment].
# spawn_supervisor re-executes task-submit via setsid; the child inherits these environment variables
# and finally passes them to the workload — replacing the old .env snapshot + runuser replay mechanism.
apply_extra_env() {
    ...
            export "$entry"
```

It is called *before* the task record is written (**`task-submit:1556`**):

```bash
        apply_extra_env
        task_id=$(submit_task "$1")
        spawn_supervisor "$task_id"
```

and `spawn_supervisor()` re-executes task-submit itself, inheriting it (**`task-submit:626`**):

```bash
# Inherits the current process environment (including variables exported by apply_extra_env), so the
# workload gets the user's live environment + --env/ptoas.
spawn_supervisor() {
    setsid "$0" --__supervise "$task_id" </dev/null >/dev/null 2>&1 &
```

So `LD_PRELOAD` is in force for the client `task-submit`, for the re-exec'd supervisor
`task-submit --__supervise`, and for every helper each of them forks — which is what the
three `ld.so` lines are showing. A variable meant for the workload is changing how
task-submit's own machinery starts.

## What we could NOT isolate

We did not find the exact process that gets SIGKILLed, and we are not going to guess at it.
Ruled out by direct test — all of these survive `LD_PRELOAD=<libhccl.so>` with rc=0:

| tested | result |
|---|---|
| `bash -c 'echo ...'` | ok |
| `setsid bash -c ...` (what `spawn_supervisor` does) | ok |
| `bash -c 'bash -c "bash -c ..."'` (3 deep) | ok |
| `hostname`, `date`, `id`, `whoami`, `uname -n` | ok |
| the exact `submit_cid` pipeline from `task-submit:344` | ok, returns the right id |

**Forensic breadcrumb.** Comparing the two `done/` records, the failed one is missing
`SUBMIT_HOST` *and* `LOG_FILE`, with `FINISH_TIME` equal to `SUBMIT_TIME`:

```
# FAILED (--env LD_PRELOAD=libhccl.so)     # OK (--env FOO=bar)
SUBMIT_USER=root                           SUBMIT_USER=root
                                           SUBMIT_HOST=6a0fe2da5009      <-- absent when failing
SUBMIT_TIME=2026-08-14T12:17:06+00:00      SUBMIT_TIME=2026-08-14T12:16:52+00:00
COMMAND=echo HELLO_C                       COMMAND=echo HELLO_B; ...
DEVICE=                                    DEVICE=
FINISH_TIME=2026-08-14T12:17:06+00:00      FINISH_TIME=2026-08-14T12:16:53+00:00
EXIT_CODE=137                              EXIT_CODE=0
                                           LOG_FILE=/var/...log          <-- absent when failing
```

The absent `SUBMIT_HOST` is what renders as `?` in `submitter_of()` (**`task-submit:1356`**),
and the absent `LOG_FILE` is why `--log` has nothing to show. Both fields are missing
entirely rather than present-and-empty, which should localise it quickly for someone who
knows which path writes the record.

## Impact

`LD_PRELOAD` is the sharpest case because the dynamic loader acts on it in *every* process,
but the same leak applies to anything read at process startup — `LD_LIBRARY_PATH`,
`LD_AUDIT`, `PYTHONPATH`, `PYTHONHOME`, `MALLOC_*`. `PYTHONPATH` is the quiet one: it would
not crash anything, it would silently change which modules task-submit's own Python helpers
import.

## Suggested fix

1. **Apply `--env` at the task's `exec`, not to the client process.** The variables should
   reach the workload without ever being in force for task-submit's own machinery. This is
   the real fix; the current design is documented in the comments, so it is a deliberate
   choice whose consequence for loader variables looks unconsidered.
2. **Cheap mitigation if (1) is too invasive:** reject or warn on loader-sensitive variables
   (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT`) with a message pointing at the wrapper-script
   approach, i.e. set them inside the submitted command instead.
3. **Regardless of the above: make this failure legible.** A task that dies with exit 137,
   no log, and `root@?` gives the user nothing to work with. Writing *any* diagnostic line
   into the log would have saved the entire investigation.

## Workaround

Set the variable inside the submitted command rather than via `--env`:

```bash
task-submit --device 0,1 --run "bash env_shim.sh bash real_job.sh"
# where env_shim.sh does: export LD_PRELOAD=<cann>/lib64/libhccl.so; exec "$@"
```

This works reliably and is what we now do for every distributed run.

## Environment

- `task-submit` at `/var/lib/taskqueue/bin/task-submit` (69905 bytes, dated 2026-08-12)
- container `6a0fe2da5009`, submitting as root
- reproduced with and without `--device`, and on both the `--device N,M` and
  `--device auto --device-num N` forms
