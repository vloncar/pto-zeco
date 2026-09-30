# task-submit: unresolved `--device auto` reaches npu-lock and kills the task

**Symptom.** A task submitted with `--device auto` dies in ~1 s, exit 1, whole log:

```
Error: invalid device_id 'auto', must be a non-negative integer
```

Intermittent. Resubmitting the identical command works.

## Root cause

`--device auto` is stored in the task record as the literal string `auto`, which the broker
is meant to rewrite to a card number. The supervisor reads it back at **`task-submit:449`**,
under a comment asserting that already happened:

```bash
# Read task metadata (DEVICE has now been resolved by the broker into a specific card number / none)
DEVICE=$(read_field DEVICE "$rf")
```

and builds the exec line at **`task-submit:514`**:

```bash
if [[ -n "$DEVICE" && "$DEVICE" != "none" && "$COMMAND" != *npu-lock* ]]; then
    exec_cmd="'$npulock' $DEVICE --timeout 0 -- bash '$task_script'"
fi
```

The guard excludes the `none` sentinel but **not the unresolved `auto`**, so when the read
races ahead of the broker's write, `npu-lock auto …` runs and `validate_device_id`
(`npu-lock:76`, requires `^[0-9]+$`) exits 1.

**`task-submit:462` has the same flaw** and sets `ASCEND_RT_VISIBLE_DEVICES=auto`,
`TASK_DEVICE=0` — a nonsense visible-device set that would silently mis-place the workload
if it ever got past `npu-lock`.

## Not card exhaustion

Starvation manifests as `pending`. Observed 2026-08-07: submission at 13:22:46 died with this
error; the identical command at 13:24:28, while another host still held all 8 cards, went
pending and then ran normally. The message never means "no free card" — which makes it
actively misleading, since it appears exactly when the box is busy.

## Fix

Validate before use, in both places — treat non-numeric as not-yet-granted and re-read or
requeue rather than exec:

```bash
if [[ "$DEVICE" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
```

## Impact

Low severity, high confusion: a one-second failure with no context, on a shared box, that
reads like starvation. Callers currently need a retry wrapper. Our `tq_env.sh` carries a
fallback for the same phenomenon one layer down ("grant raced the assign"), but it runs
*after* `npu-lock` and so cannot cover this case.
