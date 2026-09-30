# FILED 2026-08-17

Issue **hw-native-sys/pypto#2397**, PR **hw-native-sys/pypto#2398**
(branch `vloncar/pypto:fix/comm-dispatch-ordering`, rebased onto `main` `b10cae3d`).

History of the pre-filing checks is kept below.

# Pre-filing items — all resolved

Drafts are complete (`UPSTREAM-ISSUE.md`, `UPSTREAM-MR.md`, `repro.py`, `fix.patch`,
`VALIDATION.md`). Every item below is now resolved; kept for the record.

## 1. Verify `repro.py` deadlocks on STOCK pypto — RESOLVED without a card grant

Settled by proving equivalence instead of re-running: the generated `host_orch.py` for
`repro.py` is **byte-identical** to that of the probe variant measured stalling `2/2` with
`code -100` on stock, once the fix's own token lines are removed. Same dispatch sequence
(`c_compute c_send c_compute c_recv c_compute` per rank), same TaskArgs, same kernels — so it
is the same program, not a re-typing that might have drifted. Reverting the environment was
therefore unnecessary. The original revert recipe is kept below in case it is ever needed.

### (original plan, no longer required)

`repro.py` is a **standalone re-typing** of the probe variant that was measured deadlocking
(`sendfirst`, 2/2 with `code -100` on stock). It has been confirmed to *compile*, and it is
structurally the same program, but it has **never been run on a stock build**. Past experience
on this bug is explicit that a retyped "equivalent" program can quietly stop reproducing, so an
issue whose reproducer was never executed against the commit it names would be a bad filing.

Reverting to stock needs **both** halves — the `.so` alone is not enough, because the patched
`distributed_runner.py` passes `world_size=` to an `_alloc_intermediates` that stock codegen
emits without that parameter, which raises `TypeError` and would look like a third failure mode:

```bash
SP=/usr/local/python3.12.13/lib/python3.12/site-packages/pypto
cp /root/env-backup-2026-08-12/pypto_core.PRE-B4FIX.so \
   $SP/pypto_core.cpython-312-aarch64-linux-gnu.so          # md5 b748e22ee0b7c6a5fb042b689e8e7a4e
cd /opt/pypto && git stash                                   # drops all three patched files
cp python/pypto/runtime/distributed_runner.py $SP/runtime/distributed_runner.py
find $SP -name __pycache__ -type d -exec rm -rf {} +
```

Then, on a canary-green pair, expect a hang ending in `finalize_native_run failed with code
-100`. Restore afterwards with `git stash pop`, `ninja pypto_core`, copy the `.so` back, and
re-copy `distributed_runner.py` — then re-check `md5sum` is `bd8b37b08614b3e33a086fc42636fad7`
and that the reproducer passes again.

## 2. Confirm the stock commit is the right one to cite — RESOLVED

`origin/main` had moved **23 commits** past our base `71020585`. Checked: none of the three
patched files changed in that range (`git diff --stat HEAD..origin/main` on them is empty) and
the only ordering-adjacent commit, `0e25b5e0` (dual-AIV lane dispatch), is unrelated. The PR
branch is therefore cut from `origin/main` `b10cae3d` and the patch applied cleanly there; the
issue cites `71020585` as measured and notes the files are identical on current main.

### (original plan)

`git rev-parse HEAD` in `/opt/pypto` is `71020585278b68f56c72c40d5570f07dbb20bc8b` and the fix
is uncommitted working-tree edits, so HEAD is genuinely the stock commit. But **check whether
`origin/main` has moved** — if the ordering behaviour changed upstream since this pin, the issue
needs re-basing:

```bash
cd /opt/pypto && git fetch origin && git log --oneline HEAD..origin/main | head -20
```

## 3. Done already

* `pre-commit run --files <the three changed files>` — **all 17 hooks pass, no file modified**,
  so CI will not fail on formatting. Re-run after any further edit; the hooks fetch their own
  clang-format, so never hand-format.
* Citations re-derived against this checkout, not copied from notes. One claim was dropped in
  the process: `MaterializeCommDomainScopes` does **not** have a user-facing "unconsumed window
  buffer" rejection — only the two `device=` messages at
  `materialize_comm_domain_scopes_pass.cpp:249` and `:261`, which is what the drafts now cite.
* Template fields identified: `bug_report.yml` needs Component=`Codegen`, NPU Kind=
  `Ascend 910B`, Host Platform=`Linux (aarch64)`, plus the commit ID. There is **no** PR
  template in `hw-native-sys/pypto`.

## 4. Filing mechanics

pypto upstream is **GitHub** (`hw-native-sys/pypto`, fork `vloncar/pypto`, default branch
`main`) — not GitCode, so the `gc` CLI workflow in the pto-isa notes does not apply here; use
`gh`. Identity for commits stays LOCAL-only and carries **no AI attribution**.

Suggested order: file the issue, get its number, then open the PR with `Fixes #<n>` filled into
`UPSTREAM-MR.md` (it currently has a `#<ISSUE>` placeholder).
