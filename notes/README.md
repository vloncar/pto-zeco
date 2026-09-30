# Working notes — a snapshot, not the living copy

The live working area for this operator sits **outside** this repository, alongside it:

    ../allscan/issues/   the write-ups  (what a defect is, how it was measured, whether to file it)
    ../devtools/         the probes     (one shape, one blocking, one capability question each)

That layout is deliberate — the probes are scratch, they churn, and they are not part of the
operator. Every reference in `ROADMAP.md` and in the kernel docstrings points at those paths,
and still does.

What is here is a **copy taken on 2026-09-30**, made because the machine was about to be
powered down and rebuilt, and losing this would mean losing the reasoning behind the code
rather than the code itself. Three things in particular are hard to reconstruct:

* `issues/pypto-transpose-scratch-space/` — why the pypto pin cannot move yet, with the root
  cause traced to one line and a compile-only reproducer.
* `issues/pypto-cube-side-gm-copy/` — the defect that keeps the backward at chunk 64, and the
  allow-list test that distinguishes it from legitimate cube-side work.
* the carried patch in `issues/` for the instruction headers. Losing it silently corrupts
  every shape whose head or value dimension is smaller than the chunk.

`devtools/` here holds only this work stream's probes; the GDN and simulator-corruption
material that shares the live directory belongs to other tasks and was left out.

One thing this copy does not carry: the `.log` evidence files, because this repository
ignores `*.log`. The originals keep them, and the write-ups quote the lines that matter.

**If the live copies are still there, prefer them** — this snapshot does not track their
changes. If they are gone, restore from here and the paths above will resolve again.
