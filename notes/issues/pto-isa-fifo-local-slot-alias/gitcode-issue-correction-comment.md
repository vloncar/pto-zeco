Two corrections to the issue text above, both from re-checking my own claims while revising
the MR.

**1. "Every shape in the existing `tmatmul` ST case has `N >= M`" is wrong.** Four of the bias
cases already run `N < M` — `TMATMULBIASTest` 101x288x67, 55x127x29, 150x89x50, 135x64x88 —
and they pass. I checked the eight non-bias shapes and assumed the bias ones followed them.

This does not touch the root cause, the `N < M` predicate, or the fix; none of them depended
on it. What it does change is the coverage story. The gap was never the shape — those cases
pass because the cube `TLOAD`s its own operands and never touches a ring. The gap is narrower:
no existing case pushes tiles of **different sizes** through one ring and holds both at once.
It also removed the reason to ship the cube-only companion testcase, which is no longer in the
MR — it was justified by the claim above, and the bias cases already cover cube-only `N < M`.

**2. The reproducer has been renamed** `tmatmul_tall_output_mix` → `tpushpop_mixed_tile_size`,
with case names saying `unequal`/`equal` rather than `tall`/`square`. The defect is a `TPipe`
FIFO bug and the matmul is only the vehicle, so it belongs with the other `tpushpop_*` cases
under a name that states the trigger condition.

MR !1457 is updated (still a single commit) and also picks up the review and code-quality
findings.
