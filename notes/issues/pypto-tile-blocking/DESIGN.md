# Task 5 / F3.1b — real tile blocking for the pypto GLA chunk kernels

## Where the ceiling is

`C = D = 64` sits at 96% of the 188416 B vector buffer. Every next size overflows, and
`C`/`D` at 128 also blows the 65536 B cube operand buffers. Measured baseline in
`../devtools/task5_baseline.log`.

## What is live, and why

Per chunk iteration of `gla_stage2` the working set is

    5 x [DK, DV]   s_run, s_run_v, kv, s_new, (+1 across the carry)
    5-6 x [C, DK]  q, k, a, la, b, qt, kb   (kbt is [DK, C])
    1 x [C, C]     tril / scores
    1 x [DK, 1]    gamma
    (+ [C, DV])    v, o_intra, o_inter, o_n

At `DK = 128` the `[C, DK]` family alone is ~164 KB of a ~184 KB budget, so **blocking DV
alone cannot work — DK has to be blocked**.

## The one hard part

Two matmuls contract over `DK`:

    o_inter = qt @ S          (contracts DK)
    scores  = qt @ kb^T       (contracts DK)

so their partial products have to be summed across DK blocks. fp32 accumulation inside the
cube is broken on a2a3 — plain accumulate drops a block and an explicit accumulate phase
faults the core, and the CPU simulator accumulates correctly so it catches neither. See
`../fp32-cube-k-accumulation/`. **The sum therefore happens in the vector unit**, exactly the
shape simpler's deferred F3.4 design landed on. One design, both backends.

## What blocks cleanly and what does not

With `DK` split into `NB` blocks of `BK`:

| quantity | over DK | why |
|---|---|---|
| `la`, `b`, `gamma`, `qt`, `kb` | **independent** | every one is elementwise or a per-column cumulative product; column `j` never reads column `j'` |
| `kv = kb^T @ v` | **independent** | DK is the output row dim, not the contraction |
| `s_new = gamma * (s_run + kv)` | **independent** | DK is the row dim |
| `o_inter = qt @ S` | **contracts** | needs a `[C, DV]` accumulator summed over blocks |
| `scores = qt @ kb^T` | **contracts** | needs a `[C, C]` accumulator summed over blocks |
| `o_intra = scores @ v` | after the fact | runs once, after `scores` is complete |

So the chunk body becomes:

    scores_acc = 0                                  # [C, C]
    o_inter_acc = 0                                 # [C, DV]
    for blk in range(NB):                           # BK-wide slice of DK
        la_b    = log(a[:, blk])
        b_b     = exp(tril @ la_b)
        gamma_b = exp(col_sum(la_b))                # [BK, 1]
        qt_b    = q[:, blk] * b_b
        kb_b    = k[:, blk] / b_b
        scores_acc  += qt_b @ kb_b^T                # vector add of a [C, C] partial
        o_inter_acc += qt_b @ S[blk, :]             # vector add of a [C, DV] partial
        S[blk, :] = gamma_b * (S[blk, :] + kb_b^T @ v)
    o_n = o_inter_acc + (scores_acc * tril) @ v

Live set per block drops to `2 x [C, BK] + [BK, DV] + [BK, 1]` plus the two accumulators
`[C, C] + [C, DV]`, which do not depend on DK at all.

## Settled: the DSL can express it (2026-08-21)

`../../../devtools/t5_block_probe.py` — a standalone DK-blocked chunk scan, checked against a
torch golden so "it compiled" is never mistaken for "it works".

Three things pypto had never been asked to do, all of which work:

1. **A loop-carried tile can be sliced.** `pl.tile.slice(s_run_v, [BK, DV], [dof, 0])` with a
   `dof` that is an affine expression of the block-loop variable.
2. **It can be rebuilt block-row at a time.** `pl.tile.assemble(s_acc, s_blk_new, [dof, 0])`,
   seeded from the whole previous state so untouched rows survive.
3. **Matmul partials sum in the vector unit.** Two accumulators carried through the block
   loop, added with `pl.add` — no cube accumulation anywhere.

**Q, K and A need no tile slicing at all**: a DK block is just a narrower `pl.load`. Only the
carried state does. That is a real simplification over what the design assumed.

### The trap: a Python `for` inside a kernel is a device loop

The first attempt unrolled the block loop in Python (`for j in range(NB)`), on the reasoning
that NB is small and static so unrolling avoids nested loop-carry. It does not compile:

    Failed to parse function 'scan': For loop has 1 iteration arguments but 0 return
    variables. They must match.

pypto parses the kernel's own **source**, so a Python `for` is read as a device loop — one
that reassigns names without `pl.yield_`, which is exactly what that message rejects. Same
for a Python `if` inside a kernel. The block loop has to be a real nested `pl.range` with its
own `init_values` / `yield_`. **Nested loop-carry works**, including a `[DK, DV]` tile carried
by the inner loop inside an outer loop that carries `[DK, DV]` too.

Seeding the accumulators needs no new host parameters: `pl.mul(tril_t, 0.0)` and
`pl.mul(v, 0.0)` give correctly-shaped zeros, the same idiom as the existing
`s_run_v = pl.mul(s_run, 1.0)` detach.

## The second consumer: the cube<->vector pipe ring

Blocking DK does far less than expected on its own. Measured at `C=64, DK=DV=128`:

| blocks | vector bytes wanted |
|---|---|
| 2 | 442624 |
| 4 | 401536 |
| 8 | 380992 |

Four times the blocking buys 14%. The reason is in the overflow message, which names it:

> The first 131072 bytes of that space are **reserved by system.reserve_buffer**, so tiles are
> allocated above them — this is the **cross-core pipe ring**.

The ring between the cube and vector units is carved off the top of the vector buffer as
`(biggest tile crossing the boundary) x (ring depth)`, *before* a single tile is allocated. At
`C=64, DV=128` that is 4 x 32768 = 131072 B of a 188416 B budget — 70% of the space gone
before the kernel asks for anything. It does not shrink with DK blocking, so it needs its own
lever.

### The knob is not where the message says it is

The message points at `pl.cross_core_slot(slot_num=N)` "on the enclosing `pl.at(...)`". That is
right for a `@pl.jit` function that opens a core-group scope. **A declared
`@pl.function(type=InCore)` has no enclosing `pl.at`**, and wrapping its body in one fails:

    InCore ScopeStmt found in non-InCore function (should have been outlined)

`ExpandMixedKernel` reads the override off a **function attribute**
(`expand_mixed_kernel_pass.cpp:1195`), so the route that works is

    @pl.function(type=pl.FunctionType.InCore, attrs={"slot_num": 2})

Verified to take effect rather than merely compile — the reserved figure moves exactly as
asked, at `C=64, DK=DV=128, NB=4`:

| ring depth | reserved | total wanted |
|---|---|---|
| 4 (default) | 131072 | 401536 |
| 2 | 65536 | 336000 |
| 1 | 32768 | 303232 |

Also worth noting: the slot literal cannot be a variable in a `pl.at` — the parser reads the
source and rejects a name with "Use pl.cross_core_slot(slot_num=4)." The attribute route takes
a normal Python value, so it sidesteps that too.

## The open question this design turns on

`S` is `[DK, DV]` and is **loop-carried across chunks**. Blocking DK only helps if a
`[BK, DV]` slice can be live instead of the whole thing. Two ways:

1. **Slice the carried tile.** Keep `S` as a loop-carry value and read/write `BK`-row slices
   of it. Whether pypto lets a carried tile be sliced in place is the thing to establish
   first — the loop-carry model has bitten us before (a `Tensor` carry silently dropped
   matmul accumulation, pypto `99d7f37c`).
2. **Put `S` in a workspace tensor.** Carry nothing; load and store `[BK, DV]` slices to a
   GM scratch tensor each block. Costs traffic, but the state is `[DK, DV]` and is touched
   once per block, so it is bounded.

Option 1 is better if it works. **Establish which, on a throwaway probe, before touching the
real kernel** — that ordering is the whole lesson from F3.1c.

## Cube operand buffers

Separate from the vector budget: `Left`/`Right` are 65536 B each, and a `[128, 128]` fp32
tile is exactly 65536 B, so `C = 128` fills an operand buffer on its own. Blocking DK to
`BK <= 64` fixes the two DK-contracting matmuls; `scores @ v` and `tril @ la` still carry a
`C`-sized operand, so `C = 128` needs its own answer. Confirm against the probe rather than
assuming — the failure message reports the legacy non-reused packing and overstates by ~2x.

## Order of work

1. Baseline table with the probe. *(done — and the probe was silently blind twice: it globbed
   a pass number that upstream renumbered, and looked for build output in a directory that
   stopped existing when devtools moved out of the repo. Both fixed.)*
2. Settle the carried-state question on a throwaway probe.
3. Block DK in `gla_stage2` only, validate against torch at a shape that already passes, so
   a regression is unambiguous.
4. Then `gla_stage1`, then the backward kernels.
5. Re-run the ceiling table; push `C`/`D` to 128.
