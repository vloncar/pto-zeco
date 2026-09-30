# What it takes for ANY size to work, not just the ones that happen to fit

The head-dim blocking that landed makes `dk=128` reachable. That is not the goal. The goal is
that the on-core working set depends **only on block sizes** — never on `C`, `dk`, `dv` or `L` —
so `dv=1024` is a scheduling question, not a compile error.

## Where each dimension stands

| dim | bounded by | why |
|---|---|---|
| `L` (tokens/device) | **nothing** | it is the chunk-loop trip count; no tile is `L`-sized |
| `dk` | the `[dk, dv]` state | the `[C, dk]` family is blocked, but the state carry is whole |
| `dv` | the `[dk, dv]` state, `[C, dv]` tiles | not blocked at all |
| `C` | `[C, C]` and `[C, dv]` tiles | not blocked at all; at `C=128` four of them exceed the budget on their own |

So `L` is already arbitrary. The other three are not, and **the `[dk, dv]` state is the common
blocker**: it is a single tile whose size is the product of two dimensions we want unbounded.
No amount of blocking the *other* tiles helps, because the state is not one of them.

## The shape of the real answer

Three levels of blocking, plus streaming the state.

### 1. Value dim blocked OUTSIDE the chunk loop

`dv` is fully separable: every value column of the output and of the state depends only on that
value column of `V`. Nothing contracts over `dv`. So the outermost loop can be over value
blocks, each carrying its own `[dk, BV]` state across chunks:

    for i in value blocks:            # outermost -- state slice is [dk, BV]
        for n in chunks:              # carries the state for THIS value block
            ...

Cost: the gate terms (`log`, the within-chunk cumulative product, `gamma`) do not depend on `dv`,
so they are recomputed once per value block. Roughly `#value blocks` x the gate arithmetic in
exchange for dividing the state and every `[C, dv]` tile by the same factor.

### 2. Head dim blocked INSIDE the chunk (already done)

`o_inter = qt @ S` and `scores = qt @ kb^T` contract over `dk`, so their partials are summed in
the vector unit. Already shipped and hardware-validated.

### 3. Chunk rows blocked — the flash-attention part, and the real work

`[C, C]` and `[C, dv]` tiles are what make `C=128` impossible. Blocking chunk rows into `BC`
turns the within-chunk term into the standard causal-attention nest: query row block `r` needs
key row blocks `c <= r`.

Two things stop this being a mechanical change:

* **The within-chunk decay is a cumulative product down rows.** Today it is one matmul,
  `b = tril @ la`. Blocked, row block `r` needs the running sum from every earlier row block, so
  it becomes a **sequential scan** carrying a `[1, BK]` running total. `gamma` accumulates the
  same way.
* **Key rows must be revisited.** Row block `r` needs `kb[c]` for every `c <= r`, and `kb`
  depends on that block's own cumulative decay. Either recompute it (walking `c` ascending
  carries the running sum forward, so this is cheap) or spill it. Recompute is the flash choice
  and keeps the footprint flat.

The state update also has to wait: `S_new = gamma * (S + kb^T v)` uses the WHOLE-chunk `gamma`
and `kv`, and `o_inter` uses the state from the chunk's start. So `kv` accumulates over row
blocks into `[BK, BV]` and the state is updated once, after the row loop.

### 4. Streaming the state — required for arbitrary `dk`

Steps 1-3 bound everything except the state, which is still `[dk, BV]` across chunks: linear in
`dk`. To bound it, the state has to live in a global workspace and be read/written a
`[BK, BV]` block at a time. That is exactly the partial loading this is all about.

**This is the one hard dependency.** It is blocked by
[`pypto-cube-side-gm-copy`](../pypto-cube-side-gm-copy/): the pre-loop copy that seeds a state
tensor is emitted onto the cube core, which has no vector buffer, as soon as the kernel has a
nested loop. `ir.compile` returns success and a2a3sim passes, so it only shows up as a failed
device build.

**There is a plausible way around it that needs no upstream change**: do not seed the state
inside the kernel at all. Have the orchestration hand the kernel a tensor that already holds the
boundary — the ring already writes `S_recv_all[r]`, and `pl.create_tensor(..., init_value=0)`
pre-fills rank 0's on the AICPU before any kernel runs. Then the kernel only ever reads and
writes state blocks inside the loop, and the offending pre-loop copy does not exist. **Test this
before assuming the upstream fix is on the critical path.** The open question is whether
`create_tensor`'s fill happens per program-run or once per allocation — if once, a second call
starts from the previous state and is silently wrong, which needs checking, not assuming.

## Work breakdown

| # | item | depends on | size |
|---|---|---|---|
| 1 | Does an orchestration-seeded state tensor avoid the cube placement? | — | hours |
| 2 | Is `create_tensor(init_value=0)` re-filled per run? | — | hours |
| 3 | Value-dim blocking, outermost | — | ~1 day |
| 4 | Streamed `[BK, BV]` state from a workspace | 1, 2 | ~1 day |
| 5 | Chunk-row blocking: scan-based decay + causal key revisit | 3 | several days |
| 6 | Choose block sizes from a cost model rather than "first that fits" | 3, 4, 5 | ~1 day |
| 7 | Same treatment for the four backward kernels | 3-5 | ~1 week |

1 and 2 are cheap and decide whether 4 needs an upstream fix, so they go first. 3 is the best
value-per-effort: it alone makes `dv` arbitrary and cuts every `[C, dv]` tile. 5 is the big one
and is what `C` needs.

## What to stop doing

The current `blocking_plans` search — try settings, keep the first that fits — is the right shape
for a kernel whose footprint is *nearly* bounded, and the wrong shape for one that is properly
tiled. Once 3-5 land, block sizes should come from a cost model (occupancy, pipe depth, traffic)
with the budget as a constraint, not from a search over a fixed candidate list. Keep the search
until then; it is honest about failing, which a table would not be.


---

# Measured 2026-08-21: the carry is removable, and the walls move

## The state carry was the whole problem, and it is gone

`../../../devtools/t5_snapshot_math.py` — the recurrence is linear in the state, so

    S_n(boundary) = G_n * boundary + S_n(0)

and stage1 already walks the chunks from zero. If it keeps its per-chunk snapshot `S_n(0)` and
running decay `G_n`, stage2 **rebuilds** the state per chunk instead of carrying it. Exact in
double precision: 24 shapes, worst 3.6e-15. This is what pto-kernels' KDA chunk_o does
("chunks within a work item are fully independent — each reads its own s_snapshots entry"), and
it is the same trick as the cross-device AllScan, one level down.

Three consequences, in order of importance:

1. **The state becomes read-only per chunk, so it blocks.** `[BK, DV]` live instead of
   3 x `[dk, dv]` — the carry, its detached copy and the block-assembly target all disappear.
   That was 60% of the vector budget and no head-dim blocking could touch it.
2. **Chunks become independent** — a parallelism lever we do not have today.
3. It costs stage1 `N x dk x dv` of stores (256 KB at N=4, dk=dv=128) and makes stage1 live for
   P=1, where it is currently dead-code-eliminated. That is one extra dispatch at P=1.

## Where the walls are now

`../../../devtools/t5_nocarry_probe.py`, on a2a3 hardware:

| shape | blocks | result |
|---|---|---|
| C=64, dk=dv=64 | 4 | **PASS** err 3.6e-05 |
| C=64, dk=dv=128 | 8 | **PASS** err 6.1e-05 |
| C=64, dk=dv=256 | 8 | Vec 245888 |
| C=64, dk=dv=256 | 16 | Vec 204864 — over by 16448 |
| C=64, dk=dv=512 | 16 | **Right** (L0B operand) 98304 |
| C=128, dk=dv=128 | 8 | **Left** (L0A operand) 98304 |

Two distinct remaining constraints, and neither is the state any more:

**(a) The output accumulator scales with `dv`.** At `dv=256` the `[C, dv]` accumulator and its
zero seed are 65536 B each. This is exactly what value-dim blocking (item 3 above) fixes, and
`dk=dv=256` is only 16448 B over — well within reach.

**(b) The matmul OPERAND buffers, 64 KB each, at fp32 width >= 128.** A `[128,128]` fp32 tile is
65536 B — the entire buffer — so there is no room to double-buffer. This is the same constraint
the parked fp32-cube-accumulation note names ("128-wide fp32 fills a bank and forbids
ping-pong"). It is what stops `C=128` and `dk=dv=512`, and it is unrelated to everything else
measured here.

**(b) is where fp16/bf16 operands come in.** pto-kernels' flash attention uses `half` L0 tiles;
KDA chunk_o casts fp16 inputs up so the cube has fp32 sources. Halving the operand width halves
L0 pressure and restores double-buffering. For GLA the gate terms and the state must stay fp32
(the decay is an exponential of a cumulative sum, so its range is wide), but `q`, `k`, `v` and
the score matrix are candidates — which is what the reference implementations do.

## Revised order

| # | item | why now |
|---|---|---|
| 1 | Snapshot-based stage1/stage2 in the real program | proven on HW; removes the state wall and makes chunks independent |
| 2 | Value-dim blocking | the only thing between here and `dk=dv=256` |
| 3 | fp16/bf16 matmul operands where the range allows | the L0 operand wall, which blocks `C=128` and `dk=dv=512` |
| 4 | Chunk-row blocking | only worth it for `C=128/256`, and (3) may be enough on its own |

Note that (4) — the flash-attention-style row blocking I assumed was the main work — has moved
to LAST. `C=128` fails on operand width, not on working-set size, so tiling rows does not
address it; narrower operands might.


---

# Landed 2026-08-21: snapshot state, then value-dim blocking

## The "stream the state through a workspace" step never happened, and is not needed

The work breakdown above made item 4 — a `[BK, BV]` state streamed from a global workspace —
the one hard dependency, blocked on [`pypto-cube-side-gm-copy`](../pypto-cube-side-gm-copy/).
That framing was wrong, and usefully so: it assumed the state had to stay a *carry* and
therefore had to be spilled somewhere.

Dropping the carry removes the problem instead of relocating it. Once stage2 rebuilds
`S_n = G_n · boundary + S_n(0)` per chunk, the state is **read-only within a chunk**, so it is
just another operand that blocks like the rest — `[BK, BV]` live, read straight from stage1's
snapshot tensor, dead at the end of the block. No pre-loop seeding copy exists, so the cube-side
placement bug is not on this path at all. It stays worth reporting upstream (C1); it is not
blocking anything here.

The same reasoning applies to stage1, from the other direction. Nothing in stage1 contracts over
the head dim, so its blocks are independent for the *whole slice*, not merely within a chunk.
Hoisting its block loop outside the chunk scan makes its live carry `[BK, BV]` too. stage1 was
about to become the new ceiling — three copies of a `[128,128]` state are 196608 B of a
188416 B buffer — and the loop swap removed it without any spilling either.

## Value-dim blocking is a wrapper, not a second pass

The natural reading of "block `dv`" is two passes: compute the score matrix once over all head
blocks, then loop value blocks accumulating output. That is *worse* at one value block, which is
the common case — it computes the within-chunk decay twice where the old body computed it once.

Writing the value loop *around the whole chunk body* instead makes one value block bit-identical
to the previous code, and pays only when the split is real. Measured: forcing 2 and 4 value
blocks on shapes that do not need them reproduces the unsplit answer to the digit.

That asymmetry is also why the plan search varies the value split slowest. Head blocking and
ring depth were both measured to cost no latency; the value split costs recomputed gate terms.
So every setting without a value split is tried before `dv` is cut at all.

## The ones-vector trap

`pl.create_tensor(..., init_value=1)` looks like the obvious way to seed the decay accumulator.
It compiles, builds, runs, and silently delivers **zeros** when the tensor is created in a HOST
orchestrator — see [`pypto-host-init-value-zeroed`](../pypto-host-init-value-zeroed/). Note this
supersedes work item 2 in the breakdown above ("is `create_tensor(init_value=0)` re-filled per
run?"): zero fills are honoured everywhere, non-zero ones are not honoured at host level at all.
Two other obvious routes are closed as well — a one-column tile cannot be *allocated* (32-byte
row alignment), and a one-column *load* out of a wider tensor is a layout change. Deriving the
column by reducing a tile you already have is the route that works.
