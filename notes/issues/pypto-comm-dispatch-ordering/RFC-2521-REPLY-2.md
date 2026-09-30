Thanks — and the `_ord` outcome is exactly right. Splitting it from this RFC and gating the
removal on the manual L3 dependency API is the resolution we were hoping for.

## 3. Measurement — persistent mode confirmed, with fresh numbers

Rather than answer from the earlier run, we re-measured. Three conditions inside a single
device allocation so all three see the same four cards, on cards we first confirmed could open
connections to each other. p50 in ms, 20 samples per row, every row verified against the
sequential reference.

| shape | A cost shared over 16/dispatch | B per call | C per call + persistent | B→C | C vs A |
| --- | ---: | ---: | ---: | ---: | ---: |
| P=2 64² K=1 | 6.29 | 12.10 ⚑ | 2.61 | 4.6x | 2.4x |
| P=2 64² K=4 | 12.12 ⚑ | 11.12 | 2.52 ⚑ | 4.4x | 4.8x |
| P=4 64² K=1 | 13.24 ⚑ | 217.40 | 2.92 | 74.6x | 4.5x |
| P=4 64² K=4 | 16.23 | 219.22 | 3.05 | 71.9x | 5.3x |
| P=4 128² K=1 | 16.00 | 218.10 | 3.47 ⚑ | 62.8x | 4.6x |
| P=4 128² K=4 | 13.18 | 219.10 | **17.65** | 12.4x | **0.7x** |

⚑ marks a row whose fastest and slowest samples differ by more than 2x, so its median is not
representative.

**Your expectation holds, and the effect is larger than you suggested.** You expected the
per-call figure to collapse toward the shared-cost one. At `P=4`, persistent mode is **63-75x
faster than per call, and a further 4.5-5.3x faster than sharing the cost across a 16-exchange
batch**. Keeping the domain alive is not merely equivalent to amortising it — it is better,
because 16 exchanges on the same chips contend with each other in a way a single exchange with
a live domain does not. This is a strong argument for persistent mode as the default for any
benchmark, and it makes the `CommDomain` lifecycle cost in #2069 look even more worth
removing than the original figure suggested.

The cost is also strongly rank-dependent: at `P=2` per-call and shared-cost are already close,
while at `P=4` it is 218 ms against 13-16 ms. The overhead this RFC removes grows with rank
count, which strengthens the motivation section rather than qualifying it.

**One row does not follow the pattern.** `P=4 128² K=4` sits at 17.65 ms — only 12x better than
per call, where its neighbours are around 70x. It is the largest working set in the sweep. We
have not explained it yet, so we would not rely on the `P=4 128²` persistent figure until we
have.

### The methodology questions

**Timed region.** Compilation and the first call are excluded and reported separately, followed
by five warmup iterations. Each sample after that is one full dispatch, 20 timed iterations per
configuration. In the earlier run the dispatch created and destroyed the communication domain
on every call — that is what the per-call figure measures. The shared-cost variant issues one
dispatch containing 16 independent AllScans and divides by 16.

**Statistic — median now, mean before.** This is worth stating precisely because it changes one
of the numbers we sent. PyPTO per-call latency on this runtime is bimodal, so we now report p50
and flag any row whose spread exceeds 2x. The new run shows why: at `P=2, 64x64, K=1` without
persistent domains, the median is 12.10 ms and the mean 51.86 ms. Where the two diverge like
that the mean is not a useful summary, and five of the 18 rows above are flagged on that basis.

Against the figures in our first comment:

| figure | now |
| --- | --- |
| 67-200 ms per call at `P=4` | **Holds** — measures 217-219 ms |
| 32-40 ms with cost shared over 16 | **6-16 ms** on the median |
| hand-written and generated level at ~32 ms | not covered — this run measures the generated path only |

**One AIV per rank, on both sides.** Our PyPTO AllScan uses no `spmd`, no `core_num` and no
block indexing, and neither does the hand-written kernel it is compared against. These are all
**single-AIV** numbers and say nothing about `B`-lane scaling in either direction — they are
not evidence for or against this RFC's central claim, and we would not want them read that way.

**Payload.** FP32, state accounted as `P * dk * dv * 4` bytes; 16-64 KiB across the sweep, so
latency-sensitive rather than bandwidth-sensitive. That is the regime where your Phase 0 note
about gang-launch overhead dominating is most likely to apply.

## 1. `_ord`

Option (2), with (1) held in place until the manual L3 API is available, is exactly right, and the
`DEP_WAIT` / `DEP_WAIT | DEP_RETAIN` distinction is the narrower edge — ordering without
retention is what this needs. Two things:

- Is there an issue tracking the compiler-side L3 wiring that we can follow? We would rather
  migrate off `_ord` deliberately once that API exists than have it retired underneath us, and we are
  happy to be an early user of the manual path since our ring is the case that exercises it.
- We cannot locate `9f1536df`. It is not in `hw-native-sys/pypto` or `hw-native-sys/PTOAS` via
  the commits API, and it is not in our fetch of pypto `main`. Could you confirm the repository
  or branch? We would like to re-test the storage-base collapse against it directly, because if
  disjoint views of one backing no longer fold, that changes the advice we would give anyone
  building a per-lane control object at L2 — and we would rather verify than assume in either
  direction.

## 2. Fence

Agreed, and stating it as an ISA requirement rather than an implementation choice is stronger
than how we put it. Nothing to add.

## 5. Repetition counts — 50 is a good baseline; consider scaling it per defect class

Standardising on 50 and quoting it is the right move. One refinement, since the number that
matters is the failure rate you intend to exclude rather than the count itself. Against a
defect that fires at rate `p`, `n` runs detect it with probability `1 - (1-p)^n`:

| defect rate | 50 runs detect | runs for 95% detection |
| ---: | ---: | ---: |
| 5% (our intermittent corruption) | 92% | 59 |
| 2.6% (the measured pipe-drain-only residual) | 73% | 114 |
| 1% | 39% | 299 |

So 50 comfortably covers the class we hit, but for the publish-before-signal cases specifically
it is light — the one rate anyone has actually measured for that failure mode is the ~2.6%
residual from the `remote_store` case, and 50 runs misses that better than one time in four. It
may be worth letting the remote-visibility cases carry a higher count than the functional ones
rather than applying 50 uniformly.

**On writing the equivalent PTO code and letting PTOAS compile it** — we agree with the
diagnosis and use the A/B as a localisation technique, but our own experience does not support
the generated path being the safe side of it. Our two worst low-probability corruptions were
both in the shared ISA layer and we hit them **through** PyPTO-generated code, not in
hand-written kernels: a bidirectional cube/vector pipe indexing both direction rings from the
same base so the two overwrote each other, and a consumer's local ring striding by the popped
tile's own size while the producer used a fixed slot size, so differently-sized consecutive
tiles aliased. The second reproduced at roughly 1 in 20 and presented as an accuracy failure.
Both are fixed upstream in pto-isa now.

So the useful form of your advice, in our experience, is the differential rather than the
direction: run the same shape through both paths, and the one that disagrees localises the
defect — but either side can be the broken one, and when the defect is in the layer they share,
both will agree and still be wrong. That last case is the expensive one, and repetition count
is what catches it.

## 4 and 6

Nothing to add — agreed on both.
