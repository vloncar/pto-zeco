"""Shared benchmark/CLI harness, backend-agnostic.

These helpers are common to the AllScan collective and the ZeCO/GLA operator
benchmarks: device-id parsing, a percentile helper, and the result-table
printer. Operator-specific interfaces (``AllscanImpl``, ``ZeCoImpl``) and their
reference math live in their own packages (``allscan``, ``gla``); only the
generic plumbing lives here so both layers share one harness.
"""

from __future__ import annotations


def parse_devices(raw: str) -> list[int]:
    """Parse a ``--device`` string into an ordered, deduplicated list of ints.

    Accepts single ids (``"4"``), inclusive ranges (``"4-7"``), comma-separated
    lists (``"4,5,6"``), or any mix (``"0-2,8"``).

    Args:
        raw: The raw ``--device`` option value.

    Returns:
        Ordered, de-duplicated list of device ids.
    """
    devices: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            devices.extend(range(int(a), int(b) + 1))
        else:
            devices.append(int(token))
    return list(dict.fromkeys(devices))


def percentile(data: list[float], p: float) -> float:
    """Return the ``p``-th percentile of ``data`` (nearest-rank, unsorted input).

    Args:
        data: Sample values.
        p: Percentile in ``[0, 100]``.

    Returns:
        The percentile value.
    """
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


#: A latency sample is "dispersed" when its high or low tail is at least this many times
#: the median. Calibrated on the B5.4 samples: every ``simpler`` row sits at 1.00-1.09 and
#: the six unrepresentative ``pypto`` rows sit at 2.29-6.59, with the nearest clean pypto
#: row at 1.29 — so 2.0 separates them with room on both sides.
DISPERSION_THRESHOLD = 2.0

#: Below this many samples a distribution cannot be characterised, so ``latency_stats``
#: reports ``enough_samples=False`` and the benches print a warning. Ten is the working
#: minimum, not a target: it is what B5.4 used, and it was still only just enough to make
#: pypto's bimodality visible in the percentiles.
MIN_SAMPLES = 10


def latency_stats(samples: list[float]) -> dict:
    """Summarise latency samples with the median as the headline, plus a dispersion check.

    **Why the median, and why the extra fields.** Two measurement defects were found in the
    B5.4 numbers (see ``devtools/F66-STEP2-RESULTS.md``). One was a stopwatch that started
    at different points in different directions — fixed at the source. The other is what
    this function exists to prevent: pypto's per-call latency at ``P=2`` turns out to be
    **bimodal** (most calls ~20 ms, occasional ~120 ms), and the benchmark reported means.
    The mean of a bimodal sample describes no call that actually happened — the ``P=2``
    forward mean was 45.89 ms against a median of 21.15 ms — and every ratio derived from
    it was wrong.

    So: ``p50_ms`` is the number to quote. ``mean_ms`` is kept because dropping it would
    break existing result JSONs, but it is not the headline and must never be quoted for a
    row whose ``dispersed`` flag is set. ``spread`` is deliberately two-sided — the defect
    showed up as a **high** tail in the forward rows and a **low** tail in the backward
    ones (median 125.49 ms, minimum 23.28 ms), so a one-sided check would have missed half
    of it.

    Args:
        samples: Per-call latencies in milliseconds. Must be non-empty.

    Returns:
        Dict with ``n``, ``p05_ms``, ``p50_ms``, ``p95_ms``, ``mean_ms``, ``min_ms``,
        ``max_ms``, ``spread`` (``max(p95/p50, p50/p05)``), ``dispersed`` (bool) and
        ``enough_samples`` (bool).
    """
    assert samples, "latency_stats needs at least one sample"
    p05 = percentile(samples, 5)
    p50 = percentile(samples, 50)
    p95 = percentile(samples, 95)
    spread = max(p95 / p50 if p50 > 0 else 1.0, p50 / p05 if p05 > 0 else 1.0)
    return {
        "n": len(samples),
        "p05_ms": p05,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": sum(samples) / len(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "spread": spread,
        "dispersed": bool(spread >= DISPERSION_THRESHOLD),
        "enough_samples": bool(len(samples) >= MIN_SAMPLES),
    }


# (key, header, width, fmt) for the standard timing table shared by the benches.
BENCH_COLS = [
    ("impl", "Impl", 7, "s"),
    ("P", "P", 2, "d"),
    ("dk", "dk", 4, "d"),
    ("dv", "dv", 4, "d"),
    ("K", "K", 2, "d"),
    ("build_s", "Build(s)", 8, ".2f"),
    ("cold_ms", "Cold(ms)", 9, ".3f"),
    # p50 leads: see latency_stats for why the mean is not the headline.
    ("p50_ms", "p50(ms)", 9, ".3f"),
    ("spread", "Spread", 8, ".2f"),
    ("flag", "!", 3, "s"),
    ("p05_ms", "p05(ms)", 8, ".3f"),
    ("p95_ms", "p95(ms)", 8, ".3f"),
    ("mean_ms", "Mean(ms)", 9, ".3f"),
    ("bw_mbs", "BW(MB/s)", 9, ".2f"),
    ("correct", "OK", 4, "s"),
]


def _row_label(row: dict) -> str:
    """Identify a result row from whatever config keys it carries.

    The AllScan and GLA benches key their configs differently (``dk``/``dv``/``K`` vs
    ``L``/``C``/``D``, plus a ``dir`` on GLA rows), so build the label from what is
    present rather than assuming one schema.
    """
    parts = [str(row.get("impl", "?"))]
    if row.get("dir"):
        parts.append(str(row["dir"]))
    parts += [f"{k}={row[k]}" for k in ("P", "L", "D", "dk", "dv", "K") if k in row]
    return " ".join(parts)


def print_reading_rule(rows: list[dict]) -> None:
    """Print the latency reading rule under a results table, naming any bad rows.

    Printed with the numbers on purpose: B5.4 quoted means off a bimodal sample and every
    ratio built on them was wrong, and a rule that lives only in a document does not travel
    with the table someone pastes into a write-up.

    Args:
        rows: Result dicts as produced with ``latency_stats``.
    """
    print()
    print("Quote p50, not Mean. Spread = max(p95/p50, p50/p05); a '!' row is dispersed")
    print(f"(spread >= {DISPERSION_THRESHOLD}) — its median is NOT representative, so quote the")
    print("range or fix the variance first, and derive no ratio from it.")
    bad = [r for r in rows if r.get("dispersed")]
    if bad:
        print(f"  {len(bad)} of {len(rows)} row(s) dispersed: "
              + ", ".join(f"{_row_label(r)} ({r['spread']:.1f}x)" for r in bad))
    short = [r for r in rows if not r.get("enough_samples", True)]
    if short:
        print(f"  {len(short)} row(s) below {MIN_SAMPLES} samples — spread check unreliable.")


def print_table(rows: list[dict], cols: list[tuple] = BENCH_COLS) -> None:
    """Print a formatted timing table; footnotes any amortized-timing rows.

    Args:
        rows: Result dicts (keys matching ``cols``); a truthy ``amortized`` key
            marks a row whose timing amortizes fixed per-call setup.
        cols: Column spec as ``(key, header, width, fmt)`` tuples.
    """
    if not rows:
        return
    header = "  ".join(f"{label:>{width}}" for _, label, width, _ in cols)
    sep = "  ".join("-" * width for _, _, width, _ in cols)
    print("\n" + header)
    print(sep)
    any_amortized = False
    for row in rows:
        parts = []
        for key, _, width, fmt in cols:
            val = row[key]
            if key == "impl" and row.get("amortized"):
                val = f"{val}*"
                any_amortized = True
            if key == "correct":
                val = "?" if val is None else ("Y" if val else "N")
                parts.append(f"{val:>{width}}")
            else:
                parts.append(f"{val:{width}{fmt}}")
        print("  ".join(parts))
    print()
    if any_amortized:
        print("* timing amortizes fixed per-call orchestration setup (comm-domain "
              "alloc/free + drain) across a batch, so Mean/Min/etc. reflect the "
              "marginal kernel+comm cost rather than full per-call latency.\n")
