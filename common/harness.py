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


# (key, header, width, fmt) for the standard timing table shared by the benches.
BENCH_COLS = [
    ("impl", "Impl", 7, "s"),
    ("P", "P", 2, "d"),
    ("dk", "dk", 4, "d"),
    ("dv", "dv", 4, "d"),
    ("K", "K", 2, "d"),
    ("build_s", "Build(s)", 8, ".2f"),
    ("cold_ms", "Cold(ms)", 9, ".3f"),
    ("mean_ms", "Mean(ms)", 9, ".3f"),
    ("min_ms", "Min(ms)", 8, ".3f"),
    ("p50_ms", "p50(ms)", 8, ".3f"),
    ("p95_ms", "p95(ms)", 8, ".3f"),
    ("bw_mbs", "BW(MB/s)", 9, ".2f"),
    ("correct", "OK", 4, "s"),
]


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
