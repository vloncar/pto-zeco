"""Tests for the latency summary + dispersion check (``common.harness.latency_stats``).

These exist because of a specific failure: B5.4 reported **means** for pypto's per-call
latency, which at ``P=2`` turns out to be bimodal, so the published number described no
call that ever happened and every ratio built on it was wrong. The check added to catch
that is only worth having if it can fail, so the central cases below are the **real
recorded samples** from that run — the ones that must be flagged, and the ones that must
not be. A change that stops flagging them breaks these tests.
"""

from __future__ import annotations

import pytest

from common.harness import DISPERSION_THRESHOLD, MIN_SAMPLES, latency_stats

# The actual pypto P=2 L=128 D=32 forward samples from devtools/b54_results_merged.json.
# Median 21.15 ms, mean 45.89 ms: most calls ~20 ms with occasional ~120 ms ones.
PYPTO_P2_FWD_BIMODAL = [122.06, 19.33, 21.15, 20.24, 20.87, 121.55, 19.79, 21.44, 20.55, 71.91]

# The pypto P=4 L=128 D=32 forward samples: tight, and must NOT be flagged.
PYPTO_P4_FWD_TIGHT = [224.31, 225.52, 226.04, 225.18, 230.71, 224.88, 226.61, 225.90,
                      227.33, 226.30]


def test_bimodal_sample_is_flagged():
    """The exact distribution that produced the wrong B5.4 numbers must be caught."""
    s = latency_stats(PYPTO_P2_FWD_BIMODAL)
    assert s["dispersed"], (
        f"the recorded bimodal pypto sample was not flagged (spread={s['spread']:.2f} < "
        f"{DISPERSION_THRESHOLD}); this is the exact case the check exists for")
    assert s["spread"] > 4.0
    # And the reason it matters: the mean is nowhere near the median.
    assert s["mean_ms"] > 1.8 * s["p50_ms"]


def test_tight_sample_is_not_flagged():
    """A well-behaved distribution must not be flagged, or the marker means nothing."""
    s = latency_stats(PYPTO_P4_FWD_TIGHT)
    assert not s["dispersed"], f"tight sample wrongly flagged (spread={s['spread']:.2f})"
    assert s["spread"] < 1.2


def test_dispersion_is_two_sided():
    """A LOW tail must flag too — the backward rows failed that way, not the forward way.

    pypto's P=2 backward had median 125.49 ms with a minimum of 23.28 ms: the outliers
    were fast, not slow. A one-sided p95/p50 check reads 1.01 there and sees nothing.
    """
    low_tail = [23.28, 125.49, 126.10, 124.88, 126.81, 125.02, 125.77, 126.33, 124.51, 125.60]
    s = latency_stats(low_tail)
    assert s["p95_ms"] / s["p50_ms"] < 1.1, "this sample has no high tail, by construction"
    assert s["dispersed"], "a low-tail outlier must still be flagged"


def test_headline_is_the_median_not_the_mean():
    """p50 must be the median of the samples, independent of the mean."""
    s = latency_stats([1.0, 1.0, 1.0, 1.0, 100.0])
    assert s["p50_ms"] == 1.0
    assert s["mean_ms"] == pytest.approx(20.8)


def test_sample_count_gate():
    """Too few samples cannot characterise a distribution and must say so."""
    assert not latency_stats([10.0, 10.0, 10.0])["enough_samples"]
    assert latency_stats([10.0] * MIN_SAMPLES)["enough_samples"]


def test_constant_sample_is_clean():
    """Zero variance must give spread 1.0, not a division blow-up."""
    s = latency_stats([12.5] * MIN_SAMPLES)
    assert s["spread"] == pytest.approx(1.0)
    assert not s["dispersed"]


def test_zero_latency_does_not_divide_by_zero():
    """A degenerate all-zero sample must not raise (guards the ratio arithmetic)."""
    s = latency_stats([0.0] * MIN_SAMPLES)
    assert s["spread"] == pytest.approx(1.0)


def test_empty_sample_rejected():
    with pytest.raises(AssertionError):
        latency_stats([])
