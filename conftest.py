"""pytest fixtures shared by every AllScan implementation test.

Provides ``test_config`` (carrying the target ``platform``) and ``device_ids``
fixtures, plus the ``--platform`` / ``--device`` CLI options that wire the
execution target (a2a3 hardware or a2a3sim simulator) into the tests. This
conftest sits at the package root, so it is picked up whether tests are run as
``pytest`` from ``pto-allscan/`` or ``pytest tests/``.

Usage examples::

    # simulator, two virtual devices:
    pytest tests/ --platform a2a3sim --device 0,1

    # real Ascend hardware, four chips:
    pytest tests/ --platform a2a3 --device 0-3

    # a single implementation:
    pytest tests/test_simpler.py --platform a2a3sim --device 0-3
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path bootstrap — make ``allscan`` importable when pytest is invoked from
# the workspace root rather than from inside pto-allscan/.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# ---------------------------------------------------------------------------
# Minimal config object — only ``platform`` is used by the tests.
# ---------------------------------------------------------------------------


@dataclass
class TestConfig:
    """Minimal stand-in for pypto's RunConfig."""

    platform: str = "a2a3"
    device_id: int = 0


# ---------------------------------------------------------------------------
# CLI option registration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--platform",
        action="store",
        default="a2a3",
        help=(
            "Target platform: a2a3 (Ascend 910B hardware), a2a3sim (simulator), "
            "a5 (Ascend 950 hardware), a5sim (simulator). Default: a2a3."
        ),
    )
    parser.addoption(
        "--device",
        action="store",
        default="0,1",
        type=str,
        help=(
            "Device id(s) for hardware tests. Accepts a single id ('0'), an "
            "inclusive range ('0-7'), or a comma-separated list ('0,1,4'). "
            "Ranges and lists may be mixed ('0-3,8'). Default: 0,1."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_device_option(raw: str | int) -> list[int]:
    """Parse the ``--device`` option into an ordered, deduplicated list of ints.

    Accepts: single int ``"0"``, inclusive range ``"0-3"``, comma-separated
    list ``"0,1,4"``, or any combination ``"0-2,8,12-15"``.
    """
    text = str(raw).strip()
    if not text:
        raise pytest.UsageError("--device must not be empty")

    devices: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                start_str, end_str = token.split("-", 1)
                start, end = int(start_str), int(end_str)
                if end < start:
                    raise pytest.UsageError(
                        f"--device range must be non-decreasing, got {token!r}"
                    )
                devices.extend(range(start, end + 1))
            else:
                devices.append(int(token))
        except ValueError:
            raise pytest.UsageError(
                f"Invalid device ID or range in --device: {token!r}"
            ) from None

    if not devices:
        raise pytest.UsageError(f"--device yielded no device ids: {raw!r}")
    return list(dict.fromkeys(devices))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_config(request: pytest.FixtureRequest) -> TestConfig:
    """Minimal test configuration derived from CLI options."""
    platform: str = request.config.getoption("--platform")
    devices = _parse_device_option(request.config.getoption("--device"))
    return TestConfig(platform=platform, device_id=devices[0])


@pytest.fixture(scope="session")
def device_ids(request: pytest.FixtureRequest) -> list[int]:
    """Full list of device IDs from ``--device``.

    Distributed tests use this to pick a slice matching the number of ranks
    they require (e.g. ``device_ids[:P]``).
    """
    return _parse_device_option(request.config.getoption("--device"))
