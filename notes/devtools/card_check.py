#!/usr/bin/env python3
"""Run the canary on one card and report pass/fail. Nothing else.

Used to answer "is this card actually dead, or was it busy?" — see the rc=13 retry comment
in the runtime's `a2a3/platform/onboard/host/host_regs.cpp`, which documents halMemCtl
failing on a *driver-side serialization window* during concurrent chip_process bring-up
across paired dies. If a card that failed under load passes when the box is quiet, the
card is fine and the failure was contention, not hardware.

Usage: python3 devtools/card_check.py <device> <platform>
"""

from __future__ import annotations

import sys

from canary import BAD_CARD_EXIT, check


def main() -> int:
    device = int(sys.argv[1])
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    return 0 if check(device, platform) else BAD_CARD_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
