#!/usr/bin/env python3
"""Paper Section 09: temporal PPtheta-Post ablations.

Runs the fixed-L4 aggregation/head sweep and appends L2T/L3T
TabPFN-style feature-teacher ablations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal.ablations import main as run_ablations  # noqa: E402


DEFAULT_ARGS = [
    "--include-ts-feature-teacher",
    "--ts-teacher-backend",
    "tabpfn_ts",
    "--output-dir",
    str(ROOT / "output" / "paper" / "09_temporal_ablations"),
]


def main(argv: list[str] | None = None) -> int:
    return run_ablations(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
