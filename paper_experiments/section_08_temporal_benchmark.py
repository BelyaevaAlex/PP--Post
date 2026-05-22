#!/usr/bin/env python3
"""Paper Section 08: temporal benchmark and optional vendored baselines.

Includes temporal TabPFN-TS distillation rows and the standalone
black-box TabPFN-TS baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal.compare_temporal import main as run_temporal  # noqa: E402


DEFAULT_ARGS = [
    "--levels",
    "L1", "L2", "L3", "L4",
    "--include-tabpfn-ts-distill",
    "--baselines",
    "tabpfn_ts",
    "--ts-teacher-backend",
    "tabpfn_ts",
    "--output-dir",
    str(ROOT / "output" / "paper" / "08_temporal_main"),
]


def main(argv: list[str] | None = None) -> int:
    return run_temporal(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
