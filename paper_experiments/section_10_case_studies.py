#!/usr/bin/env python3
"""Paper Section 10: rule-level interpretability case studies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal.case_studies import main as run_case_studies  # noqa: E402


DEFAULT_ARGS = [
    "--output-dir",
    str(ROOT / "output" / "paper" / "10_case_studies"),
]


def main(argv: list[str] | None = None) -> int:
    return run_case_studies(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

