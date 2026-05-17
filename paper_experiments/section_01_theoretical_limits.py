#!/usr/bin/env python3
"""Paper Section 01: theoretical analysis of model limits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study_expressivity import main as run_expressivity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "paper" / "section_01_theoretical_limits.json"),
    )
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run_expressivity(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

