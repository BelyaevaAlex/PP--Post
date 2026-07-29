#!/usr/bin/env python3
"""Paper Section 17: per-feature rule evidence reliability.

Estimates reliability from train-set feature/condition groups and uses it to
scale posterior signed evidence.  This tests whether auditable reliability
weights can close part of the practical gap without adding opaque predictors.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


DEFAULT_ARGS = [
    "--rule-sources", "xgb,tabpfn_distill_xgb",
    "--baselines", "none",
    "--variants", "pp_theta_post_signed_logit,pp_theta_post_feature_reliability",
    "--output-dir", str(ROOT / "output" / "paper" / "17_pppost_feature_reliability"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
