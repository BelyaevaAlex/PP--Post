#!/usr/bin/env python3
"""Paper Section 16: frozen empirical support prior.

Compares the original frozen condition prior against a fully interpretable
empirical branch-support prior.  The latter uses train-set rule coverage as
P(z_b) and updates it with patient evidence at test time.
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
    "--variants", "pp_theta_post_frozen,pp_theta_post_frozen_support_prior",
    "--output-dir", str(ROOT / "output" / "paper" / "16_pppost_support_prior"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
