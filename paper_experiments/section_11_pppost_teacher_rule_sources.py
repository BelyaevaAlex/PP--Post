#!/usr/bin/env python3
"""Paper Section 11: stronger rule resources for PPtheta-Post.

Tests the main resource upgrade suggested by the reviewer-style analysis:
keep PPtheta-Post fixed, but replace the default ExtraTrees rule pool with
XGBoost rules and TabPFN-to-XGBoost distilled rules.  Rows compare the native
source, NeuralPrior, warm/aux PPtheta-Post, and the new signed/sparse evidence
heads on the same three mortality datasets in the cluster jobs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


DEFAULT_ARGS = [
    "--rule-sources", "extratrees,xgb,tabpfn_distill_xgb",
    "--baselines", "tabpfn",
    "--variants",
    (
        "source_native,neural,pp_theta_post_warm,pp_theta_post_aux,"
        "pp_theta_post_signed_logit,pp_theta_post_sparse_logit"
    ),
    "--sparse-logit-top-k", "64",
    "--output-dir", str(ROOT / "output" / "paper" / "11_pppost_teacher_rule_sources"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
