#!/usr/bin/env python3
"""Paper Section 26: clinical metrics and prediction artifacts.

Runs a focused comparison with the clinical metrics now emitted by
``compare_datasets``: AUROC, AUPRC, Brier score, ECE, sensitivity, specificity,
and decision-curve net benefit at fixed thresholds. The wrapper enables
``--save-predictions`` so downstream sections can compute patient-bootstrap CIs.

Pass mortality dataset specs and cluster options through as usual.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402

OUTPUT_DIR = ROOT / "output" / "paper" / "26_clinical_metrics"

DEFAULT_ARGS = [
    "--datasets", "sklearn:breast_cancer",
    "--rule-sources", "xgb,tabpfn_distill_xgb",
    "--baselines", "ebm,tabpfn",
    "--variants", "source_native,pp_theta_post_evidence_logit_aux,pp_theta_post_evlogit_likelihood,pp_theta_post_evlogit_threshold,pp_theta_post_teacher_anchored",
    "--folds", "3",
    "--save-predictions",
    "--output-dir", str(OUTPUT_DIR),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
