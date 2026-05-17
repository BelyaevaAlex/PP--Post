#!/usr/bin/env python3
"""Paper Section 05: TabPFN distillation as a rule source.

Demonstrates the ``tabpfn_distill_{xgb,et,cb}`` rule sources from
README §10.8.3.  TabPFN is fitted as a teacher, then a tree-ensemble
student is trained on its hard argmax with confidence weighting and
branches are extracted from the student exactly like for any other
rule source.  Empirical ``class_proportions`` are re-refined against
the *original* ``(X, y)`` so cp stays sample-grounded.

This section is **interpretable end-to-end** — the student is a normal
XGBoost / ExtraTrees / CatBoost ensemble whose branches the rest of
PPθ-Post inspects.  TabPFN appears only at training time and is dropped
afterwards.

Default recipe runs all three distill flavours side-by-side against
their non-distilled siblings so the gap is visible per row.  TabPFN is
also added as a standalone Track-B baseline (the "teacher's own
ceiling").

    python paper_experiments/section_05_tabular_tabpfn_distill.py

Limit to a single distill student to speed things up:

    python paper_experiments/section_05_tabular_tabpfn_distill.py \\
        --rule-sources extratrees,tabpfn_distill_et
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


DEFAULT_ARGS = [
    "--datasets", "sklearn:wine", "sklearn:breast_cancer",
    "--rule-sources", (
        "extratrees,xgb,catboost,"
        "tabpfn_distill_xgb,tabpfn_distill_et,tabpfn_distill_cb"
    ),
    "--baselines", "tabpfn",
    "--variants", "source_native,pl_wmean,pl_full",
    "--folds", "3",
    "--output-dir", str(ROOT / "output" / "paper" / "05_tabular_tabpfn_distill"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
