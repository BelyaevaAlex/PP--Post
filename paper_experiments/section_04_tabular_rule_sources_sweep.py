#!/usr/bin/env python3
"""Paper Section 04: tabular rule-source sweep + standalone baselines.

Track A from README §10.8.1 paired with Track B (§10.8.2).  Sweeps
every registered tabular rule source (ExtraTrees, XGBoost, CatBoost,
FIGS, RuleFit) against the core PPθ-Post inference variants and adds
the four standalone competitors (EBM, FIGS, RuleFit, TabPFN) as a
ceiling reference.  The output table answers "how much does the
symbolic backbone matter?" — each rule source produces its own row in
the CSV via the ``rule_source`` column.

Run with the default datasets (sklearn:wine + breast_cancer):

    python paper_experiments/section_04_tabular_rule_sources_sweep.py

Pick a different dataset list (any spec accepted by
:mod:`compare_datasets` works):

    python paper_experiments/section_04_tabular_rule_sources_sweep.py \\
        --datasets sklearn:digits openml:adult --folds 3

Forward any other ``compare_datasets`` flag (`--epochs`, `--folds`,
`--refinement-max-samples`, …) directly on the command line.
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
    "--rule-sources", "all",
    "--baselines", "ebm,figs,rulefit,tabpfn",
    "--variants", "core",
    "--folds", "3",
    "--output-dir", str(ROOT / "output" / "paper" / "04_tabular_rule_sources"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
