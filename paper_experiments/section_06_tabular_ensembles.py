#!/usr/bin/env python3
"""Paper Section 06: ensemble variants — pl_ens_distill vs pl_ens_tabpfn.

README §10.8.4.  Both ensembles mix three members
(``α₁·teacher + α₂·PL-wmean + α₃·source_native``) with α learned per
fold via SLSQP on the probability simplex.  The difference is the
``teacher``:

* ``pl_ens_distill`` — TabPFN-distilled tree-ensemble student.  All
  three members are tree-based and have extractable branches → the
  ensemble is **end-to-end interpretable**.
* ``pl_ens_tabpfn`` — raw TabPFN as the teacher.  Usually 1-2 p.p.
  ahead on accuracy but the TabPFN contribution is **black-box**.

Both ensembles share a per-fold TabPFN cache, so adding both variants
to a run does not double TabPFN-fit cost.

Default run shows both variants side-by-side with shrinkage 0.3 (a
defence against α over-fitting on the ≤200-sample inner-val).

    python paper_experiments/section_06_tabular_ensembles.py

Try a different distill student or turn off shrinkage:

    python paper_experiments/section_06_tabular_ensembles.py \\
        --distill-student cb --ensemble-shrinkage 0.0
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
    "--rule-sources", "extratrees,catboost,ebm_terms",
    "--baselines", "tabpfn,ebm",
    "--variants", "source_native,pp_theta_post_frozen,pl_wmean,pl_ens_distill,pl_ens_tabpfn",
    "--distill-student", "et",
    "--ensemble-shrinkage", "0.3",
    "--folds", "3",
    "--output-dir", str(ROOT / "output" / "paper" / "06_tabular_ensembles"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
