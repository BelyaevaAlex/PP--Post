#!/usr/bin/env python3
"""Paper Section 03: tabular PPθ-Post inference ablations.

Mirror of section_09 (temporal ablations), but for the tabular pipeline.
Holds the symbolic backbone fixed (default ExtraTrees) and sweeps every
inference variant — the 8 cheap ``core`` variants
(``source_native``, ``neural``, ``condition_wmean``, ``hybrid_wmean``,
``hybrid_noisy_or``, ``pl_fast``, ``pl_full``, ``pl_wmean``) plus the
expensive end-to-end variants (``theta_learn``, ``pp_theta_post_e2e``,
``pp_theta_post_warm``, ``pp_theta_post_aux``,
``pp_theta_post_learn_evidence``, ``e2e_noisy_or``,
``calibrated_e2e_noisy_or`` and ensemble variants). Each row tells you exactly
which component of PPθ-Post is doing the work on a given dataset.

To isolate "what would I lose without component X?", read the gap
between sibling rows (e.g. ``pl_fast`` vs ``pl_full`` quantifies the
posterior update; ``pl_full`` vs ``pl_wmean`` quantifies noisy-or vs
weighted-mean aggregation; ``pp_theta_post_aux`` vs
``pp_theta_post_e2e`` quantifies branch-truth supervision; and
``pp_theta_post_learn_evidence`` tests whether evidence reliability should
be learned instead of fixed).

Default datasets: wine + breast_cancer + digits.  Override with
``--datasets`` for paper-scale runs.

    python paper_experiments/section_03_tabular_ablations.py
    python paper_experiments/section_03_tabular_ablations.py \\
        --datasets openml:adult --folds 3 --epochs 80
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


DEFAULT_ARGS = [
    "--datasets", "sklearn:wine", "sklearn:breast_cancer", "sklearn:digits",
    "--rule-sources", "extratrees,ebm_terms",
    "--variants", "all",
    "--folds", "3",
    "--output-dir", str(ROOT / "output" / "paper" / "03_tabular_ablations"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
