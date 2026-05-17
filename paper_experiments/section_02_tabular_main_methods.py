#!/usr/bin/env python3
"""Paper Section 02: tabular benchmark for the main PPθ-Post methods.

Headline "the method works" run.  Sweeps the core inference variants
plus the two expensive end-to-end variants (``pp_theta_post_e2e``,
``e2e_noisy_or``) on the default ExtraTrees rule source.  Sections 03
(ablations), 04 (rule sources), 05 (TabPFN distill), 06 (ensembles),
and 07 (interpretability story) refine and extend it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


DEFAULT_ARGS = [
    "--variants",
    "core,pp_theta_post_e2e,e2e_noisy_or",
    "--output-dir",
    str(ROOT / "output" / "paper" / "02_tabular_main"),
]


def main(argv: list[str] | None = None) -> int:
    return run_compare_datasets(DEFAULT_ARGS + list(argv or sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

