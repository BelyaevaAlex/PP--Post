#!/usr/bin/env python3
"""Paper Section 13: empirical-Bayes theta stabilization.

Ablates theta shrinkage toward the empirical class prior.  The aim is not to
hide rule evidence, but to stabilize low-support rules while retaining the
posterior audit trail.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


DEFAULT_ARGS = [
    "--rule-sources", "xgb,tabpfn_distill_xgb",
    "--baselines", "none",
    "--variants", "pl_wmean,pp_theta_post_frozen,pp_theta_post_shrink_theta",
    "--output-dir", str(ROOT / "output" / "paper" / "13_pppost_theta_shrinkage"),
]


def _last_option(args: list[str], name: str, default: str) -> str:
    out = default
    for i, arg in enumerate(args[:-1]):
        if arg == name:
            out = args[i + 1]
    return out


def main(argv: list[str] | None = None) -> int:
    user_args = list(argv or sys.argv[1:])
    args0 = DEFAULT_ARGS + user_args
    out_dir = Path(_last_option(args0, "--output-dir", str(ROOT / "output" / "paper" / "13_pppost_theta_shrinkage")))
    strengths = [s.strip() for s in os.environ.get("PPPOST_THETA_STRENGTHS", "8,32,128").split(",") if s.strip()]
    append_csv = out_dir / "compare_datasets_theta_shrinkage_grid.csv"
    append_jsonl = out_dir / "compare_datasets_theta_shrinkage_grid.jsonl"
    for strength in strengths:
        rc = run_compare_datasets(
            args0
            + ["--theta-shrinkage-strength", strength, "--append-results-to", str(append_csv), "--append-jsonl-to", str(append_jsonl)]
        )
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
