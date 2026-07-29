#!/usr/bin/env python3
"""Paper Section 18: posterior evidence likelihood calibration.

Sweeps tau and the evidence likelihood pair (p_high, p_low) for the new
configurable PPtheta heads.  Rows include the hyperparameters, so this can feed
a compact appendix table rather than an untracked rerun note.
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
    "--variants", "pp_theta_post_shrink_theta,pp_theta_post_signed_logit,pp_theta_post_sparse_logit,pp_theta_post_frozen_support_prior",
    "--sparse-logit-top-k", "64",
    "--output-dir", str(ROOT / "output" / "paper" / "18_pppost_posterior_likelihood"),
]

DEFAULT_GRID = "1.0:0.95:0.05,0.5:0.95:0.05,1.0:0.90:0.10,0.5:0.90:0.10"


def _last_option(args: list[str], name: str, default: str) -> str:
    out = default
    for i, arg in enumerate(args[:-1]):
        if arg == name:
            out = args[i + 1]
    return out


def main(argv: list[str] | None = None) -> int:
    user_args = list(argv or sys.argv[1:])
    args0 = DEFAULT_ARGS + user_args
    out_dir = Path(_last_option(args0, "--output-dir", str(ROOT / "output" / "paper" / "18_pppost_posterior_likelihood")))
    grid = [item.strip() for item in os.environ.get("PPPOST_POSTERIOR_GRID", DEFAULT_GRID).split(",") if item.strip()]
    append_csv = out_dir / "compare_datasets_posterior_likelihood_grid.csv"
    append_jsonl = out_dir / "compare_datasets_posterior_likelihood_grid.jsonl"
    for item in grid:
        tau, p_high, p_low = item.split(":")
        rc = run_compare_datasets(
            args0
            + [
                "--condition-tau", tau,
                "--posterior-p-high", p_high,
                "--posterior-p-low", p_low,
                "--append-results-to", str(append_csv),
                "--append-jsonl-to", str(append_jsonl),
            ]
        )
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
