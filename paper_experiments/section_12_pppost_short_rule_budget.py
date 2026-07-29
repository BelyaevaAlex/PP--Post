#!/usr/bin/env python3
"""Paper Section 12: short, stable, budgeted rule resources.

Runs the rule-resource experiment: truncate rules to short subpaths, filter by
support, rank by purity/support, and evaluate compact budgets.  The CSV records
``rule_budget``, ``rule_max_depth`` and ``rule_min_support`` so the table can be
reported directly as an ablation.
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
    "--variants", "source_native,pp_theta_post_frozen,pp_theta_post_signed_logit,pp_theta_post_sparse_logit",
    "--rule-max-depth", "4",
    "--rule-min-support", "0.01",
    "--rule-selection", "diverse",
    "--sparse-logit-top-k", "64",
    "--output-dir", str(ROOT / "output" / "paper" / "12_pppost_short_rule_budget"),
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
    out_dir = Path(_last_option(args0, "--output-dir", str(ROOT / "output" / "paper" / "12_pppost_short_rule_budget")))
    budgets = [b.strip() for b in os.environ.get("PPPOST_RULE_BUDGETS", "256,512,1024").split(",") if b.strip()]
    append_csv = out_dir / "compare_datasets_rule_budget_grid.csv"
    append_jsonl = out_dir / "compare_datasets_rule_budget_grid.jsonl"
    rc = 0
    for budget in budgets:
        rc = run_compare_datasets(
            args0
            + ["--rule-budget", budget, "--append-results-to", str(append_csv), "--append-jsonl-to", str(append_jsonl)]
        )
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
