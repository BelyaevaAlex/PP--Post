#!/usr/bin/env python3
"""Paper Section 15: sparse/correlation-aware signed evidence.

Keeps only the top posterior rules per patient before signed-logit aggregation.
This is a practical correlation-control ablation: large tree ensembles often
produce many near-duplicate branches, so sparse evidence should be easier to
audit and less dominated by rule count.
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
    "--variants", "pp_theta_post_signed_logit,pp_theta_post_sparse_logit",
    "--output-dir", str(ROOT / "output" / "paper" / "15_pppost_sparse_logit"),
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
    out_dir = Path(_last_option(args0, "--output-dir", str(ROOT / "output" / "paper" / "15_pppost_sparse_logit")))
    topks = [k.strip() for k in os.environ.get("PPPOST_SPARSE_TOPKS", "32,64,128").split(",") if k.strip()]
    append_csv = out_dir / "compare_datasets_sparse_logit_grid.csv"
    append_jsonl = out_dir / "compare_datasets_sparse_logit_grid.jsonl"
    for topk in topks:
        rc = run_compare_datasets(
            args0
            + ["--sparse-logit-top-k", topk, "--append-results-to", str(append_csv), "--append-jsonl-to", str(append_jsonl)]
        )
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
