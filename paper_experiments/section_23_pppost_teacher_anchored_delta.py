#!/usr/bin/env python3
"""Delta run: append Teacher-Anchored PPtheta-Post rows to old arch CSVs.

This script intentionally runs only ``pp_theta_post_teacher_anchored`` after
the ``mortality_pppost_arch_v1`` batch. It appends the new row to each previous
section CSV/JSONL without rerunning older baselines or earlier PPtheta-Post
variants.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


TEACHER_ANCHOR_VARIANTS = "pp_theta_post_teacher_anchored"

TARGETS = {
    "pppost_teacher_rule_sources": {
        "rule_sources": "extratrees,xgb,tabpfn_distill_xgb",
        "csv": None,
        "grid": [{}],
    },
    "pppost_short_rule_budget": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_rule_budget_grid.csv",
        "grid_env": "PPPOST_RULE_BUDGETS",
        "grid_default": "256,512,1024",
        "grid_arg": "--rule-budget",
        "extra": [
            "--rule-max-depth", "4",
            "--rule-min-support", "0.01",
            "--rule-selection", "diverse",
        ],
    },
    "pppost_theta_shrinkage": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_theta_shrinkage_grid.csv",
        "grid_env": "PPPOST_THETA_STRENGTHS",
        "grid_default": "8,32,128",
        "grid_arg": "--theta-shrinkage-strength",
    },
    "pppost_signed_logit": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": None,
        "grid": [{"--signed-logit-temperature": "1.0"}],
    },
    "pppost_sparse_logit": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_sparse_logit_grid.csv",
        "grid_env": "PPPOST_SPARSE_TOPKS",
        "grid_default": "32,64,128",
        "grid_arg": "--sparse-logit-top-k",
    },
    "pppost_support_prior": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": None,
        "grid": [{}],
    },
    "pppost_feature_reliability": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": None,
        "grid": [{}],
    },
    "pppost_posterior_likelihood": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_posterior_likelihood_grid.csv",
        "posterior_grid_env": "PPPOST_POSTERIOR_GRID",
        "posterior_grid_default": "1.0:0.95:0.05,0.5:0.95:0.05,1.0:0.90:0.10,0.5:0.90:0.10",
    },
}


def _target_grid(spec: dict) -> Iterable[list[str]]:
    if "grid" in spec:
        for item in spec["grid"]:
            args: list[str] = []
            for key, value in item.items():
                args.extend([key, value])
            yield args
        return
    if "posterior_grid_env" in spec:
        raw = os.environ.get(spec["posterior_grid_env"], spec["posterior_grid_default"])
        for item in [x.strip() for x in raw.split(",") if x.strip()]:
            tau, p_high, p_low = item.split(":")
            yield [
                "--condition-tau", tau,
                "--posterior-p-high", p_high,
                "--posterior-p-low", p_low,
            ]
        return
    raw = os.environ.get(spec["grid_env"], spec["grid_default"])
    for value in [x.strip() for x in raw.split(",") if x.strip()]:
        yield [spec["grid_arg"], value]


def _append_paths(append_root: Path, dataset: str, target_stage: str, explicit_csv: str | None) -> tuple[Path, Path]:
    target_dir = append_root / dataset / target_stage
    if explicit_csv:
        csv_path = target_dir / explicit_csv
    else:
        candidates = sorted(target_dir.glob("compare_datasets*.csv"), key=lambda x: x.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"no existing compare_datasets*.csv under {target_dir}")
        csv_path = candidates[-1]
    if not csv_path.exists():
        raise FileNotFoundError(f"target CSV does not exist: {csv_path}")
    jsonl_path = csv_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"target JSONL does not exist: {jsonl_path}")
    return csv_path, jsonl_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-stage", choices=sorted(TARGETS), default=os.environ.get("TEACHER_ANCHOR_TARGET_STAGE"))
    p.add_argument("--target-dataset", default=os.environ.get("DATASET"))
    p.add_argument(
        "--append-root",
        default=os.environ.get(
            "PPPOST_TEACHER_ANCHOR_APPEND_ROOT",
            str(ROOT / "output" / "mortality_paper_jobs" / "pppost_arch_mortality_pppost_arch_v1"),
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = build_arg_parser()
    known, passthrough = parser.parse_known_args(argv)
    if not known.target_stage:
        parser.error("--target-stage or TEACHER_ANCHOR_TARGET_STAGE is required")
    if not known.target_dataset:
        parser.error("--target-dataset or DATASET is required")

    spec = TARGETS[known.target_stage]
    append_csv, append_jsonl = _append_paths(
        Path(known.append_root), known.target_dataset, known.target_stage, spec.get("csv"),
    )
    extra = list(spec.get("extra", []))
    rc = 0
    for grid_args in _target_grid(spec):
        args = (
            passthrough
            + [
                "--rule-sources", spec["rule_sources"],
                "--baselines", "none",
                "--variants", TEACHER_ANCHOR_VARIANTS,
                "--append-results-to", str(append_csv),
                "--append-jsonl-to", str(append_jsonl),
            ]
            + extra
            + grid_args
        )
        print(
            "[pppost-teacher-anchor-delta] "
            f"dataset={known.target_dataset} target_stage={known.target_stage} "
            f"append={append_csv.name} variant={TEACHER_ANCHOR_VARIANTS} "
            f"grid={' '.join(grid_args) or 'default'}"
        )
        rc = run_compare_datasets(args)
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
