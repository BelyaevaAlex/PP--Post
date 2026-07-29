#!/usr/bin/env python3
"""Missing-only delta for Teacher-Anchored PPtheta-Post rows.

The first teacher-anchor append batch completed most rows, but a few eICU stages
failed while appending to NFS. This wrapper reruns only grid values that still
have missing keys, writes the full temporary run to local /tmp, then appends only
rows absent from the old ``mortality_pppost_arch_v1`` CSV/JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402
from compare_datasets import write_stream_row  # noqa: E402

VARIANT = "pp_theta_post_teacher_anchored"
SOURCES = ("xgb", "tabpfn_distill_xgb")
FOLDS = ("1", "2", "3")
DATASET_ALIASES = {
    "eicu": ("eicu", "eicu_mortality_48h_tabular"),
    "mimic3": ("mimic3", "mimic3_mortality_48h_tabular"),
    "mimic4": ("mimic4", "mimic4_mortality_48h_tabular"),
}

TARGETS = {
    "pppost_short_rule_budget": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_rule_budget_grid.csv",
        "cols": ("rule_budget",),
        "grid": [
            (("256",), ["--rule-budget", "256"]),
            (("512",), ["--rule-budget", "512"]),
            (("1024",), ["--rule-budget", "1024"]),
        ],
        "extra": [
            "--rule-max-depth", "4",
            "--rule-min-support", "0.01",
            "--rule-selection", "diverse",
        ],
    },
    "pppost_sparse_logit": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_sparse_logit_grid.csv",
        "cols": ("sparse_logit_top_k",),
        "grid": [
            (("32",), ["--sparse-logit-top-k", "32"]),
            (("64",), ["--sparse-logit-top-k", "64"]),
            (("128",), ["--sparse-logit-top-k", "128"]),
        ],
    },
    "pppost_theta_shrinkage": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_theta_shrinkage_grid.csv",
        "cols": ("theta_shrinkage_strength",),
        "grid": [
            (("8",), ["--theta-shrinkage-strength", "8"]),
            (("32",), ["--theta-shrinkage-strength", "32"]),
            (("128",), ["--theta-shrinkage-strength", "128"]),
        ],
    },
    "pppost_posterior_likelihood": {
        "rule_sources": "xgb,tabpfn_distill_xgb",
        "csv": "compare_datasets_posterior_likelihood_grid.csv",
        "cols": ("condition_tau", "posterior_p_high", "posterior_p_low"),
        "grid": [
            (("1.0", "0.95", "0.05"), ["--condition-tau", "1.0", "--posterior-p-high", "0.95", "--posterior-p-low", "0.05"]),
            (("0.5", "0.95", "0.05"), ["--condition-tau", "0.5", "--posterior-p-high", "0.95", "--posterior-p-low", "0.05"]),
            (("1.0", "0.90", "0.10"), ["--condition-tau", "1.0", "--posterior-p-high", "0.90", "--posterior-p-low", "0.10"]),
            (("0.5", "0.90", "0.10"), ["--condition-tau", "0.5", "--posterior-p-high", "0.90", "--posterior-p-low", "0.10"]),
        ],
    },
}


def _norm_col(row: dict, col: str) -> str:
    value = row.get(col, "")
    if col in {"rule_budget", "sparse_logit_top_k"}:
        return str(int(float(value)))
    return f"{float(value):.12g}"


def _norm_values(values: Iterable[str], cols: Iterable[str]) -> tuple[str, ...]:
    out = []
    for value, col in zip(values, cols):
        if col in {"rule_budget", "sparse_logit_top_k"}:
            out.append(str(int(float(value))))
        else:
            out.append(f"{float(value):.12g}")
    return tuple(out)


def _row_key(row: dict, cols: tuple[str, ...]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(row.get("rule_source")),
        str(int(float(row.get("fold", "nan")))),
        tuple(_norm_col(row, col) for col in cols),
    )


def _dataset_matches(row_dataset: str, target_dataset: str) -> bool:
    return str(row_dataset) in set(DATASET_ALIASES.get(target_dataset, (target_dataset,)))


def _expected_keys(spec: dict) -> set[tuple[str, str, tuple[str, ...]]]:
    out = set()
    for values, _ in spec["grid"]:
        norm_values = _norm_values(values, spec["cols"])
        for source in SOURCES:
            for fold in FOLDS:
                out.add((source, fold, norm_values))
    return out


def _existing_keys(csv_path: Path, spec: dict, dataset: str) -> set[tuple[str, str, tuple[str, ...]]]:
    out = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _dataset_matches(row.get("dataset", ""), dataset):
                continue
            if row.get("variant") != VARIANT:
                continue
            out.add(_row_key(row, spec["cols"]))
    return out


def _append_paths(append_root: Path, dataset: str, target_stage: str, explicit_csv: str) -> tuple[Path, Path]:
    csv_path = append_root / dataset / target_stage / explicit_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"target CSV does not exist: {csv_path}")
    jsonl_path = csv_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"target JSONL does not exist: {jsonl_path}")
    return csv_path, jsonl_path


def _load_temp_rows(tmp_dir: Path) -> list[dict]:
    jsonls = sorted(tmp_dir.glob("compare_datasets*.jsonl"), key=lambda p: p.stat().st_mtime)
    rows: list[dict] = []
    for jsonl_path in jsonls:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-stage", choices=sorted(TARGETS), default=os.environ.get("TEACHER_ANCHOR_MISSING_TARGET_STAGE"))
    p.add_argument("--target-dataset", default=os.environ.get("DATASET", "eicu"))
    p.add_argument(
        "--append-root",
        default=os.environ.get(
            "PPPOST_TEACHER_ANCHOR_APPEND_ROOT",
            str(ROOT / "output" / "mortality_paper_jobs" / "pppost_arch_mortality_pppost_arch_v1"),
        ),
    )
    p.add_argument("--tmp-root", default=os.environ.get("PPPOST_MISSING_TMP_ROOT", tempfile.gettempdir()))
    p.add_argument("--keep-temp", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = build_arg_parser()
    known, passthrough = parser.parse_known_args(argv)
    if not known.target_stage:
        parser.error("--target-stage or TEACHER_ANCHOR_MISSING_TARGET_STAGE is required")
    if known.target_dataset != "eicu":
        print(f"[warn] expected eicu missing-only rerun, got dataset={known.target_dataset}")

    spec = TARGETS[known.target_stage]
    append_csv, append_jsonl = _append_paths(
        Path(known.append_root), known.target_dataset, known.target_stage, spec["csv"],
    )
    existing = _existing_keys(append_csv, spec, known.target_dataset)
    missing = _expected_keys(spec) - existing
    if not missing:
        print(f"[teacher-anchor-missing] no missing rows for {known.target_dataset}/{known.target_stage}")
        return 0

    missing_values = {key[2] for key in missing}
    print(
        f"[teacher-anchor-missing] dataset={known.target_dataset} stage={known.target_stage} "
        f"existing={len(existing)} missing={len(missing)} grids={sorted(missing_values)}"
    )

    appended_total = 0
    for values, grid_args in spec["grid"]:
        norm_values = _norm_values(values, spec["cols"])
        if norm_values not in missing_values:
            continue
        tmp_dir = Path(tempfile.mkdtemp(
            prefix=f"pppost_teacher_anchor_missing_{known.target_dataset}_{known.target_stage}_",
            dir=known.tmp_root,
        ))
        args = (
            passthrough
            + [
                "--rule-sources", spec["rule_sources"],
                "--baselines", "none",
                "--variants", VARIANT,
                "--output-dir", str(tmp_dir),
            ]
            + list(spec.get("extra", []))
            + list(grid_args)
        )
        print(
            f"[teacher-anchor-missing] running grid={norm_values} tmp={tmp_dir} "
            f"args={' '.join(grid_args)}"
        )
        rc = run_compare_datasets(args)
        if rc != 0:
            return rc
        temp_rows = _load_temp_rows(tmp_dir)
        appended = 0
        existing = _existing_keys(append_csv, spec, known.target_dataset)
        missing = _expected_keys(spec) - existing
        for row in temp_rows:
            if not _dataset_matches(row.get("dataset", ""), known.target_dataset) or row.get("variant") != VARIANT:
                continue
            key = _row_key(row, spec["cols"])
            if key not in missing:
                continue
            write_stream_row(append_csv, append_jsonl, row)
            existing.add(key)
            missing.remove(key)
            appended += 1
        appended_total += appended
        print(f"[teacher-anchor-missing] appended={appended} grid={norm_values} remaining={len(missing)}")
        if not known.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    final_missing = _expected_keys(spec) - _existing_keys(append_csv, spec, known.target_dataset)
    print(
        f"[teacher-anchor-missing] done dataset={known.target_dataset} stage={known.target_stage} "
        f"appended_total={appended_total} final_missing={len(final_missing)}"
    )
    return 0 if not final_missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
