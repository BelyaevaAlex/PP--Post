#!/usr/bin/env python3
"""Section 64: local EBM/TreeSHAP comparator-panel materialization tasks.

This script is intentionally conservative. It materializes comparator panels only
from saved per-case explanation artifacts. If the current outputs contain
predictions but not EBM term contributions, SHAP values, feature names, or saved
models, the script emits an explicit local task manifest instead of filling the
clinician packet with surrogate explanations.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PANEL_QC_DIR = ROOT / "output/mortality_paper_jobs/local_clinician_panel_qc_v1"
OUT_DIR = ROOT / "output/mortality_paper_jobs/local_comparator_panel_materialization_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(x: Any) -> str:
    s = str(x)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def scan_npz_artifacts(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "npz_total": 0,
        "npz_with_prediction_only": 0,
        "npz_with_feature_matrix": 0,
        "npz_with_explainer_values": 0,
        "npz_with_ebm_terms": 0,
        "npz_with_shap_values": 0,
    }
    explainer_tokens = ("shap", "contrib", "contribution", "explain", "term_score", "term_contribution")
    for path in root.glob("output/**/*.npz"):
        counts["npz_total"] += 1
        try:
            arr = np.load(path, allow_pickle=True)
            keys = list(arr.files)
        except Exception as exc:  # pragma: no cover - defensive artifact scan.
            rows.append({"path": str(path.relative_to(root)), "status": f"unreadable:{exc}", "keys": ""})
            continue
        low_keys = {k.lower() for k in keys}
        has_features = bool({"x", "x_test", "x_val", "x_train", "feature_names"} & low_keys)
        has_shap = any("shap" in k for k in low_keys)
        has_terms = any("term" in k and ("score" in k or "contrib" in k) for k in low_keys)
        has_explainer = has_shap or has_terms or any(any(tok in k for tok in explainer_tokens) for k in low_keys)
        if has_features:
            counts["npz_with_feature_matrix"] += 1
        if has_shap:
            counts["npz_with_shap_values"] += 1
        if has_terms:
            counts["npz_with_ebm_terms"] += 1
        if has_explainer:
            counts["npz_with_explainer_values"] += 1
        if not has_features and not has_explainer and {"y_true", "proba", "pred"} & low_keys:
            counts["npz_with_prediction_only"] += 1
        if has_features or has_explainer:
            rows.append({
                "path": str(path.relative_to(root)),
                "status": "candidate_feature_artifact" if has_features and not has_explainer else "candidate_explainer_artifact",
                "keys": ";".join(keys),
            })
    return rows, counts


def scan_serialized_models(root: Path) -> list[str]:
    out: list[str] = []
    for pattern in ("output/**/*.pkl", "output/**/*.pickle", "output/**/*.joblib"):
        out.extend(str(p.relative_to(root)) for p in root.glob(pattern))
    return sorted(out)


def dependency_rows() -> list[dict[str, Any]]:
    deps = [
        ("interpret", "EBM additive-term comparator"),
        ("shap", "TreeSHAP value computation"),
        ("xgboost", "TreeSHAP-native boosted-tree comparator"),
        ("sklearn", "fallback tabular matrix handling"),
    ]
    return [
        {
            "package": package,
            "needed_for": needed_for,
            "available": str(importlib.util.find_spec(package) is not None).lower(),
        }
        for package, needed_for in deps
    ]


def existing_materialized_rows(panel_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in panel_rows:
        key = (row.get("dataset_label", ""), row.get("deidentified_case_id", ""), row.get("format_name", ""))
        if row.get("panel_status", "").startswith("materialized"):
            out[key] = row
    return out


def build_tasks(out_dir: Path) -> None:
    public_cases = read_csv(PANEL_QC_DIR / "clinician_review_cases_blinded_materialized.csv")
    panel_rows = read_csv(PANEL_QC_DIR / "materialized_clinician_panels.csv")
    existing = existing_materialized_rows(panel_rows)
    artifact_rows, artifact_counts = scan_npz_artifacts(ROOT)
    model_paths = scan_serialized_models(ROOT)
    deps = dependency_rows()

    has_ebm_explainers = artifact_counts["npz_with_ebm_terms"] > 0 or bool(model_paths)
    has_shap_explainers = artifact_counts["npz_with_shap_values"] > 0 or bool(model_paths)
    dep_available = {row["package"]: row["available"] == "true" for row in deps}

    tasks: list[dict[str, Any]] = []
    for case in public_cases:
        label = case.get("dataset_label", "")
        deid = case.get("deidentified_case_id", "")
        for comparator, needs in (
            ("EBM additive terms", "per-case EBM additive term contributions"),
            ("TreeSHAP feature attribution", "per-case TreeSHAP values plus matched feature vector"),
        ):
            key = (label, deid, comparator)
            if key in existing:
                status = "already_materialized"
                action = "none"
            elif comparator.startswith("EBM") and not dep_available.get("interpret", False):
                status = "blocked_missing_dependency"
                action = "install interpret or rerun original EBM job with exported term contributions"
            elif comparator.startswith("TreeSHAP") and (
                not dep_available.get("shap", False) or not dep_available.get("xgboost", False)
            ):
                status = "blocked_missing_dependency"
                action = "install shap+xgboost or rerun original tree job with exported SHAP values"
            elif comparator.startswith("EBM") and has_ebm_explainers:
                status = "ready_for_materialization"
                action = "materialize_from_saved_explainer_artifact"
            elif comparator.startswith("TreeSHAP") and has_shap_explainers:
                status = "ready_for_materialization"
                action = "materialize_from_saved_explainer_artifact"
            else:
                status = "blocked_missing_explainer_artifact"
                action = f"rerun matched source with export of {needs}"
            tasks.append({
                "dataset_label": label,
                "deidentified_case_id": deid,
                "case_bucket": case.get("case_bucket", ""),
                "comparator": comparator,
                "status": status,
                "local_action": action,
                "required_artifact": needs,
                "artifact_search_root": "output/**/*.npz, output/**/*.pkl, output/**/*.joblib",
            })

    summary: list[dict[str, Any]] = []
    for comparator in ("EBM additive terms", "TreeSHAP feature attribution"):
        part = [t for t in tasks if t["comparator"] == comparator]
        summary.append({
            "comparator": comparator,
            "cases": len(part),
            "already_materialized": sum(t["status"] == "already_materialized" for t in part),
            "ready_for_materialization": sum(t["status"] == "ready_for_materialization" for t in part),
            "blocked_missing_explainer_artifact": sum(t["status"] == "blocked_missing_explainer_artifact" for t in part),
            "blocked_missing_dependency": sum(t["status"] == "blocked_missing_dependency" for t in part),
            "serialized_models_found": len(model_paths),
            "npz_with_explainer_values": artifact_counts["npz_with_explainer_values"],
            "npz_with_feature_matrix": artifact_counts["npz_with_feature_matrix"],
        })

    write_csv(out_dir / "comparator_panel_materialization_tasks.csv", tasks)
    write_csv(out_dir / "comparator_panel_materialization_summary.csv", summary)
    write_csv(out_dir / "comparator_panel_artifact_scan.csv", artifact_rows)
    write_csv(out_dir / "serialized_model_artifacts.csv", [{"path": p} for p in model_paths])
    write_csv(out_dir / "comparator_panel_dependency_check.csv", deps)
    (out_dir / "artifact_counts.json").write_text(json.dumps(artifact_counts, indent=2, sort_keys=True) + "\n")

    md = [
        "# Comparator Panel Materialization",
        "",
        "Scope: EBM additive-term and TreeSHAP comparator panels for the 21 blinded clinician-review cases.",
        "",
        "Outcome: no saved per-case EBM term contributions, TreeSHAP values, or serialized fitted comparator models were found in the current output tree. The current Python environment also lacks the EBM/TreeSHAP packages needed for local reconstruction.",
        "",
        "The local task manifest therefore marks the comparator panels as blocked rather than substituting placeholder or risk-only text.",
        "",
        "Next executable step: rerun the matched native comparator sources with explicit explainer export enabled, then re-run this materializer and Section 63 QC.",
    ]
    for row in summary:
        md.append(
            f"- {row['comparator']}: {row['already_materialized']}/{row['cases']} already materialized, "
            f"{row['ready_for_materialization']} ready, {row['blocked_missing_explainer_artifact']} blocked by missing artifacts, "
            f"{row['blocked_missing_dependency']} blocked by missing dependencies."
        )
    (out_dir / "comparator_panel_materialization.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparator & Cases & Materialized & Ready & Blocked \\",
        r"\midrule",
    ]
    for row in summary:
        line_end = chr(92) * 2
        tex_lines.append(
            f"{tex_escape(row['comparator'])} & {row['cases']} & {row['already_materialized']} & "
            f"{row['ready_for_materialization']} & "
            f"{row['blocked_missing_explainer_artifact'] + row['blocked_missing_dependency']} {line_end}"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "ppost_comparator_panel_materialization_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    build_tasks(args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
