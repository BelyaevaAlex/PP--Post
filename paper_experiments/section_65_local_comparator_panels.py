#!/usr/bin/env python3
"""Section 65: locally reconstructed EBM and TreeSHAP comparator panels.

The original clinician packet contained placeholders for EBM and TreeSHAP
comparators, and Section 64 verifies that the old runs did not save fitted
explainer artifacts. This script reconstructs comparator explanations locally on
saved measurement-policy feature matrices. These panels are for the blinded
clinician/user review packet only; they are not used as benchmark results.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PANEL_QC_DIR = ROOT / "output/mortality_paper_jobs/local_clinician_panel_qc_v1"
OUT_DIR = ROOT / "output/mortality_paper_jobs/local_clinician_panel_qc_with_comparators_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"

DATASET_TO_NPZ = {
    "eICU": ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_reviewer_stress_mortality_aaai_reviewer_stress_v1/eicu/rahmatullaev_stress_measurement_policy_v2/measurement_policy_npz/eicu_measurement_policy_only.npz",
    "MIMIC-III": ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_reviewer_stress_mortality_aaai_reviewer_stress_v1/mimic3/rahmatullaev_stress_measurement_policy_v2/measurement_policy_npz/mimic3_measurement_policy_only.npz",
    "MIMIC-IV": ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_reviewer_stress_mortality_aaai_reviewer_stress_v1/mimic4/rahmatullaev_stress_measurement_policy_v2/measurement_policy_npz/mimic4_measurement_policy_only.npz",
}


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


def clean_feature(name: Any) -> str:
    s = str(name)
    s = re.sub(r"__", " ", s)
    s = s.replace("_", " ")
    s = s.replace("frac obs", "fraction observed")
    s = s.replace("mean bp", "mean blood pressure")
    s = s.replace("systolic bp", "systolic blood pressure")
    s = s.replace("diastolic bp", "diastolic blood pressure")
    s = s.replace("bun", "blood urea nitrogen")
    s = s.replace("wbc", "white blood cell count")
    return s.strip()


def fmt_value(x: float) -> str:
    if not math.isfinite(float(x)):
        return "missing"
    ax = abs(float(x))
    if ax >= 100:
        return f"{float(x):.0f}"
    if ax >= 10:
        return f"{float(x):.1f}"
    return f"{float(x):.2f}"


def top_terms(names: list[str], values: np.ndarray, contrib: np.ndarray, k: int = 4) -> tuple[list[str], list[str]]:
    contrib = np.asarray(contrib, dtype=float)
    pos = [i for i in np.argsort(-contrib) if contrib[i] > 0][:k]
    neg = [i for i in np.argsort(contrib) if contrib[i] < 0][:k]

    def line(i: int) -> str:
        sign = "+" if contrib[i] >= 0 else ""
        return f"{clean_feature(names[i])}={fmt_value(values[i])} ({sign}{contrib[i]:.3f})"

    return [line(i) for i in pos], [line(i) for i in neg]


def dependency_available() -> dict[str, bool]:
    return {pkg: importlib.util.find_spec(pkg) is not None for pkg in ("interpret", "shap", "xgboost", "sklearn")}


def train_case_comparators(dataset_label: str, case_ids: list[int], max_train: int, seed: int) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    deps = dependency_available()
    if not all(deps[pkg] for pkg in ("interpret", "shap", "xgboost", "sklearn")):
        return {}, {"dataset_label": dataset_label, "status": "blocked_missing_dependency", **{f"dep_{k}": v for k, v in deps.items()}}

    from interpret.glassbox import ExplainableBoostingClassifier
    import shap
    import xgboost as xgb

    path = DATASET_TO_NPZ[dataset_label]
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    y = np.asarray(z["y"], dtype=int)
    names = [str(x) for x in z["feature_names"]]
    valid_case_ids = sorted({i for i in case_ids if 0 <= i < len(y)})
    rng = np.random.default_rng(seed)
    train_size = min(max_train, len(y))
    train_idx = rng.choice(len(y), size=train_size, replace=False)
    train_idx = np.unique(np.concatenate([train_idx, np.asarray(valid_case_ids, dtype=int)]))
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_cases = X[valid_case_ids]

    ebm = ExplainableBoostingClassifier(
        feature_names=names,
        interactions=0,
        max_rounds=120,
        random_state=seed,
        n_jobs=1,
    )
    ebm.fit(X_train, y_train)
    ebm_exp = ebm.explain_local(X_cases).data
    ebm_proba = ebm.predict_proba(X_cases)[:, 1]

    tree = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
    )
    tree.fit(X_train, y_train)
    tree_proba = tree.predict_proba(X_cases)[:, 1]
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(X_cases)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    panels: dict[int, dict[str, str]] = {}
    for pos, case_id in enumerate(valid_case_ids):
        x_row = X_cases[pos]
        ebm_data = ebm_exp(pos)
        ebm_scores = np.asarray(ebm_data.get("scores", []), dtype=float)
        ebm_names = [str(x) for x in ebm_data.get("names", names)]
        if len(ebm_scores) != len(names):
            ebm_scores = np.resize(ebm_scores, len(names))
            ebm_names = names
        ebm_pos, ebm_neg = top_terms(ebm_names, x_row, ebm_scores)
        shap_pos, shap_neg = top_terms(names, x_row, shap_values[pos])
        panels[case_id] = {
            "EBM additive terms": (
                f"EBM additive-term comparator. Mortality risk {ebm_proba[pos]:.3f}. "
                f"Largest increasing terms: {'; '.join(ebm_pos) if ebm_pos else 'none'}. "
                f"Largest decreasing terms: {'; '.join(ebm_neg) if ebm_neg else 'none'}."
            ),
            "TreeSHAP feature attribution": (
                f"TreeSHAP comparator from a locally trained gradient-boosted tree. Mortality risk {tree_proba[pos]:.3f}. "
                f"Largest increasing attributions: {'; '.join(shap_pos) if shap_pos else 'none'}. "
                f"Largest decreasing attributions: {'; '.join(shap_neg) if shap_neg else 'none'}."
            ),
        }
    meta = {
        "dataset_label": dataset_label,
        "status": "completed",
        "matrix_path": str(path.relative_to(ROOT)),
        "rows": len(y),
        "features": X.shape[1],
        "requested_cases": len(case_ids),
        "materialized_cases": len(valid_case_ids),
        "train_rows": len(train_idx),
        "max_train": max_train,
        **{f"dep_{k}": v for k, v in deps.items()},
    }
    return panels, meta


def panel_qc(text: str) -> str:
    issues: list[str] = []
    if len(text.strip()) < 80:
        issues.append("too_short")
    if re.search(r"rahmatullaev|\.json|\.csv|mimic[34]|eicu|pp_theta|tabpfn", text, flags=re.I):
        issues.append("internal_names")
    return "pass" if not issues else ";".join(issues)


def run(output_dir: Path, max_train: int, seed: int) -> None:
    base_panels = read_csv(PANEL_QC_DIR / "materialized_clinician_panels.csv")
    base_blinded = read_csv(PANEL_QC_DIR / "clinician_review_cases_blinded_materialized.csv")
    base_key = read_csv(PANEL_QC_DIR / "clinician_review_key_private_materialized.csv")
    replay = read_csv(PANEL_QC_DIR / "clinician_packet_replay_validity.csv")
    case_id_by_key = {
        (r["dataset_label"], r["deidentified_case_id"]): int(float(r["source_case_id_private"]))
        for r in replay
        if str(r.get("source_case_id_private", "")).strip()
    }

    panels_by_dataset: dict[str, dict[int, dict[str, str]]] = {}
    meta_rows: list[dict[str, Any]] = []
    for label in DATASET_TO_NPZ:
        ids = [cid for (d, _), cid in case_id_by_key.items() if d == label]
        panels, meta = train_case_comparators(label, ids, max_train=max_train, seed=seed)
        panels_by_dataset[label] = panels
        meta_rows.append(meta)

    updated_panels: list[dict[str, Any]] = []
    for row in base_panels:
        out = dict(row)
        fmt = row.get("format_name", "")
        key = (row.get("dataset_label", ""), row.get("deidentified_case_id", ""))
        case_id = case_id_by_key.get(key)
        if fmt in {"EBM additive terms", "TreeSHAP feature attribution"} and case_id is not None:
            text = panels_by_dataset.get(key[0], {}).get(case_id, {}).get(fmt)
            if text:
                out["panel_text"] = text
                out["panel_status"] = "materialized_local_reconstruction"
                out["qc_status"] = panel_qc(text)
            else:
                out["panel_status"] = "blocked_missing_local_reconstruction"
                out["qc_status"] = "blocked_missing_local_reconstruction"
        updated_panels.append(out)

    text_by_case_format = {
        (r["dataset_label"], r["deidentified_case_id"], r["format_name"]): r["panel_text"]
        for r in updated_panels
    }
    status_by_case_format = {
        (r["dataset_label"], r["deidentified_case_id"], r["format_name"]): (r["panel_status"], r["qc_status"])
        for r in updated_panels
    }

    updated_blinded: list[dict[str, Any]] = []
    updated_key: list[dict[str, Any]] = []
    key_lookup = {(r["dataset_label"], r["deidentified_case_id"], r["display_format_id"]): r["true_format"] for r in base_key}
    for case in base_blinded:
        label = case["dataset_label"]
        deid = case["deidentified_case_id"]
        out = dict(case)
        for i in (1, 2, 3):
            fid = f"F{i}"
            fmt = key_lookup.get((label, deid, fid), "")
            if fmt:
                out[f"format_{i}_text"] = text_by_case_format.get((label, deid, fmt), out.get(f"format_{i}_text", ""))
                status, qc = status_by_case_format.get((label, deid, fmt), ("missing", "missing"))
                updated_key.append({
                    "dataset_label": label,
                    "deidentified_case_id": deid,
                    "display_format_id": fid,
                    "true_format": fmt,
                    "panel_status": status,
                    "qc_status": qc,
                })
        updated_blinded.append(out)

    summary: list[dict[str, Any]] = []
    for label in DATASET_TO_NPZ:
        part = [p for p in updated_panels if p["dataset_label"] == label]
        cases = {p["deidentified_case_id"] for p in part}
        for fmt in ("PPtheta posterior audit record", "EBM additive terms", "TreeSHAP feature attribution"):
            fp = [p for p in part if p["format_name"] == fmt]
            summary.append({
                "dataset_label": label,
                "format_name": fmt,
                "cases": len(cases),
                "materialized_panels": sum(str(p["panel_status"]).startswith("materialized") for p in fp),
                "local_reconstruction_panels": sum(p["panel_status"] == "materialized_local_reconstruction" for p in fp),
                "qc_pass_panels": sum(p["qc_status"] == "pass" for p in fp),
                "remaining_blocked_or_partial": sum(p["qc_status"] != "pass" for p in fp),
            })

    write_csv(output_dir / "materialized_clinician_panels_with_comparators.csv", updated_panels)
    write_csv(output_dir / "clinician_review_cases_blinded_with_comparators.csv", updated_blinded)
    write_csv(output_dir / "clinician_review_key_private_with_comparators.csv", updated_key)
    write_csv(output_dir / "clinician_panel_qc_with_comparators_summary.csv", summary)
    write_csv(output_dir / "local_comparator_training_summary.csv", meta_rows)

    md = [
        "# Local Comparator Panels",
        "",
        "EBM and TreeSHAP comparator panels were reconstructed locally from saved measurement-policy feature matrices.",
        "These panels are intended for clinician/user review packets, not for benchmark metric claims.",
        "",
    ]
    for row in meta_rows:
        md.append(
            f"- {row['dataset_label']}: {row['status']}, {row.get('materialized_cases', 0)}/{row.get('requested_cases', 0)} cases, "
            f"train rows {row.get('train_rows', '')}, features {row.get('features', '')}."
        )
    (output_dir / "local_comparator_panels.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{p{0.17\columnwidth}p{0.23\columnwidth}rrrr}",
        r"\toprule",
        r"Dataset & Format & Cases & Mat. & QC & Rem. \\",
        r"\midrule",
    ]
    line_end = chr(92) * 2
    for row in summary:
        display_name = {
            "PPtheta posterior audit record": r"PP$\theta$ record",
            "EBM additive terms": "EBM terms",
            "TreeSHAP feature attribution": "TreeSHAP",
        }.get(row["format_name"], row["format_name"])
        tex_lines.append(
            f"{tex_escape(row['dataset_label'])} & {display_name} & {row['cases']} & "
            f"{row['materialized_panels']} & {row['qc_pass_panels']} & {row['remaining_blocked_or_partial']} {line_end}"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "ppost_clinician_comparator_panel_qc_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-train", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    run(args.output_dir, max_train=args.max_train, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
