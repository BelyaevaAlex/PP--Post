#!/usr/bin/env python3
"""Aggregate Section 51 AAAI evidence-v2 outputs into paper-ready tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
DATASETS = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def f(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "--"
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def tex_delta(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "--"
    if not math.isfinite(x):
        return "--"
    return f"{x:+.3f}"



SOURCE_DISPLAY = {
    "rulefit": "RuleFit",
    "figs": "FIGS",
    "extratrees": "ExtraTrees",
    "xgb": "XGBoost",
    "catboost": "CatBoost",
    "ebm_terms": "EBM terms",
    "tabpfn_distill_ebm_terms": "TabPFN-distilled EBM terms",
    "tabpfn_distill_xgb_soft": "TabPFN-to-XGBoost",
}
VARIANT_DISPLAY = {
    "pp_theta_post_ebm_residual_mcc": "bounded residual evidence",
    "pp_theta_post_rule_family_calibrated": "rule-family calibrated evidence",
    "pp_theta_post_family_utility_pruned_topk": "utility-pruned top-k evidence",
    "pp_theta_post_bayes_llr_posneg": "Bayesian LLR evidence",
    "pp_theta_post_ebm_bounded_residual_gate": "bounded residual gate",
    "source_native": "native source",
}

def display_source(value: str) -> str:
    return SOURCE_DISPLAY.get(value, value.replace("_", " ").title())

def display_variant(value: str) -> str:
    return VARIANT_DISPLAY.get(value, value.replace("pp_theta_post_", "").replace("_", " "))

def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(headers)
    br = " " + "\\" * 2
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + br,
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + br)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)
    paired_rows: list[dict[str, Any]] = []
    controls_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for key, label in DATASETS.items():
        root = RUN_ROOT / key
        for stage_dir in sorted(root.glob("rahmatullaev_v2_*")):
            csvs = sorted(stage_dir.glob("*.csv"))
            manifest.append({"dataset": label, "stage": stage_dir.name, "csv_files": len(csvs), "done": int(bool(csvs))})
        paired = read_csv(root / "rahmatullaev_v2_paired_utility_ci/paired_utility_ci.csv")
        for row in paired:
            row["dataset_label"] = label; paired_rows.append(row)
        controls = read_csv(root / "rahmatullaev_v2_rich_randomized_controls/rich_randomized_controls_summary.csv")
        for row in controls:
            row["dataset_label"] = label; controls_rows.append(row)
        src = read_csv(root / "rahmatullaev_v2_source_compatibility_matrix/source_compatibility_summary.csv")
        for row in src:
            row["dataset_label"] = label; source_rows.append(row)
        corr = read_csv(root / "rahmatullaev_v2_native_wrong_correction/native_wrong_correction_summary.csv")
        for row in corr:
            row["dataset_label"] = label; correction_rows.append(row)

    write_csv(GENERATED / "aaai_evidence_v2_manifest.csv", manifest)
    write_csv(GENERATED / "aaai_evidence_v2_paired_ci.csv", paired_rows)
    write_csv(GENERATED / "aaai_evidence_v2_controls.csv", controls_rows)
    write_csv(GENERATED / "aaai_evidence_v2_source_compatibility.csv", source_rows)
    write_csv(GENERATED / "aaai_evidence_v2_native_wrong.csv", correction_rows)

    paired_tex = []
    for row in paired_rows:
        paired_tex.append([row.get("dataset_label", row.get("dataset", "")), tex_delta(row.get("delta_mcc")), f"[{f(row.get('delta_mcc_ci_low'))},{f(row.get('delta_mcc_ci_high'))}]", tex_delta(row.get("delta_sensitivity")), tex_delta(row.get("delta_brier_score")), f(row.get("fold_win_rate_mcc"))])
    write_tex_table(GENERATED / "aaai_evidence_v2_paired_ci_table.tex", "Paired native-source versus PP$\\theta$-Post utility intervals for the selected reviewer-facing rows.", "tab:aaai-evidence-v2-paired", ["Dataset", "$\\Delta$MCC", "95\\% CI", "$\\Delta$Sens.", "$\\Delta$Brier", "Fold win"], paired_tex)

    control_tex = []
    for row in controls_rows:
        if row.get("control") in {"observed", "patient_permuted", "class_prior_only", "temperature_flattened_t4"}:
            control_tex.append([row.get("dataset_label", ""), display_control(row.get("control", "")), f(row.get("mcc")), tex_delta(row.get("delta_vs_observed_mcc")), f(row.get("sensitivity")), tex_delta(row.get("delta_vs_observed_sensitivity"))])
    write_tex_table(GENERATED / "aaai_evidence_v2_controls_table.tex", "Counterfactual posterior-evidence controls. Negative deltas show degradation relative to the observed PP$\\theta$ trace.", "tab:aaai-evidence-v2-controls", ["Dataset", "Control", "MCC", "$\\Delta$MCC", "Sens.", "$\\Delta$Sens."], control_tex)

    best_source_tex = []
    for label in DATASETS.values():
        rows = [r for r in source_rows if r.get("dataset_label") == label]
        rows = sorted(rows, key=lambda r: float(r.get("delta_mcc", "nan") or "nan") if str(r.get("delta_mcc", "nan")) != "nan" else -999, reverse=True)[:6]
        for row in rows:
            best_source_tex.append([label, display_source(row.get("rule_source", "")), display_variant(row.get("variant", "")), tex_delta(row.get("delta_mcc")), tex_delta(row.get("delta_sensitivity")), tex_delta(row.get("delta_brier_score")), f(row.get("trace_fraction"))])
    write_tex_table(GENERATED / "aaai_evidence_v2_source_compatibility_table.tex", "Best source-compatibility rows for PP$\\theta$-Post as a posterior audit layer over native rule substrates.", "tab:aaai-evidence-v2-source", ["Dataset", "Source", "Variant", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier", "Trace frac."], best_source_tex)

    subset_tex = []
    for row in correction_rows:
        if row.get("subset") in {"native_wrong", "native_wrong_mortality_positive", "mortality_positive", "large_ppost_shift_top20"}:
            subset_tex.append([row.get("dataset_label", ""), display_subset(row.get("subset", "")), f(row.get("mean_n"), 1), tex_delta(row.get("delta_mcc")), tex_delta(row.get("delta_sensitivity")), tex_delta(row.get("delta_brier_score"))])
    write_tex_table(GENERATED / "aaai_evidence_v2_native_wrong_table.tex", "Subset utility of PP$\\theta$-Post on mortality positives, native errors, and high-shift cases.", "tab:aaai-evidence-v2-native-wrong", ["Dataset", "Subset", "Mean n", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier"], subset_tex)

    print(f"Wrote AAAI evidence-v2 tables to {GENERATED}")
    print(f"manifest_rows={len(manifest)} paired={len(paired_rows)} controls={len(controls_rows)} source={len(source_rows)} correction={len(correction_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
