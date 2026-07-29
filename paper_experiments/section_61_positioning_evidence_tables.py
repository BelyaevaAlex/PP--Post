#!/usr/bin/env python3
"""Section 61: paper-ready positioning evidence tables.

This aggregator collects the three reviewer-facing evidence blocks that support
PPtheta-Post as a prediction-time posterior evidence object:

1. AuditSelect final deployment table.
2. Native-error correction slice.
3. Trace perturbation and compact sufficiency summaries.

It prefers fresh local/cluster positioning runs when present, and falls back to
previous completed AAAI evidence runs so table generation is never blocked by a
long queue.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
DATASETS = ("eicu", "mimic3", "mimic4")
DATASET_LABEL = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}

EVIDENCE_ROOT_CANDIDATES = [
    os.environ.get("PPPOST_POSITIONING_EVIDENCE_ROOT", ""),
    str(ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_positioning_local_v1"),
    str(ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2_positioning"),
    str(ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2"),
]
ACCEPT_ROOT_CANDIDATES = [
    os.environ.get("PPPOST_POSITIONING_ACCEPT_ROOT", ""),
    str(ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_acceptance_clinician_symbolic_mortality_positioning_local_v1"),
    str(ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_acceptance_clinician_symbolic_mortality_accept_positioning"),
    str(ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_acceptance_clinician_symbolic_mortality_accept_clinician_symbolic_v1"),
]

SOURCE_DISPLAY = {
    "rulefit": "RuleFit",
    "figs": "FIGS",
    "xgb": "XGBoost",
    "extratrees": "ExtraTrees",
    "tabpfn_distill_xgb_soft": "TabPFN-to-XGBoost",
    "tabpfn_distill_ebm_terms": "TabPFN-to-EBM terms",
    "ebm_terms": "EBM terms",
}
VARIANT_DISPLAY = {
    "pp_theta_post_ebm_residual_mcc": "bounded residual evidence",
    "pp_theta_post_ebm_bounded_residual_gate": "bounded residual gate",
    "pp_theta_post_rule_family_calibrated": "rule-family calibrated",
    "pp_theta_post_family_utility_pruned_topk": "utility-pruned top-k",
    "pp_theta_post_bayes_llr_posneg": "Bayesian LLR pos/neg",
}
CONTROL_DISPLAY = {
    "observed": "Observed trace",
    "patient_permuted": "Patient-permuted trace",
    "class_prior_only": "Class-prior trace",
    "temperature_flattened_t4": "Flattened trace T=4",
    "column_shuffled_class_scores": "Column-shuffled scores",
    "overconfident_same_rank_t0p5": "Sharpened same-rank trace",
}
SLICE_DISPLAY = {
    "native_wrong": "Native errors",
    "large_ppost_shift_top20": "Largest PPtheta shifts",
    "native_uncertain_top20": "Native uncertainty top 20%",
}


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def f(value: Any, digits: int = 3) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{x:.{digits}f}"


def d(value: Any, digits: int = 3) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{x:+.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{100*x:.{digits}f}\\%"


def tex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("TabPFN-to-XGBoost", "TabPFN-to-XGBoost")
    text = text.replace("<=", "@@LE@@").replace(">=", "@@GE@@")
    text = (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )
    text = text.replace("@@LE@@", "$\\le$").replace("@@GE@@", "$\\ge$")
    return text.replace("PPtheta", "PP$\\theta$").replace("PPθ", "PP$\\theta$")


def display_source(value: str) -> str:
    return SOURCE_DISPLAY.get(value, value.replace("_", " ").title())


def display_variant(value: str) -> str:
    return VARIANT_DISPLAY.get(value, value.replace("pp_theta_post_", "").replace("_", " "))


def root_with_file(candidates: list[str], dataset: str, stage: str, filename: str) -> Path | None:
    for raw in candidates:
        if not raw:
            continue
        root = Path(raw)
        path = root / dataset / stage / filename
        if path.exists() and path.stat().st_size > 0:
            return root
    return None


def rows_from(candidates: list[str], dataset: str, stage: str, filename: str) -> list[dict[str, str]]:
    root = root_with_file(candidates, dataset, stage, filename)
    if root is None:
        return []
    rows = read_csv(root / dataset / stage / filename)
    for row in rows:
        row.setdefault("artifact_root", str(root))
    return rows


def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]], align: str | None = None, wide: bool = True) -> None:
    br = r" \\"
    lines = ["\\begin{table*}[t]" if wide else "\\begin{table}[t]", "\\centering", "\\small"]
    if align is None:
        align = "l" * len(headers)
    lines.extend([f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(headers) + br, "\\midrule"])
    for row in rows:
        lines.append(" & ".join(row) + br)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    lines.extend([f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}" if wide else "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_auditselect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_candidates: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidates: list[dict[str, Any]] = []
        for stage, filename, policy in (
            ("rahmatullaev_accept_clean_interpretable_calibrated", "clean_interpretable_calibrated_selected.csv", "AuditSelect-Cal"),
            ("rahmatullaev_accept_symbolic_family_calibrated", "symbolic_family_calibrated_selected.csv", "AuditSelect-Symbolic"),
        ):
            for row in rows_from(ACCEPT_ROOT_CANDIDATES, dataset, stage, filename):
                item: dict[str, Any] = dict(row)
                item["dataset"] = dataset
                item["dataset_label"] = DATASET_LABEL[dataset]
                item["policy"] = policy
                item["stage"] = stage
                item["score"] = num(item.get("delta_mcc"), -999.0) + 0.25 * num(item.get("delta_sensitivity"), 0.0) - 0.5 * max(num(item.get("delta_brier_score"), 0.0), 0.0)
                candidates.append(item)
                all_candidates.append(item)
        if not candidates:
            selected_rows.append({"dataset": dataset, "dataset_label": DATASET_LABEL[dataset], "decision": "missing outputs"})
            continue
        strict = [r for r in candidates if str(r.get("passes_strict_calibration_constraint", "")).lower() == "true"]
        relaxed = [r for r in candidates if str(r.get("passes_relaxed_calibration_constraint", "")).lower() == "true"]
        pool = strict or relaxed or candidates
        best = max(pool, key=lambda r: num(r.get("score"), -999.0))
        stage = str(best.get("stage", ""))
        controls_name = "clean_interpretable_calibrated_controls_summary.csv" if "clean" in stage else "symbolic_family_calibrated_controls_summary.csv"
        perm_gap = float("nan")
        for ctrl in rows_from(ACCEPT_ROOT_CANDIDATES, dataset, stage, controls_name):
            if "permuted" in str(ctrl.get("control", "")).lower():
                perm_gap = num(ctrl.get("mcc_gap"))
                break
        deploy = str(best.get("passes_relaxed_calibration_constraint", "")).lower() == "true"
        selected_rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABEL[dataset],
            "policy": best.get("policy", ""),
            "selected_source": best.get("rule_source", ""),
            "ppost_mode": best.get("variant", ""),
            "native_mcc": num(best.get("native_mcc")),
            "ppost_mcc": num(best.get("ppost_mcc")),
            "delta_mcc": num(best.get("delta_mcc")),
            "delta_sensitivity": num(best.get("delta_sensitivity")),
            "delta_brier": num(best.get("delta_brier_score")),
            "delta_ece": num(best.get("delta_ece_10")),
            "permuted_trace_gap": perm_gap,
            "compact_trace_fraction": num(best.get("trace_fraction")),
            "decision": "Deploy PPtheta" if deploy else "Fallback native",
            "artifact_root": best.get("artifact_root", ""),
        })
    return selected_rows, all_candidates


def native_error_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows = rows_from(EVIDENCE_ROOT_CANDIDATES, dataset, "rahmatullaev_v2_native_wrong_correction", "native_wrong_correction_summary.csv")
        by_subset = {r.get("subset", ""): r for r in rows}
        corr_rate = num(by_subset.get("native_wrong", {}).get("correction_rate"))
        pos_corr = num(by_subset.get("native_wrong", {}).get("positive_correction_rate"))
        if not math.isfinite(corr_rate):
            wrong = num(by_subset.get("native_wrong", {}).get("mean_n"))
            fixed = num(by_subset.get("native_wrong_ppost_right", {}).get("mean_n"))
            corr_rate = fixed / wrong if wrong > 0 else float("nan")
        for subset in ("native_wrong", "large_ppost_shift_top20", "native_uncertain_top20"):
            row = by_subset.get(subset)
            if not row:
                continue
            out.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "slice": subset,
                "coverage": num(row.get("coverage")),
                "native_mcc": num(row.get("native_mcc")),
                "ppost_mcc": num(row.get("ppost_mcc")),
                "delta_mcc": num(row.get("delta_mcc")),
                "native_sensitivity": num(row.get("native_sensitivity")),
                "ppost_sensitivity": num(row.get("ppost_sensitivity")),
                "delta_sensitivity": num(row.get("delta_sensitivity")),
                "correction_rate": corr_rate if subset == "native_wrong" else float("nan"),
                "positive_correction_rate": pos_corr if subset == "native_wrong" else float("nan"),
                "mean_risk_shift": num(row.get("mean_risk_shift")),
                "mean_n": num(row.get("mean_n")),
                "artifact_root": row.get("artifact_root", ""),
            })
    return out


def trace_perturbation_rows() -> list[dict[str, Any]]:
    keep = {"observed", "patient_permuted", "class_prior_only", "temperature_flattened_t4", "column_shuffled_class_scores"}
    out: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows = rows_from(EVIDENCE_ROOT_CANDIDATES, dataset, "rahmatullaev_v2_rich_randomized_controls", "rich_randomized_controls_summary.csv")
        for row in rows:
            control = row.get("control", "")
            if control not in keep:
                continue
            out.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "trace_condition": control,
                "mcc": num(row.get("mcc")),
                "delta_vs_observed_mcc": num(row.get("delta_vs_observed_mcc")),
                "sensitivity": num(row.get("sensitivity")),
                "delta_brier": num(row.get("delta_vs_observed_brier_score")),
                "delta_ece": num(row.get("delta_vs_observed_ece_10")),
                "artifact_root": row.get("artifact_root", ""),
            })
    order = {k: i for i, k in enumerate(["observed", "patient_permuted", "class_prior_only", "temperature_flattened_t4", "column_shuffled_class_scores"])}
    return sorted(out, key=lambda r: (r["dataset"], order.get(str(r["trace_condition"]), 99)))


def compact_sufficiency_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows = rows_from(EVIDENCE_ROOT_CANDIDATES, dataset, "rahmatullaev_v2_extended_trace_curve", "extended_trace_curve_summary.csv")
        if not rows:
            continue
        full_candidates = [r for r in rows if abs(num(r.get("requested_fraction")) - 1.0) < 1e-9]
        full_mcc = num(full_candidates[0].get("ppost_mcc")) if full_candidates else max(num(r.get("ppost_mcc"), -1.0) for r in rows)
        normalized: list[dict[str, Any]] = []
        for row in rows:
            ppost_mcc = num(row.get("ppost_mcc"))
            retained = ppost_mcc / full_mcc if full_mcc > 0 and math.isfinite(ppost_mcc) else float("nan")
            item = {
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "source": row.get("rule_source", row.get("source", "")),
                "variant": row.get("variant", ""),
                "requested_fraction": num(row.get("requested_fraction")),
                "trace_fraction": num(row.get("trace_fraction")),
                "ppost_mcc": ppost_mcc,
                "mcc_retained_vs_full": retained,
                "delta_mcc": num(row.get("delta_mcc")),
                "delta_sensitivity": num(row.get("delta_sensitivity")),
                "delta_brier": num(row.get("delta_brier_score")),
                "artifact_root": row.get("artifact_root", ""),
            }
            normalized.append(item)
        good = [r for r in normalized if num(r.get("mcc_retained_vs_full")) >= 0.99]
        selected = min(good, key=lambda r: (num(r.get("trace_fraction"), 999), num(r.get("requested_fraction"), 999))) if good else max(normalized, key=lambda r: num(r.get("mcc_retained_vs_full"), -999))
        selected["claim"] = ">=99% full MCC" if num(selected.get("mcc_retained_vs_full")) >= 0.99 else "best available compact trace"
        out.append(selected)
    return out


def write_outputs() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    audit, audit_candidates = selected_auditselect_rows()
    native = native_error_rows()
    perturb = trace_perturbation_rows()
    compact = compact_sufficiency_rows()

    write_csv(GENERATED / "ppost_auditselect_deployment.csv", audit)
    write_csv(GENERATED / "ppost_auditselect_deployment_candidates.csv", audit_candidates)
    write_csv(GENERATED / "ppost_native_error_slice.csv", native)
    write_csv(GENERATED / "ppost_trace_perturbation.csv", perturb)
    write_csv(GENERATED / "ppost_compact_sufficiency_curve.csv", compact)

    write_tex_table(
        GENERATED / "ppost_auditselect_deployment_table.tex",
        "AuditSelect final deployment table. The candidate family is fixed to RuleFit/FIGS teacher-free PP$\\theta$ variants; deployment requires positive validation utility without calibration degradation, otherwise the native source is retained.",
        "tab:ppost-auditselect-deployment",
        ["Dataset", "Selected source", "PP$\\theta$ mode", "Native MCC", "PP$\\theta$ MCC", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier", "$\\Delta$ECE", "Perm. gap", "Trace frac.", "Decision"],
        [[tex_escape(r["dataset_label"]), tex_escape(display_source(str(r.get("selected_source", "")))), tex_escape(display_variant(str(r.get("ppost_mode", "")))), f(r.get("native_mcc")), f(r.get("ppost_mcc")), d(r.get("delta_mcc")), d(r.get("delta_sensitivity")), d(r.get("delta_brier")), d(r.get("delta_ece")), d(r.get("permuted_trace_gap")), f(r.get("compact_trace_fraction")), tex_escape(r.get("decision", ""))] for r in audit],
        "llllllllllll",
    )

    write_tex_table(
        GENERATED / "ppost_native_error_slice_table.tex",
        "Native-error correction slices. Correction rate is the fraction of native-source errors corrected by PP$\\theta$-Post; positive correction rate restricts the denominator to mortality-positive native errors.",
        "tab:ppost-native-error-slice",
        ["Dataset", "Slice", "Coverage", "Native MCC", "PP$\\theta$ MCC", "$\\Delta$MCC", "Native sens.", "PP$\\theta$ sens.", "$\\Delta$Sens.", "Corr.", "Pos. corr.", "Risk shift"],
        [[tex_escape(r["dataset_label"]), tex_escape(SLICE_DISPLAY.get(str(r.get("slice", "")), str(r.get("slice", "")))), pct(r.get("coverage")), f(r.get("native_mcc")), f(r.get("ppost_mcc")), d(r.get("delta_mcc")), f(r.get("native_sensitivity")), f(r.get("ppost_sensitivity")), d(r.get("delta_sensitivity")), pct(r.get("correction_rate")), pct(r.get("positive_correction_rate")), d(r.get("mean_risk_shift"))] for r in native],
        "llllllllllll",
    )

    write_tex_table(
        GENERATED / "ppost_trace_perturbation_table.tex",
        "Trace perturbation controls. Patient permutation, class-prior replacement, flattening, and class-score shuffling test whether the exported posterior trace carries patient-specific prediction signal.",
        "tab:ppost-trace-perturbation",
        ["Dataset", "Trace condition", "MCC", "$\\Delta$MCC", "Sens.", "$\\Delta$Brier", "$\\Delta$ECE"],
        [[tex_escape(r["dataset_label"]), tex_escape(CONTROL_DISPLAY.get(str(r.get("trace_condition", "")), str(r.get("trace_condition", "")))), f(r.get("mcc")), d(r.get("delta_vs_observed_mcc")), f(r.get("sensitivity")), d(r.get("delta_brier")), d(r.get("delta_ece"))] for r in perturb],
        "lllllll",
    )

    write_tex_table(
        GENERATED / "ppost_compact_sufficiency_curve_table.tex",
        "Compact trace sufficiency. For each dataset, the selected row is the smallest trace fraction that retains at least 99\\% of full-trace MCC when such a row exists.",
        "tab:ppost-compact-sufficiency",
        ["Dataset", "Source", "PP$\\theta$ mode", "Trace frac.", "MCC retained", "$\\Delta$MCC", "$\\Delta$Sens.", "Claim"],
        [[tex_escape(r["dataset_label"]), tex_escape(display_source(str(r.get("source", "")))), tex_escape(display_variant(str(r.get("variant", "")))), f(r.get("trace_fraction")), f(r.get("mcc_retained_vs_full")), d(r.get("delta_mcc")), d(r.get("delta_sensitivity")), tex_escape(r.get("claim", ""))] for r in compact],
        "llllllll",
    )

    print(f"Wrote positioning evidence tables to {GENERATED}")
    print(f"audit={len(audit)} native={len(native)} perturb={len(perturb)} compact={len(compact)}")


def main() -> int:
    write_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
