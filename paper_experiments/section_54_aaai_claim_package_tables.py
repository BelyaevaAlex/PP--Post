#!/usr/bin/env python3
"""Aggregate Section 53 claim-package outputs into paper-ready tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_claim_package_mortality_aaai_claim_package_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
DATASETS = ("eicu", "mimic3", "mimic4")
STAGES = (
    "rahmatullaev_claim_contract",
    "rahmatullaev_claim_source_boundary_map",
    "rahmatullaev_claim_control_gap_audit",
    "rahmatullaev_claim_trace_sufficiency_refresh",
    "rahmatullaev_claim_reviewer_trace_examples",
    "rahmatullaev_claim_package_summary",
)


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


def tex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def f(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "--"
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def d(value: Any) -> str:
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
}
ROLE_DISPLAY = {"best_mcc": "highest utility", "best_sensitivity": "highest sensitivity", "negative_boundary": "boundary case"}

def display_source(value: str) -> str:
    if "TabPFN" in value and "XGB" in value:
        return "TabPFN-to-XGBoost"
    return SOURCE_DISPLAY.get(value, value.replace("_", " ").title())

def display_variant(value: str) -> str:
    return VARIANT_DISPLAY.get(value, value.replace("pp_theta_post_", "").replace("_", " "))

def display_role(value: str) -> str:
    return ROLE_DISPLAY.get(value, value.replace("_", " "))

def clean_trace_detail(value: str) -> str:
    text = str(value if value is not None else "")
    text = re.sub(r"value\(([^)]+)\)", lambda m: "value threshold for " + m.group(1).replace("_", " "), text)
    text = re.sub(r"mask\(([^)]+)\)", lambda m: "measurement pattern for " + m.group(1).replace("_", " "), text)
    text = text.replace(" AND ", "; ")
    text = text.replace("_", " ")
    return text[:150]


def display_control(value: str) -> str:
    return {
        "observed": "observed",
        "patient_permuted": "patient permuted",
        "class_prior_only": "class prior",
        "column_shuffled_class_scores": "class-score shuffled",
        "temperature_flattened_t4": r"flattened $T=4$",
    }.get(value, value.replace("_", " "))


def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]], widths: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = widths or ("l" * len(headers))
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


def collect(stage: str, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        rows.extend(read_csv(RUN_ROOT / dataset / stage / filename))
    return rows


def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset in DATASETS:
        for stage in STAGES:
            path = RUN_ROOT / dataset / stage
            csvs = sorted(path.glob("*.csv")) if path.exists() else []
            manifest.append({"dataset": dataset, "stage": stage, "csv_files": len(csvs), "done": int(bool(csvs))})
    write_csv(GENERATED / "ppost_claim_package_manifest.csv", manifest)

    claim_rows = collect("rahmatullaev_claim_contract", "claim_contract.csv")
    source_rows = collect("rahmatullaev_claim_source_boundary_map", "source_boundary_map.csv")
    control_rows = collect("rahmatullaev_claim_control_gap_audit", "control_gap_audit.csv")
    trace_rows = collect("rahmatullaev_claim_trace_sufficiency_refresh", "trace_sufficiency_refresh.csv")
    example_rows = collect("rahmatullaev_claim_reviewer_trace_examples", "reviewer_trace_examples.csv")
    summary_rows = collect("rahmatullaev_claim_package_summary", "claim_package_summary.csv")

    write_csv(GENERATED / "ppost_claim_contract.csv", claim_rows)
    write_csv(GENERATED / "ppost_claim_source_boundary_map.csv", source_rows)
    write_csv(GENERATED / "ppost_claim_control_gap_audit.csv", control_rows)
    write_csv(GENERATED / "ppost_claim_trace_sufficiency_refresh.csv", trace_rows)
    write_csv(GENERATED / "ppost_claim_reviewer_trace_examples.csv", example_rows)
    write_csv(GENERATED / "ppost_claim_package_summary.csv", summary_rows)

    contract = [
        ["Prediction-sufficient posterior trace", "Trace replay and finite Lean core", "Saved branch activations/supports reconstruct the deployed score", "Does not verify real-analysis limits"],
        ["Non-random evidence", "Observed vs. patient-permuted/class-prior evidence", "Patient-permuted controls lose roughly 0.48--0.55 MCC", "A control gap alone is not an accuracy claim"],
        ["Source-specific utility", "Native source vs. same source plus PP$\\theta$-Post", "Positive rows exist, strongest for weaker symbolic substrates", "Strong substrates such as EBM can be negative boundaries"],
        ["Compact audit surface", "Full trace vs. top family trace", "Compact rows retain most full-trace MCC on selected datasets", "The faithful object is the ranked posterior trace"],
    ]
    write_tex_table(
        GENERATED / "ppost_claim_contract_table.tex",
        "Claim contract for PP$\\theta$-Post. Each claim is tied to a reproducible test and an explicit boundary.",
        "tab:ppost-claim-contract",
        ["Claim", "Test", "Evidence used in paper", "Boundary"],
        contract,
        "p{0.20\\linewidth}p{0.24\\linewidth}p{0.28\\linewidth}p{0.18\\linewidth}",
    )

    summary_tex = []
    for row in summary_rows:
        summary_tex.append(
            [
                tex_escape(row.get("dataset", "")),
                d(row.get("selected_delta_mcc")),
                d(row.get("selected_delta_sensitivity")),
                d(row.get("patient_permuted_delta_mcc")),
                tex_escape(row.get("best_source", "")),
                d(row.get("best_source_delta_mcc")),
                f(row.get("compact_trace_fraction")),
            ]
        )
    write_tex_table(
        GENERATED / "ppost_claim_package_summary_table.tex",
        "Dataset-level claim package: selected within-source utility, non-random evidence control gap, best source-compatibility row, and compact trace fraction.",
        "tab:ppost-claim-package-summary",
        ["Dataset", "$\\Delta$MCC", "$\\Delta$Sens.", "Permuted $\\Delta$MCC", "Best source", "Best $\\Delta$MCC", "Trace frac."],
        summary_tex,
    )

    boundary_tex = []
    for row in source_rows:
        if row.get("kind") not in {"best_mcc", "negative_boundary"} or row.get("rank") not in {"1", "2"}:
            continue
        boundary_tex.append(
            [
                tex_escape(row.get("dataset", "")),
                tex_escape(display_role(row.get("kind", ""))),
                tex_escape(display_source(row.get("source", ""))),
                tex_escape(display_variant(row.get("variant", ""))),
                d(row.get("delta_mcc")),
                d(row.get("delta_sensitivity")),
                d(row.get("delta_brier")),
            ]
        )
    write_tex_table(
        GENERATED / "ppost_claim_source_boundary_table.tex",
        "Source compatibility and negative boundaries. PP$\\theta$-Post is claimed only for sources and operating points where the evidence layer is useful.",
        "tab:ppost-claim-source-boundary",
        ["Dataset", "Role", "Source", "Variant", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier"],
        boundary_tex,
    )

    control_tex = []
    for row in control_rows:
        if row.get("control") not in {"observed", "patient_permuted", "class_prior_only", "column_shuffled_class_scores"}:
            continue
        control_tex.append(
            [
                tex_escape(row.get("dataset", "")),
                tex_escape(display_control(row.get("control", ""))),
                f(row.get("mcc")),
                d(row.get("delta_mcc")),
                f(row.get("sensitivity")),
                d(row.get("delta_sensitivity")),
                d(row.get("delta_log_loss")),
            ]
        )
    write_tex_table(
        GENERATED / "ppost_claim_control_gap_table.tex",
        "Non-random evidence controls used to separate posterior evidence from bookkeeping artifacts.",
        "tab:ppost-claim-control-gap",
        ["Dataset", "Control", "MCC", "$\\Delta$MCC", "Sens.", "$\\Delta$Sens.", "$\\Delta$Log loss"],
        control_tex,
    )

    trace_tex = []
    for row in trace_rows:
        if row.get("budget_fraction") not in {"0.01", "0.05", "0.1", "1.0"}:
            continue
        trace_tex.append(
            [
                tex_escape(row.get("dataset", "")),
                tex_escape(display_source(row.get("source", ""))),
                tex_escape(display_variant(row.get("variant", ""))),
                f(row.get("budget_fraction"), 2),
                f(row.get("trace_fraction")),
                f(row.get("mcc_retained_vs_full")),
                d(row.get("delta_sensitivity")),
            ]
        )
    write_tex_table(
        GENERATED / "ppost_claim_trace_summary_table.tex",
        "Compact-trace sufficiency summary for selected PP$\\theta$-Post rows.",
        "tab:ppost-claim-trace-summary",
        ["Dataset", "Source", "Variant", "Budget", "Trace frac.", "MCC retained", "$\\Delta$Sens."],
        trace_tex,
    )

    example_tex = []
    for row in example_rows:
        if row.get("example_type") == "tabular correction candidate":
            detail = f"shift {d(row.get('probability_shift'))}"
        else:
            detail = tex_escape(clean_trace_detail(row.get("trace_detail", "")))
        example_tex.append(
            [
                tex_escape(row.get("dataset", "")),
                tex_escape(row.get("example_type", "")),
                tex_escape(row.get("case_id", "")),
                tex_escape(row.get("true_label", "")),
                f(row.get("ppost_probability")),
                detail,
            ]
        )
    write_tex_table(
        GENERATED / "ppost_claim_trace_examples_table.tex",
        "Reviewer-facing patient trace examples. Tabular rows identify native errors corrected by PP$\\theta$-Post; replayable trace rows show exported rule evidence.",
        "tab:ppost-claim-trace-examples",
        ["Dataset", "Example", "Case", "True", "PP$\\theta$ prob.", "Trace detail"],
        example_tex[:12],
        "lllllp{0.36\\linewidth}",
    )

    print(f"Wrote claim-package tables to {GENERATED}")
    print(f"manifest={len(manifest)} claim={len(claim_rows)} source={len(source_rows)} controls={len(control_rows)} trace={len(trace_rows)} examples={len(example_rows)} summary={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
