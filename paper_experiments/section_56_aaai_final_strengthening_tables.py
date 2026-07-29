#!/usr/bin/env python3
"""Aggregate Section 55 final strengthening outputs into paper-ready tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_final_strengthening_mortality_aaai_final_strengthening_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
DATASETS = ("eicu", "mimic3", "mimic4")
STAGES = {
    "slim_usefulness": "rahmatullaev_final_slim_usefulness",
    "replay_integrity": "rahmatullaev_final_replay_integrity",
    "clinical_trace": "rahmatullaev_final_clinical_trace",
    "deletion_sufficiency": "rahmatullaev_final_deletion_sufficiency",
    "failure_modes": "rahmatullaev_final_failure_modes",
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

def display_source(value: str) -> str:
    return SOURCE_DISPLAY.get(value, value.replace("_", " ").title())

def display_variant(value: str) -> str:
    return VARIANT_DISPLAY.get(value, value.replace("pp_theta_post_", "").replace("_", " "))

def tex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("TabPFN$\\rightarrow$XGB", "TabPFN-to-XGB")
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
    return text.replace("@@LE@@", "$\\le$").replace("@@GE@@", "$\\ge$")


def num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def f(value: Any, digits: int = 3) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{x:.{digits}f}"


def d(value: Any) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{x:+.3f}"


def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]], align: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    br = " " + "\\" * 2
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{align or ('l' * len(headers))}}}",
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


def collect(stage_key: str, filename: str) -> list[dict[str, str]]:
    stage = STAGES[stage_key]
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        rows.extend(read_csv(RUN_ROOT / dataset / stage / filename))
    return rows


def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset in DATASETS:
        for stage in STAGES.values():
            csvs = sorted((RUN_ROOT / dataset / stage).glob("*.csv"))
            manifest.append({"dataset": dataset, "stage": stage, "csv_files": len(csvs), "done": int(bool(csvs))})
    write_csv(GENERATED / "ppost_final_strengthening_manifest.csv", manifest)

    slim = collect("slim_usefulness", "slim_usefulness.csv")
    replay = collect("replay_integrity", "replay_integrity.csv")
    deletion = collect("deletion_sufficiency", "deletion_sufficiency_summary.csv")
    clinical = collect("clinical_trace", "clinical_trace.csv")
    failures = collect("failure_modes", "failure_modes.csv")

    write_csv(GENERATED / "ppost_final_slim_usefulness.csv", slim)
    write_csv(GENERATED / "ppost_final_replay_integrity.csv", replay)
    write_csv(GENERATED / "ppost_final_deletion_sufficiency.csv", deletion)
    write_csv(GENERATED / "ppost_final_clinical_trace.csv", clinical)
    write_csv(GENERATED / "ppost_final_failure_modes.csv", failures)

    write_tex_table(
        GENERATED / "ppost_final_slim_usefulness_table.tex",
        "Reviewer-facing usefulness summary. PP$\\theta$-Post is evaluated within source, with non-random evidence controls and compact trace fraction reported beside utility metrics.",
        "tab:ppost-final-slim-usefulness",
        ["Dataset", "Source", "Variant", "$\\Delta$MCC", "$\\Delta$Sens.", "Permuted $\\Delta$MCC", "Trace frac.", "Claim"],
        [[tex_escape(r.get("dataset", "")), tex_escape(r.get("native_source", "")), tex_escape(r.get("ppost_variant", "")), d(r.get("delta_mcc")), d(r.get("delta_sensitivity")), d(r.get("patient_permuted_delta_mcc")), f(r.get("compact_trace_fraction")), tex_escape(r.get("claim", ""))] for r in slim],
        "llllllll",
    )

    write_tex_table(
        GENERATED / "ppost_final_replay_integrity_table.tex",
        "Replay-integrity checks for exported posterior traces. Decision match counts compare the saved prediction with the exported probability vector at the stored decision threshold.",
        "tab:ppost-final-replay-integrity",
        ["Dataset", "Samples", "Finite prob.", "Decision match", "Top branches", "Counterfactuals", "Max sum err.", "Status"],
        [[tex_escape(r.get("dataset", "")), r.get("samples", ""), r.get("finite_probability_vectors", ""), r.get("decision_matches_argmax", ""), r.get("complete_top_branch_records", ""), r.get("complete_counterfactual_records", ""), f(r.get("max_probability_sum_error"), 2), tex_escape(r.get("status", ""))] for r in replay],
    )

    del_rows = [r for r in deletion if r.get("k") in {"1", "5", "25", "100"}]
    write_tex_table(
        GENERATED / "ppost_final_deletion_sufficiency_table.tex",
        "Deletion and sufficiency summary over exported posterior traces. Top-$K$ sufficiency gap measures how close the top-$K$ trace is to the full trace; deletion shifts measure probability change after removing top evidence.",
        "tab:ppost-final-deletion-sufficiency",
        ["Dataset", "$K$", "Cases", "Suff. gap", "Delete shift", "Family shift"],
        [[tex_escape(r.get("dataset", "")), r.get("k", ""), r.get("cases", ""), f(r.get("mean_topk_sufficiency_gap")), d(r.get("mean_topk_deletion_shift")), d(r.get("mean_family_deletion_shift"))] for r in del_rows],
    )

    clinical_rows = []
    for r in clinical:
        if r.get("example") == "corrected mortality-positive case" or (r.get("example") == "supporting posterior rule" and r.get("rank") in {"1", "2"}):
            clinical_rows.append([
                tex_escape(r.get("dataset", "")),
                tex_escape(r.get("example", "")),
                tex_escape(r.get("case_id", "")),
                tex_escape(r.get("true_label", "")),
                f(r.get("ppost_mortality_probability")),
                tex_escape(r.get("evidence", "")),
            ])
    write_tex_table(
        GENERATED / "ppost_final_clinical_trace_table.tex",
        "Readable clinical trace examples. The first row per dataset shows a native-error correction candidate; supporting-rule rows show posterior evidence rendered with clinical feature names.",
        "tab:ppost-final-clinical-trace",
        ["Dataset", "Example", "Case", "True", "PP$\\theta$ prob.", "Evidence"],
        clinical_rows[:12],
        "lllllp{0.42\\linewidth}",
    )

    fail_rows = []
    for r in failures:
        if len([x for x in fail_rows if x[0] == tex_escape(r.get("dataset", ""))]) >= 2:
            continue
        fail_rows.append([
            tex_escape(r.get("dataset", "")),
            tex_escape(display_source(r.get("source", ""))),
            tex_escape(display_variant(r.get("variant", ""))),
            d(r.get("delta_mcc")),
            d(r.get("delta_sensitivity")),
            d(r.get("delta_brier")),
            tex_escape(r.get("interpretation", "")),
        ])
    write_tex_table(
        GENERATED / "ppost_final_failure_modes_table.tex",
        "Failure modes and negative boundaries. These rows define when PP$\\theta$-Post should not be used as the operating point over a given source.",
        "tab:ppost-final-failure-modes",
        ["Dataset", "Source", "Variant", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier", "Interpretation"],
        fail_rows,
    )

    print(f"Wrote final strengthening tables to {GENERATED}")
    print(f"manifest={len(manifest)} slim={len(slim)} replay={len(replay)} deletion={len(deletion)} clinical={len(clinical)} failures={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
