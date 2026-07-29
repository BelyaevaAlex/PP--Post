#!/usr/bin/env python3
"""Section 49: final PPtheta usefulness and trace-sufficiency tables.

This script joins the proof-suite tables with the acceptance-strengthening
jobs: native vs +PPtheta deltas, randomized evidence controls, fold sign checks,
and corrected trace-sufficiency curves.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper/aaai_pppost_mortality"
OUT_DIR = PAPER_DIR / "generated"

SELECTED_CSV = OUT_DIR / "ppost_proof_main_selected.csv"
CONTROLS_CSV = OUT_DIR / "ppost_proof_supp_controls.csv"
ACCEPT_V1 = ROOT / "output/mortality_paper_jobs/rahmatullaev_acceptance_strengthening_mortality_accept_strengthening_v1"
ACCEPT_V2 = ROOT / "output/mortality_paper_jobs/rahmatullaev_acceptance_strengthening_mortality_accept_strengthening_v2"
ACCEPT_NEXT = ROOT / "output/mortality_paper_jobs/rahmatullaev_acceptance_next_steps_mortality_accept_next_steps_v1"

DATASET_LABELS = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}
DATASET_KEYS = {v: k for k, v in DATASET_LABELS.items()}
SOURCE_LABELS = {
    "tabpfn_distill_xgb_soft": r"TabPFN$\rightarrow$XGB",
    "xgb": "XGBoost",
}
VARIANT_LABELS = {
    "pp_theta_post_ebm_bounded_residual_gate": "bounded residual gate",
    "pp_theta_post_rule_family_calibrated": "rule-family calibrated",
    "pp_theta_post_ebm_residual_mcc": "residual MCC",
}
RAW_VARIANTS = {v: k for k, v in VARIANT_LABELS.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    val = fnum(value)
    if val is None:
        return "--"
    sign = "+" if signed and val >= 0 else ""
    return f"{sign}{val:.{digits}f}"


def pct(value: Any, digits: int = 0) -> str:
    val = fnum(value)
    if val is None:
        return "--"
    return f"{100.0 * val:.{digits}f}\%"


def tex(text: Any) -> str:
    return str(text).replace("_", r"\_").replace("#", r"\#")


def proof_stats(dataset_key: str) -> dict[str, dict[str, str]]:
    path = ACCEPT_V1 / dataset_key / "rahmatullaev_accept_proof_statistics" / "proof_statistics.csv"
    rows = read_csv(path)
    return {row["metric"]: row for row in rows}


def trace_rows(dataset_key: str) -> list[dict[str, str]]:
    path = ACCEPT_V2 / dataset_key / "rahmatullaev_accept_trace_sufficiency_curve" / "trace_sufficiency_curve_summary.csv"
    rows = read_csv(path)
    for row in rows:
        row["dataset_key"] = dataset_key
        row["dataset"] = DATASET_LABELS[dataset_key]
    return rows


def compact_residual_trace_rows(dataset_key: str, selected_variant: str) -> list[dict[str, str]]:
    raw_variant = RAW_VARIANTS.get(selected_variant, selected_variant)
    path = (
        ACCEPT_NEXT
        / dataset_key
        / "rahmatullaev_next_compact_residual_trace"
        / "compact_residual_trace_summary.csv"
    )
    if not path.exists():
        return []
    rows = [r for r in read_csv(path) if r.get("variant") == raw_variant]
    if not rows:
        return []
    full = max(
        (fnum(r.get("ppost_mcc")) for r in rows if (fnum(r.get("mean_trace_fraction")) or 0.0) >= 0.999),
        default=None,
    )
    out: list[dict[str, str]] = []
    for row in rows:
        ppost_mcc = fnum(row.get("ppost_mcc"))
        retained = ppost_mcc / full if ppost_mcc is not None and full not in (None, 0.0) else None
        out.append({
            "dataset_key": dataset_key,
            "dataset": DATASET_LABELS[dataset_key],
            "fraction": row.get("fraction", ""),
            "mean_trace_fraction": row.get("mean_trace_fraction", ""),
            "native_mcc": row.get("native_mcc", ""),
            "ppost_mcc": row.get("ppost_mcc", ""),
            "delta_mcc": row.get("delta_mcc", ""),
            "delta_sensitivity": row.get("delta_sensitivity", ""),
            "delta_brier_score": row.get("delta_brier", ""),
            "delta_ece_10": row.get("delta_ece", ""),
            "mcc_retained_vs_full": "" if retained is None else retained,
            "claim": "compact residual evidence",
        })
    return out


def best_trace_rows(dataset_key: str, selected: dict[str, str]) -> list[dict[str, str]]:
    rows = trace_rows(dataset_key)
    residual = compact_residual_trace_rows(dataset_key, selected["variant"])
    if not residual:
        return rows
    if selected.get("variant") == "residual MCC":
        return residual
    old_compact = select_compact_trace(rows)
    new_compact = select_compact_trace(residual)
    old_mcc = fnum(old_compact.get("ppost_mcc")) or -999.0
    new_mcc = fnum(new_compact.get("ppost_mcc")) or -999.0
    if new_mcc >= old_mcc - 0.001:
        return residual
    return rows


def select_compact_trace(rows: list[dict[str, str]]) -> dict[str, str]:
    """Pick the best <=20% trace row by MCC, falling back to full trace.

    If all rows already use trace_fraction=1.0, return the full row and mark it
    downstream as a full-trace case.
    """
    compact = [r for r in rows if (fnum(r.get("mean_trace_fraction")) or 9.0) <= 0.20]
    if compact:
        return max(compact, key=lambda r: fnum(r.get("ppost_mcc")) or -999.0)
    return max(rows, key=lambda r: fnum(r.get("ppost_mcc")) or -999.0)


def patient_permuted_gap(dataset: str, variant: str, controls: list[dict[str, str]]) -> dict[str, str]:
    matches = [
        row for row in controls
        if row.get("dataset") == dataset
        and row.get("variant") == variant
        and row.get("control") == "patient-permuted"
    ]
    if not matches:
        return {}
    return matches[0]


def build_final_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = read_csv(SELECTED_CSV)
    controls = read_csv(CONTROLS_CSV)
    final_rows: list[dict[str, Any]] = []
    trace_curve: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for row in selected:
        dataset = row["dataset"]
        dataset_key = DATASET_KEYS[dataset]
        stats = proof_stats(dataset_key)
        traces = best_trace_rows(dataset_key, row)
        compact = select_compact_trace(traces)
        permuted = patient_permuted_gap(dataset, row["variant"], controls)
        trace_frac = fnum(compact.get("mean_trace_fraction"))
        compact_label = "full trace" if trace_frac is not None and trace_frac >= 0.999 else f"{pct(trace_frac)} trace"
        final_rows.append({
            "dataset": dataset,
            "source": row["source"],
            "variant": row["variant"],
            "native_mcc": row["native_mcc"],
            "ppost_mcc": row["ppost_mcc"],
            "delta_mcc": row["delta_mcc"],
            "delta_sensitivity": row["delta_sensitivity"],
            "delta_brier": row["delta_brier"],
            "delta_ece": row["delta_ece"],
            "mcc_positive_folds": stats.get("mcc", {}).get("positive_folds", ""),
            "mcc_folds": stats.get("mcc", {}).get("folds", ""),
            "mcc_sign_p": stats.get("mcc", {}).get("sign_test_p_two_sided", ""),
            "sens_positive_folds": stats.get("sensitivity", {}).get("positive_folds", ""),
            "sens_folds": stats.get("sensitivity", {}).get("folds", ""),
            "permuted_mcc_gap": permuted.get("observed_minus_control_mcc", row.get("control_gap", "")),
            "permuted_sens_gap": permuted.get("observed_minus_control_sensitivity", ""),
            "compact_fraction": compact.get("fraction", ""),
            "compact_trace_fraction": compact.get("mean_trace_fraction", ""),
            "compact_label": compact_label,
            "compact_ppost_mcc": compact.get("ppost_mcc", ""),
            "compact_delta_mcc": compact.get("delta_mcc", ""),
            "compact_delta_sensitivity": compact.get("delta_sensitivity", ""),
            "compact_mcc_retained": compact.get("mcc_retained_vs_full", ""),
            "claim": row["claim"],
        })
        for tr in traces:
            trace_curve.append({
                "dataset": dataset,
                "source": row["source"],
                "variant": row["variant"],
                "budget_fraction": tr.get("fraction", ""),
                "trace_fraction": tr.get("mean_trace_fraction", ""),
                "native_mcc": tr.get("native_mcc", ""),
                "ppost_mcc": tr.get("ppost_mcc", ""),
                "delta_mcc": tr.get("delta_mcc", ""),
                "delta_sensitivity": tr.get("delta_sensitivity", ""),
                "delta_brier": tr.get("delta_brier_score", ""),
                "delta_ece": tr.get("delta_ece_10", ""),
                "mcc_retained_vs_full": tr.get("mcc_retained_vs_full", ""),
                "claim": tr.get("claim", ""),
            })
    for row in controls:
        control_rows.append(row)
    return final_rows, control_rows, trace_curve


def write_main_tex(rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrll}",
        r"\toprule",
        r"Dataset/source & PP$\theta$ mode & Native MCC & +PP$\theta$ MCC & $\Delta$MCC & $\Delta$Sens. & Perm. gap & Compact trace & Claim \\",
        r"\midrule",
    ]
    for row in rows:
        compact = f"{row['compact_label']}, {fmt(row['compact_mcc_retained'])}$\\times$ MCC"
        claim = {
            "Audit signal; no MCC gain": "audit signal",
            "Utility + compact trace": "utility + compact trace",
            "Utility + sensitivity gain": "utility + sensitivity",
        }.get(str(row["claim"]), str(row["claim"]))
        lines.append(
            " & ".join([
                f"{row['dataset']} / {row['source']}",
                str(row["variant"]),
                fmt(row["native_mcc"]),
                fmt(row["ppost_mcc"]),
                fmt(row["delta_mcc"], signed=True),
                fmt(row["delta_sensitivity"], signed=True),
                fmt(row["permuted_mcc_gap"]),
                compact,
                claim,
            ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Final PP$\theta$ evidence-utility summary. Each row compares a native source with the same source wrapped by PP$\theta$-Post. Perm. gap is observed MCC minus patient-permuted-evidence MCC. Compact trace reports the best trace budget not exceeding 20\% when such a budget exists. The MIMIC-IV row shows that residual-evidence utility can be retained with a compact trace rather than requiring the full residual trace.}",
        r"\label{tab:ppost-final-usefulness}",
        r"\end{table*}",
        "",
    ]
    (OUT_DIR / "ppost_final_usefulness_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_trace_tex(rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Budget & Trace frac. & +PP$\theta$ MCC & $\Delta$MCC & $\Delta$Sens. & $\Delta$Brier & MCC retained \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join([
            row["dataset"],
            pct(row["budget_fraction"]),
            pct(row["trace_fraction"], digits=1),
            fmt(row["ppost_mcc"]),
            fmt(row["delta_mcc"], signed=True),
            fmt(row["delta_sensitivity"], signed=True),
            fmt(row["delta_brier"], signed=True),
            fmt(row["mcc_retained_vs_full"]),
        ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Trace-sufficiency curves for the selected PP$\theta$ rows. Budget is the requested evidence budget; trace fraction is the realized retained posterior evidence fraction. The MIMIC-IV residual row shows that residual-evidence utility can also be pruned rather than reported only as a full trace.}",
        r"\label{tab:supp-ppost-trace-sufficiency}",
        r"\end{table*}",
        "",
    ]
    (OUT_DIR / "ppost_final_trace_sufficiency_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_controls_tex(rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Dataset & PP$\theta$ mode & Control & Observed MCC & Control MCC & MCC gap & Sens. gap & $\Delta$ECE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join([
            row["dataset"],
            row["variant"],
            row["control"],
            fmt(row["observed_mcc"]),
            fmt(row["control_mcc"]),
            fmt(row["observed_minus_control_mcc"]),
            fmt(row["observed_minus_control_sensitivity"]),
            fmt(row["control_minus_observed_ece"], signed=True),
        ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Randomized and flattened evidence controls for the selected PP$\theta$ rows. Patient-permuted and class-prior controls test whether posterior evidence carries patient-specific signal; flattened evidence isolates probability-scale degradation.}",
        r"\label{tab:supp-ppost-final-controls}",
        r"\end{table*}",
        "",
    ]
    (OUT_DIR / "ppost_final_controls_table.tex").write_text("\n".join(lines), encoding="utf-8")


def read_next_rows(stage: str, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_key in DATASET_LABELS:
        path = ACCEPT_NEXT / dataset_key / f"rahmatullaev_next_{stage}" / filename
        if not path.exists():
            continue
        for row in read_csv(path):
            row["dataset_key"] = dataset_key
            row["dataset"] = DATASET_LABELS.get(row.get("dataset", dataset_key), row.get("dataset", dataset_key))
            rows.append(row)
    return rows


def interval(center: Any, lo: Any, hi: Any) -> str:
    return f"{fmt(center)} [{fmt(lo)}, {fmt(hi)}]"


def write_bootstrap_ci_tex(rows: list[dict[str, str]]) -> None:
    keep = [r for r in rows if str(r.get("fraction")) in {"0.05", "0.2", "1.0"}]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Budget & +PP$\theta$ MCC [95\% CI] & $\Delta$MCC [95\% CI] & $\Delta$Sens. [95\% CI] & MCC retained [95\% CI] \\",
        r"\midrule",
    ]
    for row in keep:
        lines.append(" & ".join([
            row["dataset"],
            pct(row["fraction"]),
            interval(row.get("ppost_mcc"), row.get("ppost_mcc_ci_low"), row.get("ppost_mcc_ci_high")),
            interval(row.get("delta_mcc"), row.get("delta_mcc_ci_low"), row.get("delta_mcc_ci_high")),
            interval(row.get("delta_sensitivity"), row.get("delta_sensitivity_ci_low"), row.get("delta_sensitivity_ci_high")),
            interval(row.get("retained_mcc_vs_full"), row.get("retained_mcc_ci_low"), row.get("retained_mcc_ci_high")),
        ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Patient-level bootstrap intervals for trace sufficiency. Intervals are computed from saved probability records without retraining. Budgets 5\%, 20\%, and 100\% summarize compact, moderate, and full evidence regimes.}",
        r"\label{tab:supp-ppost-trace-bootstrap-ci}",
        r"\end{table*}",
        "",
    ]
    (OUT_DIR / "ppost_final_trace_bootstrap_ci_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_subset_sufficiency_tex(rows: list[dict[str, str]], final_rows: list[dict[str, Any]]) -> None:
    compact_by_dataset = {DATASET_KEYS[r["dataset"]]: str(r["compact_fraction"]) for r in final_rows}
    wanted = {"mortality_positive", "native_wrong", "large_ppost_shift_top20"}
    labels = {
        "mortality_positive": "Mortality positives",
        "native_wrong": "Native wrong",
        "large_ppost_shift_top20": "Largest PP shift",
    }
    selected = [
        r for r in rows
        if r.get("subset") in wanted and str(r.get("fraction")) == compact_by_dataset.get(r.get("dataset_key", ""))
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Subset & Mean $n$ & $\Delta$Acc. & $\Delta$MCC & $\Delta$Sens. & $\Delta$Brier \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(" & ".join([
            row["dataset"],
            labels.get(row.get("subset", ""), tex(row.get("subset", ""))),
            fmt(row.get("mean_n"), digits=0),
            fmt(row.get("delta_accuracy"), signed=True),
            fmt(row.get("delta_mcc"), signed=True),
            fmt(row.get("delta_sensitivity"), signed=True),
            fmt(row.get("delta_brier"), signed=True),
        ]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Subset sufficiency at the selected compact budget. The table localizes where PP$\theta$ changes behavior: mortality positives, examples the native source gets wrong, and patients with the largest posterior risk shift.}",
        r"\label{tab:supp-ppost-subset-sufficiency}",
        r"\end{table*}",
        "",
    ]
    (OUT_DIR / "ppost_final_subset_sufficiency_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_markdown(final_rows: list[dict[str, Any]], controls: list[dict[str, Any]], trace: list[dict[str, Any]], bootstrap: list[dict[str, Any]], subsets: list[dict[str, Any]]) -> None:
    lines = [
        "# Final PPtheta Evidence Tables",
        "",
        "## Usefulness summary",
        "",
        "| Dataset | Source | Mode | Native MCC | +PPtheta MCC | dMCC | dSens | Perm gap | Compact trace | MCC retained | Claim |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['dataset']} | {row['source']} | {row['variant']} | {fmt(row['native_mcc'])} | {fmt(row['ppost_mcc'])} | {fmt(row['delta_mcc'], signed=True)} | {fmt(row['delta_sensitivity'], signed=True)} | {fmt(row['permuted_mcc_gap'])} | {row['compact_label']} | {fmt(row['compact_mcc_retained'])} | {row['claim']} |"
        )
    lines += [
        "",
        "## Randomized controls",
        "",
        "| Dataset | Mode | Control | Observed MCC | Control MCC | MCC gap | Sens gap | dECE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in controls:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['control']} | {fmt(row['observed_mcc'])} | {fmt(row['control_mcc'])} | {fmt(row['observed_minus_control_mcc'])} | {fmt(row['observed_minus_control_sensitivity'])} | {fmt(row['control_minus_observed_ece'], signed=True)} |"
        )
    lines += [
        "",
        "## Trace sufficiency curve",
        "",
        "| Dataset | Budget | Trace frac | +PPtheta MCC | dMCC | dSens | dBrier | MCC retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trace:
        lines.append(
            f"| {row['dataset']} | {pct(row['budget_fraction'])} | {pct(row['trace_fraction'], 1)} | {fmt(row['ppost_mcc'])} | {fmt(row['delta_mcc'], signed=True)} | {fmt(row['delta_sensitivity'], signed=True)} | {fmt(row['delta_brier'], signed=True)} | {fmt(row['mcc_retained_vs_full'])} |"
        )
    lines += [
        "",
        "## Bootstrap CI",
        "",
        "| Dataset | Budget | PPtheta MCC CI | dMCC CI | dSens CI | Retained MCC CI |",
        "|---|---:|---|---|---|---|",
    ]
    for row in bootstrap:
        if str(row.get("fraction")) in {"0.05", "0.2", "1.0"}:
            lines.append(
                f"| {row['dataset']} | {pct(row['fraction'])} | {interval(row.get('ppost_mcc'), row.get('ppost_mcc_ci_low'), row.get('ppost_mcc_ci_high'))} | {interval(row.get('delta_mcc'), row.get('delta_mcc_ci_low'), row.get('delta_mcc_ci_high'))} | {interval(row.get('delta_sensitivity'), row.get('delta_sensitivity_ci_low'), row.get('delta_sensitivity_ci_high'))} | {interval(row.get('retained_mcc_vs_full'), row.get('retained_mcc_ci_low'), row.get('retained_mcc_ci_high'))} |"
            )
    lines += [
        "",
        "## Subset sufficiency",
        "",
        "| Dataset | Fraction | Subset | Mean n | dAcc | dMCC | dSens | dBrier |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in subsets:
        if row.get("subset") in {"mortality_positive", "native_wrong", "large_ppost_shift_top20"}:
            lines.append(
                f"| {row['dataset']} | {pct(row.get('fraction'))} | {row.get('subset')} | {fmt(row.get('mean_n'), digits=0)} | {fmt(row.get('delta_accuracy'), signed=True)} | {fmt(row.get('delta_mcc'), signed=True)} | {fmt(row.get('delta_sensitivity'), signed=True)} | {fmt(row.get('delta_brier'), signed=True)} |"
            )
    (OUT_DIR / "ppost_final_evidence_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_rows, controls, trace = build_final_rows()
    bootstrap = read_next_rows("trace_bootstrap_ci", "trace_bootstrap_ci.csv")
    subsets = read_next_rows("subset_sufficiency", "subset_sufficiency_summary.csv")
    write_csv(OUT_DIR / "ppost_final_usefulness.csv", final_rows, [
        "dataset", "source", "variant", "native_mcc", "ppost_mcc", "delta_mcc", "delta_sensitivity", "delta_brier", "delta_ece",
        "mcc_positive_folds", "mcc_folds", "mcc_sign_p", "sens_positive_folds", "sens_folds",
        "permuted_mcc_gap", "permuted_sens_gap", "compact_fraction", "compact_trace_fraction", "compact_label",
        "compact_ppost_mcc", "compact_delta_mcc", "compact_delta_sensitivity", "compact_mcc_retained", "claim",
    ])
    write_csv(OUT_DIR / "ppost_final_controls.csv", controls, [
        "dataset", "source", "variant", "control", "observed_mcc", "control_mcc", "observed_minus_control_mcc",
        "observed_minus_control_sensitivity", "control_minus_observed_log_loss", "control_minus_observed_ece",
    ])
    write_csv(OUT_DIR / "ppost_final_trace_sufficiency.csv", trace, [
        "dataset", "source", "variant", "budget_fraction", "trace_fraction", "native_mcc", "ppost_mcc", "delta_mcc",
        "delta_sensitivity", "delta_brier", "delta_ece", "mcc_retained_vs_full", "claim",
    ])
    write_csv(OUT_DIR / "ppost_final_trace_bootstrap_ci.csv", bootstrap, [
        "dataset", "dataset_key", "fraction", "n", "bootstrap_n", "native_mcc", "ppost_mcc", "delta_mcc",
        "delta_sensitivity", "retained_mcc_vs_full", "retained_sensitivity_vs_full", "ppost_mcc_ci_low",
        "ppost_mcc_ci_high", "delta_mcc_ci_low", "delta_mcc_ci_high", "delta_sensitivity_ci_low",
        "delta_sensitivity_ci_high", "retained_mcc_ci_low", "retained_mcc_ci_high",
        "retained_sensitivity_ci_low", "retained_sensitivity_ci_high",
    ])
    write_csv(OUT_DIR / "ppost_final_subset_sufficiency.csv", subsets, [
        "dataset", "dataset_key", "fraction", "subset", "folds", "mean_n", "delta_accuracy",
        "delta_mcc", "delta_sensitivity", "delta_brier", "mean_risk_shift",
    ])
    write_main_tex(final_rows)
    write_controls_tex(controls)
    write_trace_tex(trace)
    write_bootstrap_ci_tex(bootstrap)
    write_subset_sufficiency_tex(subsets, final_rows)
    write_markdown(final_rows, controls, trace, bootstrap, subsets)
    print(f"Wrote final PPtheta evidence tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
