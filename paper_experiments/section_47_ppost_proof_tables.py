#!/usr/bin/env python3
"""Build final PPtheta-Post proof tables for the AAAI paper.

The script reads the completed section_46 proof outputs and writes compact
LaTeX/CSV artifacts used by the main paper and supplement.
"""

from __future__ import annotations

import csv
import glob
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(
    "output/mortality_paper_jobs/"
    "rahmatullaev_ppost_proof_mortality_ppost_proof_local_v1"
)
PAPER_DIR = Path("paper/aaai_pppost_mortality")
OUT_DIR = PAPER_DIR / "generated"


DATASET_LABELS = {
    "eicu": "eICU",
    "mimic3": "MIMIC-III",
    "mimic4": "MIMIC-IV",
}

SOURCE_LABELS = {
    "xgb": "XGBoost",
    "tabpfn_distill_xgb_soft": r"TabPFN$\rightarrow$XGB",
    "ebm_terms": "EBM terms",
    "tabpfn_distill_ebm_terms": r"TabPFN$\rightarrow$EBM",
}

VARIANT_LABELS = {
    "pp_theta_post_ebm_bounded_residual_gate": "bounded residual gate",
    "pp_theta_post_rule_family_calibrated": "rule-family calibrated",
    "pp_theta_post_ebm_residual_mcc": "residual MCC",
    "pp_theta_post_ebm_residual_calibrated": "residual calibrated",
    "pp_theta_post_agreement_gated": "agreement gated",
    "pp_theta_post_operating_mcc": "MCC operating point",
    "pp_theta_post_ebm_residual_sens92": "residual Sens92",
    "pp_theta_post_ebm_residual_sens95": "residual Sens95",
    "pp_theta_post_bayes_llr_posneg": "Bayesian LLR",
    "pp_theta_post_family_utility_pruned_topk": "utility-pruned top-k",
    "pp_theta_post_monotone_ebm_families": "monotone EBM families",
}

STAGE_LABELS = {
    "rahmatullaev_proof_audit_sufficiency": "audit",
    "rahmatullaev_proof_evidence_ablation": "evidence",
    "rahmatullaev_proof_operating_points": "operating",
    "rahmatullaev_proof_randomized_controls": "controls",
    "rahmatullaev_proof_selective_utility": "selective",
    "rahmatullaev_proof_strong_base_repair": "residual",
}


def as_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "--"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_summary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path_s in glob.glob(str(ROOT / "*" / "*" / "ppost_proof_summary.csv")):
        path = Path(path_s)
        dataset = path.parts[-3]
        stage = path.parts[-2]
        for row in read_csv(path):
            row["dataset_key"] = dataset
            row["stage"] = stage
            rows.append(row)
    return rows


def load_control_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path_s in glob.glob(str(ROOT / "*" / "*" / "ppost_proof_controls.csv")):
        path = Path(path_s)
        dataset = path.parts[-3]
        stage = path.parts[-2]
        for row in read_csv(path):
            row["dataset_key"] = dataset
            row["stage"] = stage
            rows.append(row)
    return rows


def is_ppost_row(row: dict[str, str]) -> bool:
    return row.get("variant", "").startswith("pp_theta")


def select_main_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for dataset in ["eicu", "mimic3", "mimic4"]:
        candidates = [
            row
            for row in summary_rows
            if row.get("dataset_key") == dataset and is_ppost_row(row)
        ]
        best = max(
            candidates,
            key=lambda row: as_float(row.get("mean_delta_mcc")) or -999.0,
        )
        observed_mcc = as_float(best.get("observed_mcc"))
        delta_mcc = as_float(best.get("mean_delta_mcc"))
        native_mcc = None
        if observed_mcc is not None and delta_mcc is not None:
            native_mcc = observed_mcc - delta_mcc
        claim = {
            "eicu": "Audit signal; no MCC gain",
            "mimic3": "Utility + compact trace",
            "mimic4": "Utility + sensitivity gain",
        }[dataset]
        selected.append(
            {
                "dataset_key": dataset,
                "dataset": DATASET_LABELS[dataset],
                "claim": claim,
                "stage": best["stage"],
                "source": SOURCE_LABELS.get(best.get("rule_source", ""), best.get("rule_source", "")),
                "variant": VARIANT_LABELS.get(best.get("variant", ""), best.get("variant", "")),
                "native_mcc": native_mcc,
                "ppost_mcc": observed_mcc,
                "delta_mcc": delta_mcc,
                "delta_sensitivity": as_float(best.get("mean_delta_sensitivity")),
                "delta_brier": as_float(best.get("mean_delta_brier_score")),
                "delta_ece": as_float(best.get("mean_delta_ece_10")),
                "control_gap": as_float(best.get("observed_minus_permuted_mcc")),
                "trace_fraction": as_float(best.get("mean_trace_fraction")),
            }
        )
    return selected


def top_summary_rows(summary_rows: list[dict[str, str]], top_k: int = 6) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for dataset in ["eicu", "mimic3", "mimic4"]:
        candidates = [
            row
            for row in summary_rows
            if row.get("dataset_key") == dataset and is_ppost_row(row)
        ]
        candidates.sort(
            key=lambda row: as_float(row.get("mean_delta_mcc")) or -999.0,
            reverse=True,
        )
        for row in candidates[:top_k]:
            observed_mcc = as_float(row.get("observed_mcc"))
            delta_mcc = as_float(row.get("mean_delta_mcc"))
            native_mcc = None
            if observed_mcc is not None and delta_mcc is not None:
                native_mcc = observed_mcc - delta_mcc
            out.append(
                {
                    "dataset": DATASET_LABELS[dataset],
                    "stage": STAGE_LABELS.get(row.get("stage", ""), row.get("stage", "")),
                    "source": SOURCE_LABELS.get(row.get("rule_source", ""), row.get("rule_source", "")),
                    "variant": VARIANT_LABELS.get(row.get("variant", ""), row.get("variant", "")),
                    "native_mcc": native_mcc,
                    "ppost_mcc": observed_mcc,
                    "delta_mcc": delta_mcc,
                    "delta_sensitivity": as_float(row.get("mean_delta_sensitivity")),
                    "delta_brier": as_float(row.get("mean_delta_brier_score")),
                    "delta_ece": as_float(row.get("mean_delta_ece_10")),
                    "control_gap": as_float(row.get("observed_minus_permuted_mcc")),
                    "trace_fraction": as_float(row.get("mean_trace_fraction")),
                }
            )
    return out


def selected_control_rows(
    selected_rows: list[dict[str, object]],
    control_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    controls = [
        "control_permuted_patients",
        "control_class_prior_only",
        "control_temperature_flattened_t4",
    ]
    labels = {
        "control_permuted_patients": "patient-permuted",
        "control_class_prior_only": "class prior",
        "control_temperature_flattened_t4": r"flattened $T=4$",
    }
    out: list[dict[str, object]] = []
    for selected in selected_rows:
        dataset = str(selected["dataset_key"])
        stage = str(selected["stage"])
        variant = next(
            key
            for key, value in VARIANT_LABELS.items()
            if value == selected["variant"]
        )
        source = next(
            key
            for key, value in SOURCE_LABELS.items()
            if value == selected["source"]
        )
        stage_rows = [
            row
            for row in control_rows
            if row.get("dataset_key") == dataset
            and row.get("stage") == stage
            and row.get("rule_source") == source
            and row.get("variant") == variant
        ]
        observed = [
            row for row in stage_rows if row.get("control") == "ppost_observed"
        ]
        observed_mcc = mean(as_float(row.get("mcc")) for row in observed)
        for control in controls:
            rows = [row for row in stage_rows if row.get("control") == control]
            out.append(
                {
                    "dataset": DATASET_LABELS[dataset],
                    "source": selected["source"],
                    "variant": selected["variant"],
                    "control": labels[control],
                    "observed_mcc": observed_mcc,
                    "control_mcc": mean(as_float(row.get("mcc")) for row in rows),
                    "observed_minus_control_mcc": mean(
                        -(as_float(row.get("delta_vs_observed_mcc")) or 0.0)
                        for row in rows
                    ),
                    "observed_minus_control_sensitivity": mean(
                        -(as_float(row.get("delta_vs_observed_sensitivity")) or 0.0)
                        for row in rows
                    ),
                    "control_minus_observed_log_loss": mean(
                        as_float(row.get("delta_vs_observed_log_loss")) for row in rows
                    ),
                    "control_minus_observed_ece": mean(
                        as_float(row.get("delta_vs_observed_ece_10")) for row in rows
                    ),
                }
            )
    return out


def mean(values: Iterable[float | None]) -> float | None:
    vals = [value for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def write_main_table(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"Dataset and selected source & PP$\theta$ mode & Native MCC & +PP$\theta$ MCC & $\Delta$MCC & $\Delta$Sens. & Control gap & Trace frac. & Claim \\",
        r"\midrule",
    ]
    for row in rows:
        source = row["source"]
        dataset_source = f"{row['dataset']} / {source}"
        lines.append(
            " & ".join(
                [
                    str(dataset_source),
                    str(row["variant"]),
                    fmt(row["native_mcc"]),
                    fmt(row["ppost_mcc"]),
                    fmt(row["delta_mcc"], signed=True),
                    fmt(row["delta_sensitivity"], signed=True),
                    fmt(row["control_gap"]),
                    fmt(row["trace_fraction"]),
                    str(row["claim"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Final PP$\theta$-Post evidence-utility checks. For each dataset we select the teacher-free PP$\theta$ configuration with the best held-out $\Delta$MCC against its own native source. The eICU row is included as a negative boundary case: the trace remains non-random and compact, but MCC does not improve. Control gap is observed MCC minus patient-permuted evidence MCC; trace fraction is the fraction of posterior evidence retained by the exported compact trace.}",
            r"\label{tab:ppost-proof-main}",
            r"\end{table*}",
            "",
        ]
    )
    (OUT_DIR / "ppost_proof_main_table.tex").write_text("\n".join(lines))


def write_supp_utility_table(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Dataset & Source / PP$\theta$ mode & Stage & Native MCC & +PP$\theta$ MCC & $\Delta$MCC & $\Delta$Sens. & $\Delta$Brier & $\Delta$ECE & Trace frac. \\",
        r"\midrule",
    ]
    for row in rows:
        source_variant = f"{row['source']} / {row['variant']}"
        lines.append(
            " & ".join(
                [
                    str(row["dataset"]),
                    str(source_variant),
                    str(row["stage"]),
                    fmt(row["native_mcc"]),
                    fmt(row["ppost_mcc"]),
                    fmt(row["delta_mcc"], signed=True),
                    fmt(row["delta_sensitivity"], signed=True),
                    fmt(row["delta_brier"], signed=True),
                    fmt(row["delta_ece"], signed=True),
                    fmt(row["trace_fraction"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Top teacher-free PP$\theta$-Post proof rows by within-source $\Delta$MCC for each dataset. Positive $\Delta$ values mean that the posterior evidence layer improves the native source. Negative rows are retained to show substrate and dataset boundaries rather than hiding failed operating points.}",
            r"\label{tab:supp-ppost-proof-utility}",
            r"\end{table*}",
            "",
        ]
    )
    (OUT_DIR / "ppost_proof_supp_utility_table.tex").write_text("\n".join(lines))


def write_supp_control_table(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Dataset & Selected PP$\theta$ mode & Control & Observed MCC & Control MCC & MCC gap & Sens. gap & $\Delta$LogLoss & $\Delta$ECE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["dataset"]),
                    str(row["variant"]),
                    str(row["control"]),
                    fmt(row["observed_mcc"]),
                    fmt(row["control_mcc"]),
                    fmt(row["observed_minus_control_mcc"]),
                    fmt(row["observed_minus_control_sensitivity"]),
                    fmt(row["control_minus_observed_log_loss"], signed=True),
                    fmt(row["control_minus_observed_ece"], signed=True),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Controls for the selected main PP$\theta$ rows. Patient permutation and class-prior controls test whether the trace carries patient-specific signal. Temperature flattening can preserve the binary decision while damaging the probability scale; therefore $\Delta$LogLoss and $\Delta$ECE are reported as control minus observed values.}",
            r"\label{tab:supp-ppost-proof-controls}",
            r"\end{table*}",
            "",
        ]
    )
    (OUT_DIR / "ppost_proof_supp_controls_table.tex").write_text("\n".join(lines))


def write_markdown(
    selected: list[dict[str, object]],
    utility: list[dict[str, object]],
    controls: list[dict[str, object]],
) -> None:
    lines = [
        "# PPtheta-Post Proof Table Audit",
        "",
        f"Input root: `{ROOT}`",
        "",
        "## Main selected rows",
        "",
        "| Dataset | Source | Mode | Native MCC | +PPtheta MCC | dMCC | dSens | dBrier | dECE | Control gap | Trace |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {dataset} | {source} | {variant} | {native_mcc} | {ppost_mcc} | {delta_mcc} | {delta_sensitivity} | {delta_brier} | {delta_ece} | {control_gap} | {trace_fraction} |".format(
                dataset=row["dataset"],
                source=row["source"],
                variant=row["variant"],
                native_mcc=fmt(row["native_mcc"]),
                ppost_mcc=fmt(row["ppost_mcc"]),
                delta_mcc=fmt(row["delta_mcc"], signed=True),
                delta_sensitivity=fmt(row["delta_sensitivity"], signed=True),
                delta_brier=fmt(row["delta_brier"], signed=True),
                delta_ece=fmt(row["delta_ece"], signed=True),
                control_gap=fmt(row["control_gap"]),
                trace_fraction=fmt(row["trace_fraction"]),
            )
        )
    lines += [
        "",
        "## Randomized controls for selected rows",
        "",
        "| Dataset | Mode | Control | Observed MCC | Control MCC | MCC gap | Sensitivity gap | dLogLoss | dECE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controls:
        lines.append(
            "| {dataset} | {variant} | {control} | {observed_mcc} | {control_mcc} | {observed_minus_control_mcc} | {observed_minus_control_sensitivity} | {control_minus_observed_log_loss} | {control_minus_observed_ece} |".format(
                dataset=row["dataset"],
                variant=row["variant"],
                control=row["control"],
                observed_mcc=fmt(row["observed_mcc"]),
                control_mcc=fmt(row["control_mcc"]),
                observed_minus_control_mcc=fmt(row["observed_minus_control_mcc"]),
                observed_minus_control_sensitivity=fmt(row["observed_minus_control_sensitivity"]),
                control_minus_observed_log_loss=fmt(
                    row["control_minus_observed_log_loss"], signed=True
                ),
                control_minus_observed_ece=fmt(row["control_minus_observed_ece"], signed=True),
            )
        )
    lines += [
        "",
        "## Top utility rows",
        "",
        "| Dataset | Source | Mode | Stage | Native MCC | +PPtheta MCC | dMCC | dSens | dBrier | dECE | Trace |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in utility:
        lines.append(
            "| {dataset} | {source} | {variant} | {stage} | {native_mcc} | {ppost_mcc} | {delta_mcc} | {delta_sensitivity} | {delta_brier} | {delta_ece} | {trace_fraction} |".format(
                dataset=row["dataset"],
                source=row["source"],
                variant=row["variant"],
                stage=row["stage"],
                native_mcc=fmt(row["native_mcc"]),
                ppost_mcc=fmt(row["ppost_mcc"]),
                delta_mcc=fmt(row["delta_mcc"], signed=True),
                delta_sensitivity=fmt(row["delta_sensitivity"], signed=True),
                delta_brier=fmt(row["delta_brier"], signed=True),
                delta_ece=fmt(row["delta_ece"], signed=True),
                trace_fraction=fmt(row["trace_fraction"]),
            )
        )
    (OUT_DIR / "ppost_proof_table_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = load_summary_rows()
    control_rows = load_control_rows()
    if len(glob.glob(str(ROOT / "*" / "*" / "ppost_proof_summary.csv"))) != 18:
        raise SystemExit("Expected 18 completed proof summary files.")
    selected = select_main_rows(summary_rows)
    utility = top_summary_rows(summary_rows, top_k=6)
    controls = selected_control_rows(selected, control_rows)
    write_main_table(selected)
    write_supp_utility_table(utility)
    write_supp_control_table(controls)
    write_markdown(selected, utility, controls)
    write_csv(
        OUT_DIR / "ppost_proof_main_selected.csv",
        selected,
        [
            "dataset",
            "source",
            "variant",
            "native_mcc",
            "ppost_mcc",
            "delta_mcc",
            "delta_sensitivity",
            "delta_brier",
            "delta_ece",
            "control_gap",
            "trace_fraction",
            "claim",
        ],
    )
    write_csv(
        OUT_DIR / "ppost_proof_supp_utility.csv",
        utility,
        [
            "dataset",
            "stage",
            "source",
            "variant",
            "native_mcc",
            "ppost_mcc",
            "delta_mcc",
            "delta_sensitivity",
            "delta_brier",
            "delta_ece",
            "control_gap",
            "trace_fraction",
        ],
    )
    write_csv(
        OUT_DIR / "ppost_proof_supp_controls.csv",
        controls,
        [
            "dataset",
            "source",
            "variant",
            "control",
            "observed_mcc",
            "control_mcc",
            "observed_minus_control_mcc",
            "observed_minus_control_sensitivity",
            "control_minus_observed_log_loss",
            "control_minus_observed_ece",
        ],
    )
    print(f"Wrote PPtheta proof tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
