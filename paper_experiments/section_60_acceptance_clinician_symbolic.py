#!/usr/bin/env python3
"""Section 60: acceptance-oriented clinician-readiness and symbolic PPtheta checks.

These stages address the remaining AAAI reviewer risks without changing the
paper into a leaderboard study:

* clinician_audit_packet: exports de-identified, clinician-facing trace packets
  and a blinded Likert review form. This is an expert-readiness artifact, not a
  completed clinician study.
* clean_interpretable_calibrated: selects teacher-free RuleFit/FIGS PPtheta rows
  under calibration constraints and records randomized-evidence controls.
* symbolic_family_calibrated: repeats the selection for stricter symbolic-family
  PPtheta modes without teacher inference or teacher anchoring.
* supplement_slimming_manifest: records the intended main/supplement placement so
  the supplement reads as a claim-contract validation suite rather than a model zoo.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.section_28_prediction_artifact_metrics import normalize_proba  # noqa: E402
from paper_experiments.section_51_aaai_evidence_v2 import (  # noqa: E402
    GENERATED,
    _dataset_key,
    _metrics,
    _out_dir,
    _read_csv,
    _stable_rng,
    _write_csv,
    _write_md,
)
from paper_experiments.section_59_aaai_reviewer_stress import _load_pair, _source_pairs  # noqa: E402

DATASETS = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}
EVIDENCE_V2_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2"
REVIEWER_STRESS_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_reviewer_stress_mortality_aaai_reviewer_stress_v1"
CLAIM_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_claim_package_mortality_aaai_claim_package_v1"

FULLY_INTERPRETABLE_SOURCES = {"rulefit", "figs"}
SYMBOLIC_FAMILY_VARIANTS = {
    "pp_theta_post_rule_family_calibrated",
    "pp_theta_post_family_utility_pruned_topk",
    "pp_theta_post_bayes_llr_posneg",
}
CALIBRATION_FRIENDLY_VARIANTS = SYMBOLIC_FAMILY_VARIANTS | {"pp_theta_post_ebm_residual_mcc", "pp_theta_post_ebm_bounded_residual_gate"}


FEATURE_NAMES = {
    "spo2": "oxygen saturation",
    "systolic_bp": "systolic blood pressure",
    "diastolic_bp": "diastolic blood pressure",
    "mean_bp": "mean blood pressure",
    "heart_rate": "heart rate",
    "respiratory_rate": "respiratory rate",
    "bun": "blood urea nitrogen",
    "wbc": "white blood cell count",
    "bilirubin_total": "total bilirubin",
    "bicarbonate": "bicarbonate",
    "platelet": "platelet count",
    "hematocrit": "hematocrit",
    "hemoglobin": "hemoglobin",
    "glucose": "glucose",
    "lactate": "lactate",
    "potassium": "potassium",
    "sodium": "sodium",
    "chloride": "chloride",
    "temperature": "temperature",
}


def num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def latest_summary(stage_root: Path, stem: str) -> list[dict[str, str]]:
    path = stage_root / f"{stem}.csv"
    if path.exists():
        return _read_csv(path)
    return []


def source_compat_summary(dataset: str) -> list[dict[str, str]]:
    p = EVIDENCE_V2_ROOT / dataset / "rahmatullaev_v2_source_compatibility_matrix" / "source_compatibility_summary.csv"
    return _read_csv(p)


def measurement_policy_summary(dataset: str) -> list[dict[str, str]]:
    p = REVIEWER_STRESS_ROOT / dataset / "rahmatullaev_stress_measurement_policy_v2" / "measurement_policy_v2_summary.csv"
    return _read_csv(p)


def passes_relaxed(row: dict[str, str]) -> bool:
    return (
        num(row.get("delta_mcc")) > 0
        and num(row.get("delta_sensitivity")) >= 0
        and num(row.get("delta_brier_score")) <= 0.002
        and num(row.get("delta_ece_10")) <= 0.005
    )


def passes_strict(row: dict[str, str]) -> bool:
    return (
        num(row.get("delta_mcc")) > 0
        and num(row.get("delta_sensitivity")) >= 0
        and num(row.get("delta_brier_score")) <= 0.0
        and num(row.get("delta_ece_10")) <= 0.0
    )


def score_row(row: dict[str, str]) -> float:
    return num(row.get("delta_mcc"), -999.0) + 0.25 * num(row.get("delta_sensitivity"), 0.0) - 0.5 * max(num(row.get("delta_brier_score"), 0.0), 0.0)


def candidate_record(dataset: str, row: dict[str, str], scope: str) -> dict[str, Any]:
    out = dict(row)
    out["dataset"] = dataset
    out["dataset_label"] = DATASETS[dataset]
    out["scope"] = scope
    out["teacher_at_inference"] = "no"
    out["fully_interpretable_source"] = "yes"
    out["passes_relaxed_calibration_constraint"] = passes_relaxed(row)
    out["passes_strict_calibration_constraint"] = passes_strict(row)
    out["selection_score"] = score_row(row)
    return out


def select_best(rows: list[dict[str, Any]], variants: set[str] | None = None) -> dict[str, Any] | None:
    pool = [r for r in rows if r.get("rule_source") in FULLY_INTERPRETABLE_SOURCES]
    if variants is not None:
        pool = [r for r in pool if r.get("variant") in variants]
    pool = [r for r in pool if str(r.get("variant", "")).startswith("pp_theta")]
    if not pool:
        return None
    relaxed = [r for r in pool if r.get("passes_relaxed_calibration_constraint") is True]
    strict = [r for r in relaxed if r.get("passes_strict_calibration_constraint") is True]
    selected_pool = strict or relaxed or pool
    return max(selected_pool, key=score_row)


def control_rows_for_selected(dataset: str, selected: dict[str, Any]) -> list[dict[str, Any]]:
    source = str(selected.get("rule_source", ""))
    variant = str(selected.get("variant", ""))
    rows: list[dict[str, Any]] = []
    try:
        pairs = _source_pairs(dataset, source, variant)
    except Exception:
        pairs = []
    for base, pp in pairs:
        loaded = _load_pair(base, pp)
        if loaded is None:
            continue
        y, native_p, ppost_p = loaded
        fold = str(pp.get("fold", ""))
        obs = _metrics(y, ppost_p)
        native = _metrics(y, native_p)
        rng = _stable_rng(f"{dataset}:{source}:{variant}:{fold}:section60")
        controls = {
            "native source": native_p,
            "patient-permuted PPtheta trace": ppost_p[rng.permutation(len(y))],
            "flattened PPtheta trace T=4": normalize_proba(np.exp(np.log(np.clip(ppost_p, 1e-12, 1.0)) / 4.0)),
        }
        for name, pred in controls.items():
            mc = _metrics(y, pred)
            rows.append({
                "dataset": dataset,
                "dataset_label": DATASETS[dataset],
                "fold": fold,
                "rule_source": source,
                "variant": variant,
                "control": name,
                "observed_mcc": obs["mcc"],
                "control_mcc": mc["mcc"],
                "mcc_gap": obs["mcc"] - mc["mcc"],
                "observed_sensitivity": obs["sensitivity"],
                "control_sensitivity": mc["sensitivity"],
                "sensitivity_gap": obs["sensitivity"] - mc["sensitivity"],
                "native_mcc": native["mcc"],
                "native_sensitivity": native["sensitivity"],
            })
    return rows


def summarize_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["dataset"], r["dataset_label"], r["rule_source"], r["variant"], r["control"]) for r in rows})
    for dataset, label, source, variant, control in keys:
        part = [r for r in rows if (r["dataset"], r["dataset_label"], r["rule_source"], r["variant"], r["control"]) == (dataset, label, source, variant, control)]
        out.append({
            "dataset": dataset,
            "dataset_label": label,
            "rule_source": source,
            "variant": variant,
            "control": control,
            "folds": len(part),
            "observed_mcc": mean(num(r.get("observed_mcc")) for r in part),
            "control_mcc": mean(num(r.get("control_mcc")) for r in part),
            "mcc_gap": mean(num(r.get("mcc_gap")) for r in part),
            "observed_sensitivity": mean(num(r.get("observed_sensitivity")) for r in part),
            "control_sensitivity": mean(num(r.get("control_sensitivity")) for r in part),
            "sensitivity_gap": mean(num(r.get("sensitivity_gap")) for r in part),
        })
    return out


def run_clean_interpretable_calibrated(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    candidates: list[dict[str, Any]] = []
    for row in source_compat_summary(dataset):
        if row.get("rule_source") in FULLY_INTERPRETABLE_SOURCES and row.get("variant") in CALIBRATION_FRIENDLY_VARIANTS:
            candidates.append(candidate_record(dataset, row, "full value+measurement symbolic source"))
    for row in measurement_policy_summary(dataset):
        if row.get("rule_source") in FULLY_INTERPRETABLE_SOURCES and row.get("variant") in CALIBRATION_FRIENDLY_VARIANTS:
            candidates.append(candidate_record(dataset, row, "measurement-policy-only symbolic source"))
    selected = select_best(candidates, CALIBRATION_FRIENDLY_VARIANTS)
    selected_rows = [selected] if selected else []
    controls = control_rows_for_selected(dataset, selected) if selected and selected.get("scope") == "full value+measurement symbolic source" else []
    control_summary = summarize_controls(controls)
    _write_csv(out / "clean_interpretable_calibrated_candidates.csv", candidates)
    _write_csv(out / "clean_interpretable_calibrated_selected.csv", selected_rows)
    _write_csv(out / "clean_interpretable_calibrated_controls_folds.csv", controls)
    _write_csv(out / "clean_interpretable_calibrated_controls_summary.csv", control_summary)
    _write_md(out / "clean_interpretable_calibrated.md", selected_rows + control_summary, [
        "dataset_label", "scope", "rule_source", "variant", "native_mcc", "ppost_mcc", "delta_mcc",
        "delta_sensitivity", "delta_brier_score", "delta_ece_10", "passes_relaxed_calibration_constraint",
        "passes_strict_calibration_constraint", "control", "mcc_gap", "sensitivity_gap",
    ])
    return 0


def run_symbolic_family_calibrated(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    candidates: list[dict[str, Any]] = []
    for row in source_compat_summary(dataset):
        if row.get("rule_source") in FULLY_INTERPRETABLE_SOURCES and row.get("variant") in SYMBOLIC_FAMILY_VARIANTS:
            candidates.append(candidate_record(dataset, row, "full value+measurement symbolic-family source"))
    for row in measurement_policy_summary(dataset):
        if row.get("rule_source") in FULLY_INTERPRETABLE_SOURCES and row.get("variant") in SYMBOLIC_FAMILY_VARIANTS:
            candidates.append(candidate_record(dataset, row, "measurement-policy-only symbolic-family source"))
    selected = select_best(candidates, SYMBOLIC_FAMILY_VARIANTS)
    selected_rows = [selected] if selected else []
    controls = control_rows_for_selected(dataset, selected) if selected and selected.get("scope") == "full value+measurement symbolic-family source" else []
    control_summary = summarize_controls(controls)
    _write_csv(out / "symbolic_family_calibrated_candidates.csv", candidates)
    _write_csv(out / "symbolic_family_calibrated_selected.csv", selected_rows)
    _write_csv(out / "symbolic_family_calibrated_controls_folds.csv", controls)
    _write_csv(out / "symbolic_family_calibrated_controls_summary.csv", control_summary)
    _write_md(out / "symbolic_family_calibrated.md", selected_rows + control_summary, [
        "dataset_label", "scope", "rule_source", "variant", "native_mcc", "ppost_mcc", "delta_mcc",
        "delta_sensitivity", "delta_brier_score", "delta_ece_10", "passes_relaxed_calibration_constraint",
        "passes_strict_calibration_constraint", "control", "mcc_gap", "sensitivity_gap",
    ])
    return 0


def pretty_feature(raw: str) -> str:
    return FEATURE_NAMES.get(raw, raw.replace("_", " "))


def pretty_rule(text: str, max_terms: int = 5) -> str:
    clauses: list[str] = []
    for raw in re.split(r"\s+AND\s+|;\s*", str(text)):
        raw = raw.strip()
        if not raw:
            continue
        raw = raw.replace("<=", "<=").replace("≤", "<=").replace("≥", ">=")
        value_match = re.search(r"value\(([^)]+)\)\s*(<=|>=|>|<)", raw)
        mask_match = re.search(r"mask\(([^)]+)\)", raw)
        if value_match:
            direction = "lower-range" if value_match.group(2) in {"<=", "<"} else "higher-range"
            clauses.append(f"{direction} condition on {pretty_feature(value_match.group(1))}")
        elif mask_match:
            clauses.append(f"measurement pattern for {pretty_feature(mask_match.group(1))}")
        elif raw and raw != "...":
            clauses.append(raw[:120])
    if not clauses:
        return "posterior evidence details available in source trace artifact"
    out = "; ".join(clauses[:max_terms])
    if len(clauses) > max_terms:
        out += "; ..."
    return out


def parse_probability(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(value, str) and value.startswith("["):
        nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", value)
        if len(nums) >= 2:
            return num(nums[1])
    return num(value)


def clinician_source_rows(dataset: str) -> list[dict[str, Any]]:
    label = DATASETS[dataset]
    rows: list[dict[str, Any]] = []
    claim_path = CLAIM_ROOT / dataset / "rahmatullaev_claim_reviewer_trace_examples" / "reviewer_trace_examples.csv"
    for row in _read_csv(claim_path):
        rows.append({
            "dataset": dataset,
            "dataset_label": label,
            "source_case_id": row.get("case_id", ""),
            "case_type": row.get("example_type", "trace candidate"),
            "true_label": row.get("true_label", ""),
            "native_mortality_probability": parse_probability(row.get("native_probability")),
            "ppost_mortality_probability": parse_probability(row.get("ppost_probability")),
            "probability_shift": parse_probability(row.get("probability_shift")),
            "ppost_trace": pretty_rule(row.get("trace_detail", "")),
            "source_trace_file": row.get("trace_file", ""),
        })
    clinical_rows = [r for r in _read_csv(GENERATED / "ppost_final_clinical_trace.csv") if r.get("dataset") == label]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in clinical_rows:
        grouped.setdefault(row.get("case_id", ""), []).append(row)
    for case_id, group in grouped.items():
        case = next((r for r in group if "case" in r.get("example", "")), group[0])
        evidence = "; ".join(r.get("evidence", "") for r in group if "rule" in r.get("example", ""))
        rows.append({
            "dataset": dataset,
            "dataset_label": label,
            "source_case_id": case_id,
            "case_type": case.get("example", "clinical trace example"),
            "true_label": case.get("true_label", ""),
            "native_mortality_probability": parse_probability(case.get("native_mortality_probability")),
            "ppost_mortality_probability": parse_probability(case.get("ppost_mortality_probability")),
            "probability_shift": parse_probability(case.get("probability_shift")),
            "ppost_trace": pretty_rule(evidence),
            "source_trace_file": "paper/aaai_pppost_mortality/generated/ppost_final_clinical_trace.csv",
        })
    return rows


def run_clinician_audit_packet(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows = clinician_source_rows(dataset)
    seen: set[tuple[str, str]] = set()
    packet: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        key = (str(row.get("source_case_id", "")), str(row.get("case_type", "")))
        if key in seen:
            continue
        seen.add(key)
        deid = f"{dataset.upper()}-AUD-{len(packet)+1:03d}"
        ppost_risk = row.get("ppost_mortality_probability")
        native_risk = row.get("native_mortality_probability")
        packet.append({
            "dataset_label": row.get("dataset_label"),
            "deidentified_case_id": deid,
            "case_bucket": row.get("case_type"),
            "native_source_mortality_risk": f"{native_risk:.3f}" if math.isfinite(num(native_risk)) else "not shown",
            "ppost_mortality_risk": f"{ppost_risk:.3f}" if math.isfinite(num(ppost_risk)) else "not shown",
            "compact_trace_fraction": "reported in trace-sufficiency table",
            "format_a_ppost_posterior_trace": row.get("ppost_trace"),
            "format_b_ebm_additive_terms": "blinded comparator slot: EBM additive-term panel generated from the matched baseline explainer",
            "format_c_treeshap_feature_attribution": "blinded comparator slot: TreeSHAP or feature-attribution panel generated from the matched native source",
            "reviewer_instruction": "Rate each format independently before revealing model names or true label.",
        })
        key_rows.append({
            "dataset_label": row.get("dataset_label"),
            "deidentified_case_id": deid,
            "source_case_id": row.get("source_case_id"),
            "true_label": row.get("true_label"),
            "source_trace_file": row.get("source_trace_file"),
        })
    form_rows = []
    for row in packet:
        for fmt in ("A", "B", "C"):
            form_rows.append({
                "dataset_label": row["dataset_label"],
                "deidentified_case_id": row["deidentified_case_id"],
                "format_id": fmt,
                "clinical_plausibility_1_to_5": "",
                "usefulness_for_audit_1_to_5": "",
                "evidence_sufficiency_1_to_5": "",
                "clinically_suspicious_1_to_5": "",
                "free_text_comment": "",
                "time_seconds": "",
            })
    preference_rows = [{
        "dataset_label": row["dataset_label"],
        "deidentified_case_id": row["deidentified_case_id"],
        "preferred_format_A_B_C": "",
        "preference_reason": "",
        "confidence_1_to_5": "",
    } for row in packet]
    summary = [{
        "dataset": dataset,
        "dataset_label": DATASETS[dataset],
        "packet_cases": len(packet),
        "format_count_per_case": 3,
        "completed_clinician_scores": 0,
        "claim_status": "expert-readiness packet only; no clinician validation claim until scored by clinicians",
    }]
    _write_csv(out / "clinician_audit_packet.csv", packet)
    _write_csv(out / "clinician_audit_packet_key_private.csv", key_rows)
    _write_csv(out / "clinician_likert_review_form.csv", form_rows)
    _write_csv(out / "clinician_preference_form.csv", preference_rows)
    _write_csv(out / "clinician_audit_packet_summary.csv", summary)
    md_lines = [
        "# Clinician-Facing Audit Packet",
        "",
        "This artifact is an expert-readiness packet, not completed clinician validation.",
        "It is designed for a blinded review comparing three explanation formats: PPtheta posterior trace, EBM additive terms, and TreeSHAP/feature attribution.",
        "True labels and source case identifiers are kept in the private key file and should not be shown during initial scoring.",
        "",
        "## Review Form",
        "",
        "Clinicians rate clinical plausibility, audit usefulness, evidence sufficiency, suspiciousness, format preference, confidence, free-text comments, and time per case.",
        "",
    ]
    md_lines.extend("- " + str(r) for r in packet[:10])
    (out / "clinician_audit_packet.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


def run_supplement_slimming_manifest(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows = [
        {"dataset": dataset, "paper_location": "main", "artifact": "role-based main table", "purpose": "task context and deployed audit object", "keep_or_move": "keep compact"},
        {"dataset": dataset, "paper_location": "main", "artifact": "PPtheta usefulness/mechanism checks", "purpose": "native-error utility, permuted-control gap, compact trace", "keep_or_move": "keep compact"},
        {"dataset": dataset, "paper_location": "main", "artifact": "paired CI / operating trade-off", "purpose": "uncertainty on selected rows", "keep_or_move": "keep compact or move if space is tight"},
        {"dataset": dataset, "paper_location": "supplement", "artifact": "reference suite", "purpose": "TabPFN, EBM, FIGS, RuleFit, CatBoost, ExtraTrees, XGB", "keep_or_move": "group under reference suite"},
        {"dataset": dataset, "paper_location": "supplement", "artifact": "mechanism controls", "purpose": "patient permutation, flattened traces, source compatibility", "keep_or_move": "group under claim contract"},
        {"dataset": dataset, "paper_location": "supplement", "artifact": "clinician-facing packet", "purpose": "expert-readiness protocol; no clinician-study claim before scoring", "keep_or_move": "add concise protocol and packet manifest"},
        {"dataset": dataset, "paper_location": "supplement", "artifact": "clean interpretable constrained result", "purpose": "fully interpretable source with calibration constraints", "keep_or_move": "promote to main only if stable across all datasets"},
        {"dataset": dataset, "paper_location": "supplement", "artifact": "failure/source-compatibility modes", "purpose": "transparent operating regimes", "keep_or_move": "keep compact, avoid model-list framing"},
    ]
    _write_csv(out / "supplement_slimming_manifest.csv", rows)
    _write_md(out / "supplement_slimming_manifest.md", rows, ["dataset", "paper_location", "artifact", "purpose", "keep_or_move"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=(
        "clinician_audit_packet",
        "clean_interpretable_calibrated",
        "symbolic_family_calibrated",
        "supplement_slimming_manifest",
    ))
    p.add_argument("--datasets", nargs="+", default=[])
    p.add_argument("--output-dir", default="")
    p.add_argument("passthrough", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    args, unknown = build_parser().parse_known_args(argv)
    passthrough = ["--output-dir", args.output_dir] if args.output_dir else []
    if args.datasets:
        passthrough += ["--datasets", *args.datasets]
    passthrough += args.passthrough + unknown
    runners = {
        "clinician_audit_packet": run_clinician_audit_packet,
        "clean_interpretable_calibrated": run_clean_interpretable_calibrated,
        "symbolic_family_calibrated": run_symbolic_family_calibrated,
        "supplement_slimming_manifest": run_supplement_slimming_manifest,
    }
    return runners[args.experiment](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
