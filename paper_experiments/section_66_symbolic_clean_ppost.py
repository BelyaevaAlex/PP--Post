#!/usr/bin/env python3
"""Section 66: clean fully-interpretable PPtheta-Post evidence.

Each stage tests a teacher-free symbolic source under a fixed audit contract:
Native RuleFit/FIGS -> +PPtheta should improve MCC and sensitivity without
material calibration damage, and the observed posterior record should beat
patient-permuted or flattened controls. The stages are independent cluster jobs
and write the same CSV/Markdown artifacts as the rest of the mortality paper
protocol.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402
from paper_experiments.section_28_prediction_artifact_metrics import normalize_proba  # noqa: E402
from paper_experiments.section_51_aaai_evidence_v2 import (  # noqa: E402
    _compare_args,
    _dataset_key,
    _latest_compare_csv,
    _load_prediction,
    _metrics,
    _out_dir,
    _read_csv,
    _stable_rng,
    _summarize_pairwise_csv,
    _write_csv,
    _write_md,
)

METRICS = ("mcc", "balanced_accuracy", "sensitivity", "specificity", "auprc_ovr", "roc_auc_ovr", "log_loss", "brier_score", "ece_10")
DATASET_LABEL = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}
FULLY_SYMBOLIC_SOURCES = "rulefit,figs"
CAL_CONSTRAINT_BRIER = 0.002
CAL_CONSTRAINT_ECE = 0.005

STAGE_CONFIGS = {
    "rulefit_calibrated_evidence": {
        "sources": "rulefit",
        "variants": "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_bayes_llr_posneg",
        "summary_stem": "rulefit_calibrated_evidence",
        "description": "RuleFit + PPtheta calibrated/family evidence under calibration constraints",
    },
    "figs_bounded_residual": {
        "sources": "figs",
        "variants": "source_native,pp_theta_post_ebm_bounded_residual_gate,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk",
        "summary_stem": "figs_bounded_residual",
        "description": "FIGS + bounded/gated PPtheta residual and family evidence",
    },
    "symbolic_family_ppost": {
        "sources": FULLY_SYMBOLIC_SOURCES,
        "variants": "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_bayes_llr_posneg,pp_theta_post_rule_family_sensitivity",
        "summary_stem": "symbolic_family_ppost",
        "description": "Rule-family theta, redundancy-pruned top-k, and positive/negative symbolic evidence",
    },
    "calibration_constrained_thresholding": {
        "sources": FULLY_SYMBOLIC_SOURCES,
        "variants": "source_native,pp_theta_post_operating_calibrated,pp_theta_post_operating_mcc,pp_theta_post_operating_sens90,pp_theta_post_operating_sens92,pp_theta_post_operating_sens95",
        "summary_stem": "calibration_constrained_thresholding",
        "description": "Validation-only calibrated/MCC/sensitivity operating points for symbolic sources",
    },
    "rulefit_figs_auditselect": {
        "sources": FULLY_SYMBOLIC_SOURCES,
        "variants": "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_bayes_llr_posneg,pp_theta_post_ebm_bounded_residual_gate,pp_theta_post_operating_calibrated,pp_theta_post_operating_mcc,pp_theta_post_operating_sens90,pp_theta_post_operating_sens92,pp_theta_post_operating_sens95",
        "summary_stem": "rulefit_figs_auditselect",
        "description": "Pre-specified teacher-free AuditSelect over RuleFit/FIGS candidates",
    },
}


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def passes_relaxed(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("delta_mcc")) > 0.0
        and fnum(row.get("delta_sensitivity")) >= 0.0
        and fnum(row.get("delta_brier_score")) <= CAL_CONSTRAINT_BRIER
        and fnum(row.get("delta_ece_10")) <= CAL_CONSTRAINT_ECE
    )


def passes_strict(row: dict[str, Any]) -> bool:
    return (
        fnum(row.get("delta_mcc")) > 0.0
        and fnum(row.get("delta_sensitivity")) >= 0.0
        and fnum(row.get("delta_brier_score")) <= 0.0
        and fnum(row.get("delta_ece_10")) <= 0.0
    )


def selection_score(row: dict[str, Any]) -> float:
    return (
        fnum(row.get("delta_mcc"), -999.0)
        + 0.25 * fnum(row.get("delta_sensitivity"), 0.0)
        - 2.0 * max(fnum(row.get("delta_brier_score"), 0.0), 0.0)
        - 0.5 * max(fnum(row.get("delta_ece_10"), 0.0), 0.0)
    )


def enrich_summary(summary: list[dict[str, Any]], dataset: str, stage_name: str, description: str) -> list[dict[str, Any]]:
    out = []
    for row in summary:
        item = dict(row)
        item["dataset_key"] = dataset
        item["dataset_label"] = DATASET_LABEL[dataset]
        item["stage_name"] = stage_name
        item["experiment_claim"] = description
        item["teacher_at_inference"] = "no"
        item["source_family"] = "fully interpretable symbolic"
        item["passes_relaxed_calibration_constraint"] = passes_relaxed(item)
        item["passes_strict_calibration_constraint"] = passes_strict(item)
        item["selection_score"] = selection_score(item)
        out.append(item)
    return out


def select_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    relaxed = [r for r in rows if r["passes_relaxed_calibration_constraint"]]
    strict = [r for r in relaxed if r["passes_strict_calibration_constraint"]]
    pool = strict or relaxed or rows
    selected = max(pool, key=selection_score)
    out = dict(selected)
    out["decision"] = "select_ppost" if out["passes_relaxed_calibration_constraint"] else "fallback_to_native"
    out["selection_rule"] = "DeltaMCC>0, DeltaSensitivity>=0, DeltaBrier<=0.002, DeltaECE<=0.005; otherwise fallback"
    return out


def pair_rows(compare_csv: Path, selected: dict[str, Any]) -> list[tuple[dict[str, str], dict[str, str]]]:
    rows = _read_csv(compare_csv)
    source = str(selected.get("rule_source", ""))
    variant = str(selected.get("variant", ""))
    native = {(r.get("fold", ""), r.get("rule_source", "")): r for r in rows if r.get("variant") == "source_native"}
    out = []
    for row in rows:
        if row.get("rule_source") == source and row.get("variant") == variant:
            base = native.get((row.get("fold", ""), source))
            if base is not None:
                out.append((base, row))
    return out


def controls_for_selected(dataset: str, compare_csv: Path, selected: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for base, pp in pair_rows(compare_csv, selected):
        loaded_b = _load_prediction(base)
        loaded_p = _load_prediction(pp)
        if loaded_b is None or loaded_p is None:
            continue
        y, pb = loaded_b
        y2, ppred = loaded_p
        if not np.array_equal(y, y2):
            continue
        rng = _stable_rng(f"section66:{dataset}:{pp.get('fold','')}:{selected.get('rule_source')}:{selected.get('variant')}")
        observed = _metrics(y, ppred)
        native = _metrics(y, pb)
        controls = {
            "observed": ppred,
            "native_source": pb,
            "patient_permuted": ppred[rng.permutation(len(y))],
            "flattened_T4": normalize_proba(np.exp(np.log(np.clip(ppred, 1e-12, 1.0)) / 4.0)),
        }
        for control, pred in controls.items():
            met = _metrics(y, pred)
            row: dict[str, Any] = {
                "dataset_key": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "fold": pp.get("fold", ""),
                "rule_source": selected.get("rule_source", ""),
                "variant": selected.get("variant", ""),
                "control": control,
                "n": len(y),
                "observed_mcc": observed["mcc"],
                "native_mcc": native["mcc"],
                "control_gap_mcc": observed["mcc"] - met["mcc"],
                "control_gap_sensitivity": observed["sensitivity"] - met["sensitivity"],
            }
            for metric in METRICS:
                row[metric] = met[metric]
            fold_rows.append(row)
    summary: list[dict[str, Any]] = []
    keys = sorted({(r["control"], r["rule_source"], r["variant"]) for r in fold_rows})
    for control, source, variant in keys:
        part = [r for r in fold_rows if (r["control"], r["rule_source"], r["variant"]) == (control, source, variant)]
        rec: dict[str, Any] = {
            "dataset_key": dataset,
            "dataset_label": DATASET_LABEL[dataset],
            "rule_source": source,
            "variant": variant,
            "control": control,
            "folds": len(part),
            "mean_n": mean(fnum(r.get("n")) for r in part),
            "observed_mcc": mean(fnum(r.get("observed_mcc")) for r in part),
            "native_mcc": mean(fnum(r.get("native_mcc")) for r in part),
            "control_gap_mcc": mean(fnum(r.get("control_gap_mcc")) for r in part),
            "control_gap_sensitivity": mean(fnum(r.get("control_gap_sensitivity")) for r in part),
        }
        for metric in METRICS:
            rec[metric] = mean(fnum(r.get(metric)) for r in part)
        summary.append(rec)
    return fold_rows, summary


def run_stage(stage: str, passthrough: list[str]) -> int:
    cfg = STAGE_CONFIGS[stage]
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rc = run_compare_datasets(_compare_args(passthrough, out, cfg["sources"], cfg["variants"], "none"))
    if rc != 0:
        return rc
    compare_csv = _latest_compare_csv(out)
    raw_summary = _summarize_pairwise_csv(compare_csv, out / cfg["summary_stem"])
    summary = enrich_summary(raw_summary, dataset, stage, cfg["description"])
    selected = select_row(summary)
    controls_folds, controls_summary = controls_for_selected(dataset, compare_csv, selected) if selected else ([], [])
    _write_csv(out / f"{cfg['summary_stem']}_summary.csv", summary)
    _write_csv(out / f"{cfg['summary_stem']}_selected.csv", [selected] if selected else [])
    _write_csv(out / f"{cfg['summary_stem']}_controls_folds.csv", controls_folds)
    _write_csv(out / f"{cfg['summary_stem']}_controls_summary.csv", controls_summary)
    _write_md(out / f"{cfg['summary_stem']}.md", ([selected] if selected else []) + controls_summary, [
        "dataset_label", "experiment_claim", "rule_source", "variant", "decision", "native_mcc", "ppost_mcc",
        "delta_mcc", "delta_sensitivity", "delta_brier_score", "delta_ece_10",
        "passes_relaxed_calibration_constraint", "passes_strict_calibration_constraint",
        "control", "control_gap_mcc", "control_gap_sensitivity",
    ])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=tuple(STAGE_CONFIGS))
    known, passthrough = parser.parse_known_args(argv)
    return run_stage(known.experiment, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
