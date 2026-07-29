#!/usr/bin/env python3
"""Section 59: additional reviewer-stress experiments for AAAI framing.

These jobs do not redefine the paper as a SOTA mortality benchmark.  They test
reviewer-facing claims that were still fragile after the main strengthening
round: why PPtheta is not just EBM, where utility appears conditionally, how
compact traces behave, how teacher-anchored rows trade calibration for
sensitivity, and whether measurement-policy source choices explain eICU.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402
from paper_experiments.section_28_prediction_artifact_metrics import compute_metrics, normalize_proba  # noqa: E402
from paper_experiments.section_51_aaai_evidence_v2 import (  # noqa: E402
    GENERATED,
    SELECTED,
    _artifact_path,
    _compare_args,
    _dataset_key,
    _extract_option,
    _fast_binary_metrics,
    _latest_compare_csv,
    _load_prediction,
    _metrics,
    _out_dir,
    _read_csv,
    _selected_pairs,
    _stable_rng,
    _strip_options,
    _summarize_pairwise_csv,
    _write_csv,
    _write_md,
)

EVIDENCE_V2_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2"
METRICS = ("mcc", "balanced_accuracy", "sensitivity", "specificity", "auprc_ovr", "roc_auc_ovr", "log_loss", "brier_score", "ece_10")


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _delta_row(dataset: str, fold: str, source: str, variant: str, y: np.ndarray, base_p: np.ndarray, pp_p: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    mb = _metrics(y, base_p)
    mp = _metrics(y, pp_p)
    row: dict[str, Any] = {"dataset": dataset, "fold": fold, "source": source, "variant": variant, "n": int(len(y))}
    if extra:
        row.update(extra)
    for metric in METRICS:
        row[f"native_{metric}"] = mb[metric]
        row[f"ppost_{metric}"] = mp[metric]
        row[f"delta_{metric}"] = mp[metric] - mb[metric]
    return row


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({tuple(str(r.get(k, "")) for k in keys) for r in rows})
    for group in groups:
        part = [r for r in rows if tuple(str(r.get(k, "")) for k in keys) == group]
        rec: dict[str, Any] = {k: v for k, v in zip(keys, group)}
        rec["folds"] = len(part)
        rec["mean_n"] = _mean(float(r.get("n", 0)) for r in part)
        for metric in METRICS:
            for prefix in ("native", "ppost", "delta"):
                field = f"{prefix}_{metric}"
                if field in part[0]:
                    rec[field] = _mean(_float(r.get(field)) for r in part)
        for extra in ("audit_gap_mcc", "audit_gap_sensitivity", "mcc_retained_vs_full", "trace_fraction", "coverage", "mean_risk_shift"):
            vals = [_float(r.get(extra)) for r in part if extra in r]
            if vals:
                rec[extra] = _mean(vals)
        out.append(rec)
    return out


def _load_pair(row_native: dict[str, str], row_pp: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    lb = _load_prediction(row_native)
    lp = _load_prediction(row_pp)
    if lb is None or lp is None:
        return None
    y, pb = lb
    y2, pp = lp
    if not np.array_equal(y, y2):
        return None
    return y, pb, pp


def _source_matrix_rows(dataset: str) -> list[dict[str, str]]:
    path = EVIDENCE_V2_ROOT / dataset / "rahmatullaev_v2_source_compatibility_matrix"
    return _read_csv(_latest_compare_csv(path))


def _best_ppost_for_source(rows: list[dict[str, str]], source: str) -> str:
    candidates: dict[str, list[float]] = {}
    native = {(r.get("fold", ""), r.get("rule_source", "")): r for r in rows if r.get("variant") == "source_native"}
    for row in rows:
        if row.get("rule_source") != source or not row.get("variant", "").startswith("pp_theta"):
            continue
        base = native.get((row.get("fold", ""), source))
        if base is None:
            continue
        candidates.setdefault(row.get("variant", ""), []).append(_float(row.get("mcc")) - _float(base.get("mcc")))
    if not candidates:
        return "pp_theta_post_rule_family_calibrated"
    return max(candidates, key=lambda k: _mean(candidates[k]))


def _source_pairs(dataset: str, source: str, variant: str) -> list[tuple[dict[str, str], dict[str, str]]]:
    rows = _source_matrix_rows(dataset)
    native = {(r.get("fold", ""), r.get("rule_source", "")): r for r in rows if r.get("variant") == "source_native"}
    out = []
    for row in rows:
        if row.get("rule_source") == source and row.get("variant") == variant:
            base = native.get((row.get("fold", ""), source))
            if base is not None:
                out.append((base, row))
    return out


def run_ebm_vs_ppost_audit_mechanism(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows = _source_matrix_rows(dataset)
    variant = _best_ppost_for_source(rows, "ebm_terms")
    fold_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for base, pp in _source_pairs(dataset, "ebm_terms", variant):
        loaded = _load_pair(base, pp)
        if loaded is None:
            continue
        y, ebm_terms_p, pp_p = loaded
        fold = pp.get("fold", "")
        fold_rows.append(_delta_row(dataset, fold, "EBM terms", variant, y, ebm_terms_p, pp_p, {"audit_object": "posterior rule evidence over EBM-derived terms"}))
        rng = _stable_rng(str(pp.get("prediction_artifact", "")) + fold + "audit")
        controls = {
            "observed PPtheta trace": pp_p,
            "patient-permuted trace": pp_p[rng.permutation(len(y))],
            "flattened trace T=4": normalize_proba(np.exp(np.log(np.clip(pp_p, 1e-12, 1.0)) / 4.0)),
            "EBM-terms native score": ebm_terms_p,
        }
        obs = _metrics(y, pp_p)
        for name, pred in controls.items():
            mc = _metrics(y, pred)
            control_rows.append({
                "dataset": dataset,
                "fold": fold,
                "control": name,
                "mcc": mc["mcc"],
                "sensitivity": mc["sensitivity"],
                "brier_score": mc["brier_score"],
                "ece_10": mc["ece_10"],
                "audit_gap_mcc": obs["mcc"] - mc["mcc"],
                "audit_gap_sensitivity": obs["sensitivity"] - mc["sensitivity"],
            })
    summary = _summarize(fold_rows, ["dataset", "source", "variant", "audit_object"])
    controls = _summarize(control_rows, ["dataset", "control"])
    _write_csv(out / "ebm_vs_ppost_audit_folds.csv", fold_rows)
    _write_csv(out / "ebm_vs_ppost_audit_summary.csv", summary)
    _write_csv(out / "ebm_vs_ppost_audit_controls.csv", controls)
    _write_md(out / "ebm_vs_ppost_audit_mechanism.md", summary + controls, ["dataset", "source", "variant", "control", "delta_mcc", "delta_sensitivity", "audit_gap_mcc", "brier_score", "ece_10"])
    return 0


def run_conditional_utility_slices(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows_out: list[dict[str, Any]] = []
    for base, pp in _selected_pairs(dataset):
        loaded = _load_pair(base, pp)
        if loaded is None:
            continue
        y, pb, ppred = loaded
        base_pred = np.argmax(pb, axis=1)
        pp_pred = np.argmax(ppred, axis=1)
        p1b = pb[:, 1] if pb.shape[1] > 1 else pb[:, 0]
        p1p = ppred[:, 1] if ppred.shape[1] > 1 else ppred[:, 0]
        base_uncertainty = 1.0 - np.abs(2.0 * p1b - 1.0)
        pp_concentration = np.abs(2.0 * p1p - 1.0)
        shift = p1p - p1b
        masks = {
            "all patients": np.ones(len(y), dtype=bool),
            "mortality positives": y == 1,
            "native-source errors": base_pred != y,
            "native errors among positives": (base_pred != y) & (y == 1),
            "high native uncertainty": base_uncertainty >= np.quantile(base_uncertainty, 0.80),
            "largest posterior shifts": np.abs(shift) >= np.quantile(np.abs(shift), 0.80),
            "concentrated posterior evidence": pp_concentration >= np.quantile(pp_concentration, 0.80),
            "PPtheta corrections": (base_pred != y) & (pp_pred == y),
        }
        for subset, mask in masks.items():
            if int(mask.sum()) < 3 or len(np.unique(y[mask])) < 2:
                continue
            extra = {"subset": subset, "coverage": float(mask.mean()), "mean_risk_shift": float(np.mean(shift[mask]))}
            rows_out.append(_delta_row(dataset, pp.get("fold", ""), SELECTED[dataset]["source"], SELECTED[dataset]["variant"], y[mask], pb[mask], ppred[mask], extra))
    summary = _summarize(rows_out, ["dataset", "subset"])
    _write_csv(out / "conditional_utility_slices_folds.csv", rows_out)
    _write_csv(out / "conditional_utility_slices_summary.csv", summary)
    _write_md(out / "conditional_utility_slices.md", summary, ["dataset", "subset", "coverage", "mean_n", "delta_mcc", "delta_sensitivity", "delta_brier_score", "mean_risk_shift"])
    return 0


def run_trace_compression_curve_v2(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows = []
    for source_path in (
        GENERATED / "ppost_claim_trace_sufficiency_refresh.csv",
        GENERATED / "ppost_final_trace_sufficiency.csv",
        EVIDENCE_V2_ROOT / dataset / "rahmatullaev_v2_extended_trace_curve" / "extended_trace_curve_summary.csv",
    ):
        if source_path.exists() and source_path.stat().st_size > 0:
            for row in _read_csv(source_path):
                if row.get("dataset", row.get("dataset_key", "")).lower() in {dataset, SELECTED[dataset]["dataset_label"].lower()}:
                    item = dict(row)
                    item["source_artifact"] = str(source_path)
                    rows.append(item)
    normalized = []
    for row in rows:
        frac = _float(row.get("trace_fraction", row.get("requested_fraction")))
        retained = _float(row.get("mcc_retained_vs_full"))
        delta_mcc = _float(row.get("delta_mcc"))
        delta_sens = _float(row.get("delta_sensitivity"))
        normalized.append({
            "dataset": dataset,
            "source": row.get("source", row.get("rule_source", SELECTED[dataset]["source"])),
            "variant": row.get("variant", SELECTED[dataset]["variant"]),
            "requested_fraction": _float(row.get("budget_fraction", row.get("requested_fraction", frac))),
            "trace_fraction": frac,
            "mcc_retained_vs_full": retained,
            "delta_mcc": delta_mcc,
            "delta_sensitivity": delta_sens,
            "source_artifact": row.get("source_artifact", ""),
        })
    good = [r for r in normalized if math.isfinite(_float(r.get("mcc_retained_vs_full"))) and _float(r.get("mcc_retained_vs_full")) >= 0.99]
    selected = min(good, key=lambda r: _float(r.get("trace_fraction"))) if good else (max(normalized, key=lambda r: _float(r.get("mcc_retained_vs_full"), -1.0)) if normalized else {})
    _write_csv(out / "trace_compression_curve_v2.csv", normalized)
    _write_csv(out / "trace_compression_curve_v2_selected.csv", [selected] if selected else [])
    _write_md(out / "trace_compression_curve_v2.md", normalized, ["dataset", "requested_fraction", "trace_fraction", "mcc_retained_vs_full", "delta_mcc", "delta_sensitivity", "variant"])
    return 0


def run_teacher_anchor_calibration_modes(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    selected = SELECTED[dataset]
    variants = "source_native,pp_theta_post_teacher_anchored,pp_theta_post_teacher_calibrated,pp_theta_post_operating_calibrated,pp_theta_post_operating_mcc,pp_theta_post_operating_sens92"
    rc = run_compare_datasets(_compare_args(passthrough, out, selected["source"], variants, "ebm,tabpfn"))
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "teacher_anchor_calibration_modes")
    _write_md(out / "teacher_anchor_calibration_modes.md", summary, ["variant", "folds", "ppost_mcc", "delta_mcc", "ppost_sensitivity", "delta_sensitivity", "ppost_brier_score", "delta_brier_score", "ppost_ece_10"])
    return 0


def _dataset_path(passthrough: list[str]) -> Path:
    raw = _extract_option(passthrough, "--datasets", "") or ""
    if raw.startswith("npz:"):
        p = Path(raw[4:])
        return p if p.is_absolute() else ROOT / p
    key = _dataset_key(passthrough)
    return ROOT / f"data/processed/mortality/{key}_mortality_48h_tabular.npz"


def _measurement_npz(source_npz: Path, out_dir: Path, dataset: str) -> Path:
    arr = np.load(source_npz, allow_pickle=True)
    X = np.asarray(arr["X"], dtype=np.float32)
    y = np.asarray(arr["y"], dtype=np.int64)
    names = np.asarray(arr["feature_names"]).astype(str)
    keep = np.array([name.endswith("__count") or name.endswith("__frac_obs") or "frac_obs" in name or "__mask" in name for name in names], dtype=bool)
    if not keep.any():
        keep = np.array(["count" in name.lower() or "frac" in name.lower() or "mask" in name.lower() for name in names], dtype=bool)
    if not keep.any():
        raise ValueError(f"No measurement-pattern features found in {source_npz}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_measurement_policy_only.npz"
    np.savez_compressed(
        out_path,
        X=X[:, keep].astype(np.float32),
        y=y,
        feature_names=names[keep],
        class_names=np.asarray(arr["class_names"]).astype(str) if "class_names" in arr.files else np.asarray(["alive", "death"]),
        dataset_name=np.asarray(f"{dataset}_measurement_policy_only"),
    )
    return out_path


def run_eicu_measurement_policy_v2(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    source_npz = _dataset_path(passthrough)
    meas_npz = _measurement_npz(source_npz, out / "measurement_policy_npz", dataset)
    args = _strip_options(passthrough, {"--datasets", "--output-dir", "--rule-sources", "--variants", "--baselines", "--save-predictions"})
    args += [
        "--datasets", f"npz:{meas_npz}",
        "--output-dir", str(out),
        "--rule-sources", "rulefit,xgb,extratrees",
        "--variants", "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_ebm_residual_mcc,pp_theta_post_ebm_bounded_residual_gate",
        "--baselines", "none",
        "--save-predictions",
    ]
    rc = run_compare_datasets(args)
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "measurement_policy_v2")
    for row in summary:
        row["measurement_policy_scope"] = "measurement-pattern-only substrate"
    _write_csv(out / "measurement_policy_v2_summary.csv", summary)
    _write_md(out / "measurement_policy_v2.md", summary, ["rule_source", "variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score", "trace_fraction", "measurement_policy_scope"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=(
        "ebm_vs_ppost_audit_mechanism",
        "conditional_utility_slices",
        "trace_compression_curve_v2",
        "teacher_anchor_calibration_modes",
        "eicu_measurement_policy_v2",
    ))
    known, passthrough = parser.parse_known_args(argv)
    return {
        "ebm_vs_ppost_audit_mechanism": run_ebm_vs_ppost_audit_mechanism,
        "conditional_utility_slices": run_conditional_utility_slices,
        "trace_compression_curve_v2": run_trace_compression_curve_v2,
        "teacher_anchor_calibration_modes": run_teacher_anchor_calibration_modes,
        "eicu_measurement_policy_v2": run_eicu_measurement_policy_v2,
    }[known.experiment](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
