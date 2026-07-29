#!/usr/bin/env python3
"""Section 46: PPtheta-Post utility proof suite.

The goal of this section is different from another broad model sweep.  It
creates ordinary compare_datasets rows, saves per-fold prediction artifacts, and
then writes proof-oriented tables:

* pairwise deltas against the native interpretable substrate;
* diagnostic hard-slice gains;
* deterministic randomized/prior controls;
* compactness and operating-point summaries.

These tables support the paper claim that PPtheta-Post is useful as a
prediction-time posterior evidence object, even when a strong native substrate
such as EBM is already competitive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from paper_experiments.section_28_prediction_artifact_metrics import (  # noqa: E402
    compute_metrics,
    normalize_proba,
)

CORE_PPOST = (
    "pp_theta_post_rule_family_calibrated,"
    "pp_theta_post_family_utility_pruned_topk,"
    "pp_theta_post_bayes_llr_posneg,"
    "pp_theta_post_ebm_residual_calibrated"
)
GATED_PPOST = (
    "pp_theta_post_ebm_bounded_residual_gate,"
    "pp_theta_post_agreement_gated,"
    "pp_theta_post_family_utility_pruned_topk"
)
REPAIR_PPOST = (
    "pp_theta_post_ebm_residual_calibrated,"
    "pp_theta_post_ebm_residual_mcc,"
    "pp_theta_post_ebm_residual_sens92,"
    "pp_theta_post_ebm_residual_sens95"
)
OPERATING_PPOST = (
    "pp_theta_post_operating_calibrated,"
    "pp_theta_post_operating_mcc,"
    "pp_theta_post_operating_sens90,"
    "pp_theta_post_operating_sens92,"
    "pp_theta_post_operating_sens95"
)

EXPERIMENTS = {
    "proof_evidence_ablation": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "source_native," + CORE_PPOST,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "proof_selective_utility": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "source_native," + GATED_PPOST + ",pp_theta_post_operating_mcc",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "proof_strong_base_repair": [
        "--rule-sources", "ebm_terms,tabpfn_distill_ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "source_native," + REPAIR_PPOST,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "proof_audit_sufficiency": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", (
            "source_native,"
            "pp_theta_post_rule_family_calibrated,"
            "pp_theta_post_family_utility_pruned_topk,"
            "pp_theta_post_bayes_llr_posneg,"
            "pp_theta_post_monotone_ebm_families"
        ),
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "proof_operating_points": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", OPERATING_PPOST,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "proof_randomized_controls": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", (
            "source_native,"
            "pp_theta_post_rule_family_calibrated,"
            "pp_theta_post_family_utility_pruned_topk,"
            "pp_theta_post_bayes_llr_posneg,"
            "pp_theta_post_ebm_residual_calibrated"
        ),
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
}

EXPERIMENT_ENVS = {
    "proof_evidence_ablation": {
        "PPPOST_FAMILY_UTILITY_TOPK": "32",
        "PPPOST_BAYES_LLR_TOPK": "32",
        "PPPOST_BAYES_LLR_BETA_STRENGTH": "72",
        "PPPOST_EBM_RESIDUAL_TOPK": "24",
    },
    "proof_selective_utility": {
        "PPPOST_FAMILY_UTILITY_TOPK": "16",
        "PPPOST_EBM_RESIDUAL_MAX_SCALE": "0.35",
        "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL": "0.008",
        "PPPOST_EBM_RESIDUAL_ACC_TOL": "0.004",
    },
    "proof_strong_base_repair": {
        "PPPOST_EBM_RESIDUAL_TOPK": "24",
        "PPPOST_EBM_RESIDUAL_TRUE_WEIGHT": "0.60",
        "PPPOST_EBM_RESIDUAL_RIDGE_L2": "5.0",
        "PPPOST_EBM_RESIDUAL_MAX_SCALE": "0.40",
        "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL": "0.010",
    },
    "proof_audit_sufficiency": {
        "PPPOST_FAMILY_UTILITY_TOPK": "16",
        "PPPOST_BAYES_LLR_TOPK": "16",
        "PPPOST_BAYES_LLR_BETA_STRENGTH": "96",
    },
    "proof_operating_points": {
        "PPPOST_FAMILY_UTILITY_TOPK": "32",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
    "proof_randomized_controls": {
        "PPPOST_FAMILY_UTILITY_TOPK": "24",
        "PPPOST_BAYES_LLR_TOPK": "24",
        "PPPOST_EBM_RESIDUAL_TOPK": "24",
    },
}

METRICS = (
    "mcc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auprc_ovr",
    "roc_auc_ovr",
    "log_loss",
    "brier_score",
    "ece_10",
)


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compare_csvs(out_dir: Path) -> list[Path]:
    return sorted(
        p for p in out_dir.glob("compare_datasets_*.csv")
        if not p.name.startswith("ppost_proof_")
    )


def _latest_compare_csv(out_dir: Path) -> Path | None:
    csvs = _compare_csvs(out_dir)
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _artifact_path(row: dict[str, Any]) -> Path | None:
    raw = str(row.get("prediction_artifact", "") or "")
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [path, ROOT / path, ROOT.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_artifact(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    path = _artifact_path(row)
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["y_true"], dtype=int), normalize_proba(np.asarray(data["proba"], dtype=float))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("fold", "")), str(row.get("rule_source", "")))


def _method_id(row: dict[str, Any]) -> str:
    return f"{row.get('rule_source', '')}+{row.get('variant', '')}"


def _is_ppost(row: dict[str, Any]) -> bool:
    return str(row.get("variant", "")).startswith("pp_theta_post")


def _metric_deltas(base: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric in METRICS:
        lhs = _float(row, metric)
        rhs = _float(base, metric)
        out[f"delta_{metric}"] = lhs - rhs if lhs == lhs and rhs == rhs else float("nan")
    return out


def _compute_on_mask(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 3 or len(np.unique(y[mask])) < 2:
        return {metric: float("nan") for metric in METRICS}
    return compute_metrics(y[mask], p[mask], p.shape[1])


def _stable_permutation(n: int, text: str) -> np.ndarray:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def _prior_proba(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    prior = counts / max(float(counts.sum()), 1.0)
    return np.repeat(prior.reshape(1, -1), len(y), axis=0)


def _flatten_proba(p: np.ndarray, temperature: float = 4.0) -> np.ndarray:
    eps = 1e-12
    logits = np.log(np.clip(p, eps, 1.0))
    logits = logits / max(float(temperature), eps)
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return normalize_proba(out)


def _slice_masks(y: np.ndarray, base_p: np.ndarray, pp_p: np.ndarray) -> list[tuple[str, np.ndarray]]:
    base_pred = np.argmax(base_p, axis=1)
    pp_pred = np.argmax(pp_p, axis=1)
    if base_p.shape[1] == 2:
        base_unc = 1.0 - np.abs(2.0 * base_p[:, 1] - 1.0)
        correction = np.abs(pp_p[:, 1] - base_p[:, 1])
    else:
        base_unc = 1.0 - np.max(base_p, axis=1)
        correction = np.max(np.abs(pp_p - base_p), axis=1)
    corr_thr = np.quantile(correction, 0.80) if len(correction) else math.inf
    unc_thr = np.quantile(base_unc, 0.67) if len(base_unc) else math.inf
    return [
        ("all", np.ones(len(y), dtype=bool)),
        ("base_uncertain_top_tertile", base_unc >= unc_thr),
        ("pp_large_correction_top_quintile", correction >= corr_thr),
        ("base_pp_disagree", base_pred != pp_pred),
        ("base_wrong_pp_right", (base_pred != y) & (pp_pred == y)),
    ]


def _pairwise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_key = {
        _row_key(row): row
        for row in rows
        if row.get("variant") == "source_native"
    }
    out = []
    for row in rows:
        if not _is_ppost(row):
            continue
        base = base_by_key.get(_row_key(row))
        if base is None:
            continue
        item = {
            "dataset": row.get("dataset", ""),
            "fold": row.get("fold", ""),
            "rule_source": row.get("rule_source", ""),
            "base_label": base.get("label", ""),
            "ppost_label": row.get("label", ""),
            "variant": row.get("variant", ""),
            "n_test": row.get("n_test", ""),
            "n_branches": row.get("n_branches", ""),
            "top_k": row.get("top_k", ""),
        }
        item.update({f"base_{m}": _float(base, m) for m in METRICS})
        item.update({f"ppost_{m}": _float(row, m) for m in METRICS})
        item.update(_metric_deltas(base, row))
        out.append(item)
    return out


def _selective_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_key = {
        _row_key(row): row
        for row in rows
        if row.get("variant") == "source_native"
    }
    out = []
    for row in rows:
        if not _is_ppost(row):
            continue
        base = base_by_key.get(_row_key(row))
        if base is None:
            continue
        loaded_base = _load_artifact(base)
        loaded_pp = _load_artifact(row)
        if loaded_base is None or loaded_pp is None:
            continue
        y, base_p = loaded_base
        y_pp, pp_p = loaded_pp
        if len(y) != len(y_pp) or not np.array_equal(y, y_pp):
            continue
        for slice_name, mask in _slice_masks(y, base_p, pp_p):
            base_metrics = _compute_on_mask(y, base_p, mask)
            pp_metrics = _compute_on_mask(y, pp_p, mask)
            item = {
                "dataset": row.get("dataset", ""),
                "fold": row.get("fold", ""),
                "rule_source": row.get("rule_source", ""),
                "variant": row.get("variant", ""),
                "ppost_label": row.get("label", ""),
                "slice": slice_name,
                "slice_n": int(np.asarray(mask, dtype=bool).sum()),
                "slice_coverage": float(np.asarray(mask, dtype=bool).mean()),
            }
            for metric in METRICS:
                item[f"base_{metric}"] = float(base_metrics.get(metric, float("nan")))
                item[f"ppost_{metric}"] = float(pp_metrics.get(metric, float("nan")))
                item[f"delta_{metric}"] = item[f"ppost_{metric}"] - item[f"base_{metric}"]
            out.append(item)
    return out


def _control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not _is_ppost(row):
            continue
        loaded = _load_artifact(row)
        if loaded is None:
            continue
        y, p = loaded
        controls = {
            "ppost_observed": p,
            "control_permuted_patients": p[_stable_permutation(len(y), str(row.get("prediction_artifact", "")))],
            "control_class_prior_only": _prior_proba(y, p.shape[1]),
            "control_temperature_flattened_t4": _flatten_proba(p, 4.0),
        }
        observed = compute_metrics(y, p, p.shape[1])
        for name, p_ctrl in controls.items():
            metrics = compute_metrics(y, p_ctrl, p.shape[1])
            item = {
                "dataset": row.get("dataset", ""),
                "fold": row.get("fold", ""),
                "rule_source": row.get("rule_source", ""),
                "variant": row.get("variant", ""),
                "ppost_label": row.get("label", ""),
                "control": name,
            }
            for metric in METRICS:
                item[metric] = float(metrics.get(metric, float("nan")))
                item[f"delta_vs_observed_{metric}"] = float(metrics.get(metric, float("nan"))) - float(observed.get(metric, float("nan")))
            out.append(item)
    return out


def _compactness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not _is_ppost(row):
            continue
        n_branches = _float(row, "n_branches")
        top_k = _float(row, "top_k")
        item = {
            "dataset": row.get("dataset", ""),
            "fold": row.get("fold", ""),
            "rule_source": row.get("rule_source", ""),
            "variant": row.get("variant", ""),
            "ppost_label": row.get("label", ""),
            "n_branches": n_branches,
            "top_k": top_k,
            "trace_fraction": top_k / n_branches if n_branches and n_branches == n_branches else float("nan"),
            "mcc": _float(row, "mcc"),
            "brier_score": _float(row, "brier_score"),
            "ece_10": _float(row, "ece_10"),
        }
        out.append(item)
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) == float(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _summary_rows(
    pairwise: list[dict[str, Any]],
    selective: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    compactness: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted({(r["dataset"], r["rule_source"], r["variant"]) for r in pairwise})
    out = []
    for dataset, source, variant in keys:
        pr = [r for r in pairwise if (r["dataset"], r["rule_source"], r["variant"]) == (dataset, source, variant)]
        sr = [r for r in selective if (r["dataset"], r["rule_source"], r["variant"]) == (dataset, source, variant)]
        cr = [r for r in controls if (r["dataset"], r["rule_source"], r["variant"]) == (dataset, source, variant)]
        kr = [r for r in compactness if (r["dataset"], r["rule_source"], r["variant"]) == (dataset, source, variant)]
        hard = [r for r in sr if r.get("slice") in {"base_uncertain_top_tertile", "pp_large_correction_top_quintile", "base_pp_disagree"}]
        perm = [r for r in cr if r.get("control") == "control_permuted_patients"]
        obs = [r for r in cr if r.get("control") == "ppost_observed"]
        out.append({
            "dataset": dataset,
            "rule_source": source,
            "variant": variant,
            "folds": len(pr),
            "mean_delta_mcc": _mean(r.get("delta_mcc", float("nan")) for r in pr),
            "mean_delta_balanced_accuracy": _mean(r.get("delta_balanced_accuracy", float("nan")) for r in pr),
            "mean_delta_sensitivity": _mean(r.get("delta_sensitivity", float("nan")) for r in pr),
            "mean_delta_brier_score": _mean(r.get("delta_brier_score", float("nan")) for r in pr),
            "mean_delta_ece_10": _mean(r.get("delta_ece_10", float("nan")) for r in pr),
            "hard_slice_delta_mcc": _mean(r.get("delta_mcc", float("nan")) for r in hard),
            "hard_slice_delta_sensitivity": _mean(r.get("delta_sensitivity", float("nan")) for r in hard),
            "observed_mcc": _mean(r.get("mcc", float("nan")) for r in obs),
            "permuted_mcc": _mean(r.get("mcc", float("nan")) for r in perm),
            "observed_minus_permuted_mcc": _mean(r.get("mcc", float("nan")) for r in obs) - _mean(r.get("mcc", float("nan")) for r in perm),
            "mean_trace_fraction": _mean(r.get("trace_fraction", float("nan")) for r in kr),
        })
    return out


def build_proof_tables(out_dir: Path) -> None:
    csv_path = _latest_compare_csv(out_dir)
    if csv_path is None:
        print(f"[section46] no compare CSV in {out_dir}; proof tables skipped")
        return
    rows = _read_csv_rows(csv_path)
    pairwise = _pairwise_rows(rows)
    selective = _selective_rows(rows)
    controls = _control_rows(rows)
    compactness = _compactness_rows(rows)
    summary = _summary_rows(pairwise, selective, controls, compactness)
    _write_csv(out_dir / "ppost_proof_pairwise.csv", pairwise)
    _write_csv(out_dir / "ppost_proof_selective_slices.csv", selective)
    _write_csv(out_dir / "ppost_proof_controls.csv", controls)
    _write_csv(out_dir / "ppost_proof_compactness.csv", compactness)
    _write_csv(out_dir / "ppost_proof_summary.csv", summary)
    print(
        "[section46] proof tables written "
        f"pairwise={len(pairwise)} selective={len(selective)} controls={len(controls)} "
        f"compactness={len(compactness)} summary={len(summary)}"
    )


def _extract_output_dir(args: list[str], default_out: Path) -> Path:
    out = default_out
    for i, value in enumerate(args[:-1]):
        if value == "--output-dir":
            out = Path(args[i + 1])
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    p.add_argument("--skip-proof-tables", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = build_arg_parser()
    known, passthrough = parser.parse_known_args(argv)
    for key, value in EXPERIMENT_ENVS.get(known.experiment, {}).items():
        os.environ.setdefault(key, value)
    default_out = ROOT / "output" / "paper" / "46_ppost_proof" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    out_dir = _extract_output_dir(args, default_out)
    print(f"[section46] experiment={known.experiment} args={' '.join(args)}")
    rc = run_compare_datasets(args)
    if rc == 0 and not known.skip_proof_tables:
        build_proof_tables(out_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
