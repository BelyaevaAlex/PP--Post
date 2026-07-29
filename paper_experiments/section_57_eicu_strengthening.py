#!/usr/bin/env python3
"""Section 57: eICU-focused PPtheta strengthening jobs.

This section targets the one dataset where the previously selected main row was
only an audit-signal boundary.  The jobs test whether eICU improves when the
source and operating point are chosen for eICU heterogeneity: RuleFit substrate,
clinical operating points, measurement-pattern evidence, measurement-policy
calibration proxy, and validation-pruned family aggregation.
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
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402
from paper_experiments.section_28_prediction_artifact_metrics import compute_metrics, normalize_proba  # noqa: E402

METRICS = ("mcc", "balanced_accuracy", "sensitivity", "specificity", "auprc_ovr", "roc_auc_ovr", "log_loss", "brier_score", "ece_10")


def _extract_option(args: list[str], option: str, default: str | None = None) -> str | None:
    for idx, value in enumerate(args[:-1]):
        if value == option:
            return args[idx + 1]
    return default


def _strip_options(args: list[str], options: set[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(args):
        if args[idx] in options:
            idx += 2
        else:
            out.append(args[idx])
            idx += 1
    return out


def _out_dir(args: list[str]) -> Path:
    return Path(_extract_option(args, "--output-dir", str(ROOT / "output/paper/57_eicu_strengthening")) or "")


def _dataset_path(args: list[str]) -> Path:
    raw = _extract_option(args, "--datasets", "") or ""
    if raw.startswith("npz:"):
        p = Path(raw[4:])
        return p if p.is_absolute() else ROOT / p
    fallback = ROOT / "data/processed/mortality/eicu_mortality_48h_tabular.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot infer eICU NPZ from --datasets={raw!r}")


def _compare_args(passthrough: list[str], out: Path, rule_sources: str, variants: str, baselines: str = "none") -> list[str]:
    args = _strip_options(passthrough, {"--output-dir", "--rule-sources", "--variants", "--baselines", "--save-predictions"})
    return args + [
        "--output-dir", str(out),
        "--rule-sources", rule_sources,
        "--variants", variants,
        "--baselines", baselines,
        "--save-predictions",
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_md(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _latest_compare_csv(path: Path) -> Path:
    csvs = sorted(p for p in path.glob("compare_datasets_*.csv") if not p.name.startswith("ppost_"))
    if not csvs:
        raise FileNotFoundError(f"No compare_datasets_*.csv in {path}")
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _summarize_pairwise_csv(csv_path: Path, out_prefix: Path) -> list[dict[str, Any]]:
    rows = _read_csv(csv_path)
    native: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("variant") == "source_native":
            native[(row.get("fold", ""), row.get("rule_source", ""))] = row
    per_fold: list[dict[str, Any]] = []
    for row in rows:
        if row.get("variant") == "source_native" or row.get("rule_source") == "baseline":
            continue
        base = native.get((row.get("fold", ""), row.get("rule_source", "")))
        if base is None:
            continue
        rec: dict[str, Any] = {
            "dataset": row.get("dataset", ""),
            "fold": row.get("fold", ""),
            "rule_source": row.get("rule_source", ""),
            "variant": row.get("variant", ""),
            "trace_fraction": _float(row.get("trace_fraction")),
        }
        for metric in METRICS:
            b = _float(base.get(metric)); p = _float(row.get(metric))
            rec[f"native_{metric}"] = b
            rec[f"ppost_{metric}"] = p
            rec[f"delta_{metric}"] = p - b if math.isfinite(p) and math.isfinite(b) else float("nan")
        per_fold.append(rec)
    summary: list[dict[str, Any]] = []
    keys = sorted({(r["rule_source"], r["variant"]) for r in per_fold})
    for source, variant in keys:
        part = [r for r in per_fold if r["rule_source"] == source and r["variant"] == variant]
        rec = {"rule_source": source, "variant": variant, "folds": len(part), "trace_fraction": _mean(r["trace_fraction"] for r in part)}
        for metric in METRICS:
            rec[f"native_{metric}"] = _mean(r[f"native_{metric}"] for r in part)
            rec[f"ppost_{metric}"] = _mean(r[f"ppost_{metric}"] for r in part)
            rec[f"delta_{metric}"] = _mean(r[f"delta_{metric}"] for r in part)
        summary.append(rec)
    _write_csv(out_prefix.with_suffix("_folds.csv") if False else Path(str(out_prefix) + "_folds.csv"), per_fold)
    _write_csv(Path(str(out_prefix) + "_summary.csv"), summary)
    return summary


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = normalize_proba(p)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {m: float("nan") for m in METRICS}
    return {m: float(compute_metrics(y, p, p.shape[1]).get(m, float("nan"))) for m in METRICS}


def run_rulefit_official(passthrough: list[str]) -> int:
    out = _out_dir(passthrough)
    variants = "source_native,pp_theta_post_ebm_residual_mcc,pp_theta_post_ebm_bounded_residual_gate,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk"
    rc = run_compare_datasets(_compare_args(passthrough, out, "rulefit", variants, "none"))
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "rulefit_official")
    _write_md(out / "rulefit_official.md", summary, ["rule_source", "variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score", "trace_fraction"])
    return 0


def run_operating_points(passthrough: list[str]) -> int:
    out = _out_dir(passthrough)
    variants = "source_native,pp_theta_post_operating_calibrated,pp_theta_post_operating_mcc,pp_theta_post_operating_sens90,pp_theta_post_operating_sens92,pp_theta_post_operating_sens95,pp_theta_post_ebm_residual_mcc"
    rc = run_compare_datasets(_compare_args(passthrough, out, "rulefit", variants, "none"))
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "operating_points")
    _write_md(out / "operating_points.md", summary, ["variant", "folds", "ppost_mcc", "delta_mcc", "ppost_sensitivity", "delta_sensitivity", "ppost_specificity", "delta_brier_score"])
    return 0


def _measurement_npz(source_npz: Path, out_dir: Path) -> Path:
    arr = np.load(source_npz, allow_pickle=True)
    X = np.asarray(arr["X"], dtype=np.float32)
    y = np.asarray(arr["y"], dtype=np.int64)
    names = np.asarray(arr["feature_names"]).astype(str)
    keep = np.array([name.endswith("__count") or name.endswith("__frac_obs") for name in names], dtype=bool)
    if not keep.any():
        keep = np.array(["count" in name or "frac" in name for name in names], dtype=bool)
    if not keep.any():
        raise ValueError("No measurement-pattern features found in eICU NPZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eicu_measurement_pattern_only.npz"
    np.savez_compressed(
        out_path,
        X=X[:, keep].astype(np.float32),
        y=y,
        feature_names=names[keep],
        class_names=np.asarray(arr["class_names"]).astype(str) if "class_names" in arr.files else np.asarray(["alive", "death"]),
        dataset_name=np.asarray("eicu_measurement_pattern_only"),
    )
    return out_path


def run_measurement_pattern_families(passthrough: list[str]) -> int:
    out = _out_dir(passthrough)
    source_npz = _dataset_path(passthrough)
    meas_npz = _measurement_npz(source_npz, out / "measurement_npz")
    args = _strip_options(passthrough, {"--datasets", "--output-dir", "--rule-sources", "--variants", "--baselines", "--save-predictions"})
    args += [
        "--datasets", f"npz:{meas_npz}",
        "--output-dir", str(out),
        "--rule-sources", "rulefit,xgb,extratrees",
        "--variants", "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_ebm_residual_mcc",
        "--baselines", "none",
        "--save-predictions",
    ]
    rc = run_compare_datasets(args)
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "measurement_pattern_families")
    _write_md(out / "measurement_pattern_families.md", summary, ["rule_source", "variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score"])
    return 0


def _artifact_path(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [ROOT / p, p]
    text = str(p)
    alias_spec = os.environ.get("PPPOST_ARTIFACT_ROOT_ALIASES", "")
    for item in alias_spec.split(os.pathsep):
        if not item or "=" not in item:
            continue
        src, dst = item.split("=", 1)
        if src and text.startswith(src):
            candidates.append(Path(dst + text[len(src):]))
    for c in candidates:
        if c.exists():
            return c
    return None


def _measurement_density(X: np.ndarray, names: np.ndarray) -> np.ndarray:
    frac_cols = np.array([str(n).endswith("__frac_obs") for n in names], dtype=bool)
    count_cols = np.array([str(n).endswith("__count") for n in names], dtype=bool)
    if frac_cols.any():
        return np.nanmean(X[:, frac_cols], axis=1)
    if count_cols.any():
        raw = np.nanmean(X[:, count_cols], axis=1)
        lo, hi = np.nanpercentile(raw, [1, 99])
        return np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.zeros(X.shape[0], dtype=float)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _splits(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds <= 1:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        return [(np.asarray(tr), np.asarray(te))]
    return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y))


def run_measurement_policy_calibration(passthrough: list[str]) -> int:
    out = _out_dir(passthrough)
    variants = "source_native,pp_theta_post_ebm_residual_mcc,pp_theta_post_rule_family_calibrated"
    rc = run_compare_datasets(_compare_args(passthrough, out, "rulefit", variants, "none"))
    if rc != 0:
        return rc
    csv_path = _latest_compare_csv(out)
    rows = _read_csv(csv_path)
    npz_path = _dataset_path(passthrough)
    arr = np.load(npz_path, allow_pickle=True)
    X = np.asarray(arr["X"], dtype=float)
    y = np.asarray(arr["y"], dtype=int)
    names = np.asarray(arr["feature_names"]).astype(str)
    density = _measurement_density(X, names)
    folds = int(_extract_option(passthrough, "--folds", "3") or 3)
    seed = int(os.environ.get("SEED", "42"))
    split_list = _splits(X, y, folds, seed)
    calibrated_rows = []
    for row in rows:
        if row.get("rule_source") != "rulefit" or row.get("variant") == "source_native":
            continue
        art = _artifact_path(row.get("prediction_artifact", ""))
        if art is None:
            continue
        fold = int(float(row.get("fold", "1")))
        train_idx, test_idx = split_list[fold - 1]
        data = np.load(art, allow_pickle=False)
        yt = np.asarray(data["y_true"], dtype=int)
        proba = normalize_proba(np.asarray(data["proba"], dtype=float))
        if len(yt) != len(test_idx):
            continue
        train_density = density[train_idx]
        test_density = density[test_idx]
        qs = np.quantile(train_density, [1/3, 2/3])
        train_group = np.digitize(train_density, qs, right=False)
        test_group = np.digitize(test_density, qs, right=False)
        global_rate = (float(np.mean(y[train_idx])) * len(train_idx) + 1.0) / (len(train_idx) + 2.0)
        offsets = []
        for g in range(3):
            mask = train_group == g
            n = int(mask.sum())
            rate = (float(np.sum(y[train_idx][mask])) + 1.0) / (n + 2.0) if n else global_rate
            shrink = n / (n + 400.0)
            offsets.append(shrink * float(_logit(np.array([rate]))[0] - _logit(np.array([global_rate]))[0]))
        p1 = proba[:, 1]
        adj = _sigmoid(_logit(p1) + np.asarray([offsets[int(g)] for g in test_group]))
        calibrated = np.column_stack([1.0 - adj, adj])
        base_m = _metrics(yt, proba)
        cal_m = _metrics(yt, calibrated)
        rec: dict[str, Any] = {"fold": fold, "rule_source": row.get("rule_source"), "variant": row.get("variant"), "calibration": "measurement_policy_train_fold_intercept"}
        for metric in METRICS:
            rec[f"base_{metric}"] = base_m[metric]
            rec[f"calibrated_{metric}"] = cal_m[metric]
            rec[f"delta_{metric}"] = cal_m[metric] - base_m[metric]
        rec["train_group_rates"] = ";".join(f"{x:.4f}" for x in offsets)
        calibrated_rows.append(rec)
    summary = []
    for variant in sorted({r["variant"] for r in calibrated_rows}):
        part = [r for r in calibrated_rows if r["variant"] == variant]
        rec = {"variant": variant, "folds": len(part), "calibration": "measurement-policy train-fold intercept"}
        for metric in METRICS:
            rec[f"base_{metric}"] = _mean(r[f"base_{metric}"] for r in part)
            rec[f"calibrated_{metric}"] = _mean(r[f"calibrated_{metric}"] for r in part)
            rec[f"delta_{metric}"] = _mean(r[f"delta_{metric}"] for r in part)
        summary.append(rec)
    _write_csv(out / "measurement_policy_calibration_folds.csv", calibrated_rows)
    _write_csv(out / "measurement_policy_calibration_summary.csv", summary)
    _write_md(out / "measurement_policy_calibration.md", summary, ["variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score", "delta_ece_10"])
    return 0


def run_family_pruning_sweep(passthrough: list[str]) -> int:
    out = _out_dir(passthrough)
    all_rows = []
    old = {k: os.environ.get(k) for k in ("PPPOST_FAMILY_UTILITY_TOPK", "PPPOST_EBM_RESIDUAL_TOPK")}
    try:
        for topk in (8, 16, 24, 32, 48, 64):
            os.environ["PPPOST_FAMILY_UTILITY_TOPK"] = str(topk)
            os.environ["PPPOST_EBM_RESIDUAL_TOPK"] = str(topk)
            sub = out / f"topk_{topk}"
            variants = "source_native,pp_theta_post_family_utility_pruned_topk,pp_theta_post_ebm_residual_mcc,pp_theta_post_rule_family_calibrated"
            rc = run_compare_datasets(_compare_args(passthrough, sub, "rulefit", variants, "none"))
            if rc != 0:
                return rc
            summary = _summarize_pairwise_csv(_latest_compare_csv(sub), sub / "family_pruning")
            for row in summary:
                row["topk"] = topk
                all_rows.append(row)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _write_csv(out / "family_pruning_sweep_summary.csv", all_rows)
    _write_md(out / "family_pruning_sweep.md", all_rows, ["topk", "variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score", "trace_fraction"])
    return 0


EXPERIMENTS = {
    "rulefit_official": run_rulefit_official,
    "operating_points": run_operating_points,
    "measurement_pattern_families": run_measurement_pattern_families,
    "measurement_policy_calibration": run_measurement_policy_calibration,
    "family_pruning_sweep": run_family_pruning_sweep,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    known, passthrough = parser.parse_known_args(argv)
    out = _out_dir(passthrough)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[section57] experiment={known.experiment} out={out}")
    return EXPERIMENTS[known.experiment](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
