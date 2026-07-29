#!/usr/bin/env python3
"""Section 51: reviewer-facing PPtheta evidence experiments, v2.

This section strengthens the empirical claim that PPtheta-Post is useful as a
prediction-sufficient posterior audit layer.  Each experiment writes ordinary
CSV/Markdown artifacts and can be launched as an independent cluster stage.
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
from paper_experiments.section_28_prediction_artifact_metrics import compute_metrics, normalize_proba  # noqa: E402

PROOF_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_ppost_proof_mortality_ppost_proof_local_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"

SELECTED = {
    "eicu": {
        "dataset_label": "eICU",
        "source": "tabpfn_distill_xgb_soft",
        "variant": "pp_theta_post_ebm_bounded_residual_gate",
        "stage": "rahmatullaev_proof_selective_utility",
    },
    "mimic3": {
        "dataset_label": "MIMIC-III",
        "source": "xgb",
        "variant": "pp_theta_post_rule_family_calibrated",
        "stage": "rahmatullaev_proof_evidence_ablation",
    },
    "mimic4": {
        "dataset_label": "MIMIC-IV",
        "source": "tabpfn_distill_xgb_soft",
        "variant": "pp_theta_post_ebm_residual_mcc",
        "stage": "rahmatullaev_proof_strong_base_repair",
    },
}

METRICS = ("mcc", "balanced_accuracy", "sensitivity", "specificity", "auprc_ovr", "roc_auc_ovr", "log_loss", "brier_score", "ece_10")
BOOT_METRICS = ("mcc", "balanced_accuracy", "sensitivity", "specificity", "log_loss", "brier_score", "ece_10")
TRACE_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)


def _read_csv(path: Path) -> list[dict[str, str]]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append("nan" if not math.isfinite(val) else f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _ci(values: Iterable[float]) -> tuple[float, float]:
    vals = np.array([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


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


def _dataset_key(args: list[str]) -> str:
    raw = (_extract_option(args, "--datasets", "") or "").lower()
    if "mimic4" in raw:
        return "mimic4"
    if "mimic3" in raw:
        return "mimic3"
    if "eicu" in raw:
        return "eicu"
    return "eicu"


def _out_dir(args: list[str]) -> Path:
    return Path(_extract_option(args, "--output-dir", str(ROOT / "output/paper/51_aaai_evidence_v2")) or "")


def _latest_compare_csv(path: Path) -> Path:
    csvs = sorted(p for p in path.glob("compare_datasets_*.csv") if not p.name.startswith("ppost_"))
    if not csvs:
        raise FileNotFoundError(f"No compare_datasets_*.csv in {path}")
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _artifact_path(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [p, ROOT / p]
    text = str(p)
    alias_spec = os.environ.get("PPPOST_ARTIFACT_ROOT_ALIASES", "")
    for item in alias_spec.split(os.pathsep):
        if not item or "=" not in item:
            continue
        src, dst = item.split("=", 1)
        if src and text.startswith(src):
            candidates.append(Path(dst + text[len(src):]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_prediction(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
    path = _artifact_path(row.get("prediction_artifact", ""))
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["y_true"], dtype=int), normalize_proba(np.asarray(data["proba"], dtype=float))


def _selected_rows(dataset: str) -> list[dict[str, str]]:
    selected = SELECTED[dataset]
    return _read_csv(_latest_compare_csv(PROOF_ROOT / dataset / selected["stage"]))


def _selected_pairs(dataset: str) -> list[tuple[dict[str, str], dict[str, str]]]:
    selected = SELECTED[dataset]
    rows = _selected_rows(dataset)
    native = {
        r.get("fold", ""): r for r in rows
        if r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"
    }
    pairs = []
    for row in rows:
        if row.get("rule_source") == selected["source"] and row.get("variant") == selected["variant"]:
            base = native.get(row.get("fold", ""))
            if base is not None:
                pairs.append((base, row))
    if not pairs:
        raise FileNotFoundError(f"No selected native/+PPtheta pairs for {dataset}")
    return pairs


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {m: float("nan") for m in METRICS}
    return {m: float(compute_metrics(y, p, p.shape[1]).get(m, float("nan"))) for m in METRICS}


def _fast_binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = normalize_proba(np.asarray(p, dtype=float))
    if len(y) == 0:
        return {m: float("nan") for m in BOOT_METRICS}
    p1 = p[:, 1] if p.shape[1] > 1 else p[:, 0]
    pred = np.argmax(p, axis=1)
    tp = float(np.sum((pred == 1) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    eps = 1e-12
    log_loss = -float(np.mean(y * np.log(np.clip(p1, eps, 1.0)) + (1 - y) * np.log(np.clip(1.0 - p1, eps, 1.0))))
    brier = float(np.mean((p1 - y.astype(float)) ** 2))
    # Fixed-bin ECE proxy matching the paper's ece_10 naming closely enough for bootstrap intervals.
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p1 >= lo) & (p1 < hi if hi < 1.0 else p1 <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(np.mean(p1[mask])) - float(np.mean(y[mask])))
    return {
        "mcc": float(mcc),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "log_loss": log_loss,
        "brier_score": brier,
        "ece_10": float(ece),
    }


def _mask_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 3 or len(np.unique(y[mask])) < 2:
        return {m: float("nan") for m in METRICS}
    return _metrics(y[mask], p[mask])


def _stable_rng(text: str) -> np.random.Generator:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little") % (2**32 - 1))


def _prior_proba(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    prior = counts / max(float(counts.sum()), 1.0)
    return np.repeat(prior.reshape(1, -1), len(y), axis=0)


def _flatten(p: np.ndarray, temp: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-12, 1.0)) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    return normalize_proba(np.exp(logits))


def _sharpen(p: np.ndarray, temp: float = 0.5) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-12, 1.0)) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    return normalize_proba(np.exp(logits))


def _compare_args(passthrough: list[str], out: Path, sources: str, variants: str, baselines: str = "ebm,tabpfn") -> list[str]:
    stripped = _strip_options(
        passthrough,
        {"--output-dir", "--rule-sources", "--variants", "--baselines", "--rule-selection", "--rule-budget", "--top-k-ratio", "--top-k-min", "--top-k-max", "--sparse-logit-top-k"},
    )
    return [
        *stripped,
        "--output-dir", str(out),
        "--rule-sources", sources,
        "--variants", variants,
        "--baselines", baselines,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--top-k-min", "1",
        "--top-k-max", "100000",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ]


def _summarize_pairwise_csv(csv_path: Path, out_prefix: Path) -> list[dict[str, Any]]:
    rows = _read_csv(csv_path)
    native = {(r.get("fold", ""), r.get("rule_source", "")): r for r in rows if r.get("variant") == "source_native"}
    fold_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("variant", "").startswith("pp_theta"):
            continue
        base = native.get((row.get("fold", ""), row.get("rule_source", "")))
        if base is None:
            continue
        item: dict[str, Any] = {"dataset": row.get("dataset", ""), "fold": row.get("fold", ""), "rule_source": row.get("rule_source", ""), "variant": row.get("variant", "")}
        for metric in METRICS:
            item[f"native_{metric}"] = _float(base.get(metric))
            item[f"ppost_{metric}"] = _float(row.get(metric))
            item[f"delta_{metric}"] = item[f"ppost_{metric}"] - item[f"native_{metric}"]
        item["n_branches"] = _float(row.get("n_branches"))
        item["top_k"] = _float(row.get("top_k"))
        item["trace_fraction"] = item["top_k"] / item["n_branches"] if item["n_branches"] > 0 else float("nan")
        fold_rows.append(item)
    summary: list[dict[str, Any]] = []
    for key in sorted({(r["rule_source"], r["variant"]) for r in fold_rows}):
        source, variant = key
        rr = [r for r in fold_rows if (r["rule_source"], r["variant"]) == key]
        row = {"rule_source": source, "variant": variant, "folds": len(rr), "trace_fraction": _mean(r["trace_fraction"] for r in rr)}
        for metric in METRICS:
            row[f"native_{metric}"] = _mean(r[f"native_{metric}"] for r in rr)
            row[f"ppost_{metric}"] = _mean(r[f"ppost_{metric}"] for r in rr)
            row[f"delta_{metric}"] = _mean(r[f"delta_{metric}"] for r in rr)
        summary.append(row)
    _write_csv(out_prefix.with_name(out_prefix.name + "_folds.csv"), fold_rows)
    _write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary)
    return summary


def run_paired_utility_ci(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rng = np.random.default_rng(int(os.environ.get("PPPOST_BOOTSTRAP_SEED", "2027")))
    n_boot = int(os.environ.get("PPPOST_BOOTSTRAP_N", "600"))
    y_all: list[np.ndarray] = []
    base_all: list[np.ndarray] = []
    pp_all: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for base, pp in _selected_pairs(dataset):
        lb, lp = _load_prediction(base), _load_prediction(pp)
        if lb is None or lp is None:
            continue
        y, pb = lb
        y2, ppred = lp
        if not np.array_equal(y, y2):
            continue
        mb, mp = _metrics(y, pb), _metrics(y, ppred)
        row = {"dataset": dataset, "fold": pp.get("fold", ""), "source": SELECTED[dataset]["source"], "variant": SELECTED[dataset]["variant"], "n": len(y)}
        for metric in METRICS:
            row[f"native_{metric}"] = mb[metric]
            row[f"ppost_{metric}"] = mp[metric]
            row[f"delta_{metric}"] = mp[metric] - mb[metric]
        fold_rows.append(row)
        y_all.append(y); base_all.append(pb); pp_all.append(ppred)
    y = np.concatenate(y_all); pb = np.vstack(base_all); pp = np.vstack(pp_all)
    obs_b, obs_p = _metrics(y, pb), _metrics(y, pp)
    summary = {"dataset": dataset, "folds": len(fold_rows), "n": len(y)}
    boot: dict[str, list[float]] = {m: [] for m in BOOT_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        mb, mp = _fast_binary_metrics(y[idx], pb[idx]), _fast_binary_metrics(y[idx], pp[idx])
        for metric in BOOT_METRICS:
            boot[metric].append(mp[metric] - mb[metric])
    for metric in METRICS:
        deltas = [r[f"delta_{metric}"] for r in fold_rows if math.isfinite(r[f"delta_{metric}"])]
        summary[f"native_{metric}"] = obs_b[metric]
        summary[f"ppost_{metric}"] = obs_p[metric]
        summary[f"delta_{metric}"] = obs_p[metric] - obs_b[metric]
        if metric in BOOT_METRICS:
            lo, hi = _ci(boot[metric])
            summary[f"delta_{metric}_ci_low"] = lo
            summary[f"delta_{metric}_ci_high"] = hi
        summary[f"fold_win_rate_{metric}"] = float(np.mean(np.array(deltas) > 0)) if deltas else float("nan")
    _write_csv(out / "paired_utility_folds.csv", fold_rows)
    _write_csv(out / "paired_utility_ci.csv", [summary])
    _write_md(out / "paired_utility_ci.md", [summary], ["dataset", "folds", "n", "delta_mcc", "delta_mcc_ci_low", "delta_mcc_ci_high", "fold_win_rate_mcc", "delta_sensitivity", "delta_brier_score"])
    return 0


def run_rich_randomized_controls(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows_out: list[dict[str, Any]] = []
    for _base, pp in _selected_pairs(dataset):
        loaded = _load_prediction(pp)
        if loaded is None:
            continue
        y, p = loaded
        rng = _stable_rng(pp.get("prediction_artifact", "") + pp.get("fold", ""))
        controls = {
            "observed": p,
            "patient_permuted": p[rng.permutation(len(y))],
            "class_prior_only": _prior_proba(y, p.shape[1]),
            "temperature_flattened_t2": _flatten(p, 2.0),
            "temperature_flattened_t4": _flatten(p, 4.0),
            "temperature_flattened_t8": _flatten(p, 8.0),
            "overconfident_same_rank_t0p5": _sharpen(p, 0.5),
            "column_shuffled_class_scores": normalize_proba(p[:, rng.permutation(p.shape[1])]),
        }
        obs = _metrics(y, p)
        for name, pc in controls.items():
            mc = _metrics(y, pc)
            row = {"dataset": dataset, "fold": pp.get("fold", ""), "control": name, "source": SELECTED[dataset]["source"], "variant": SELECTED[dataset]["variant"], "n": len(y)}
            for metric in METRICS:
                row[metric] = mc[metric]
                row[f"delta_vs_observed_{metric}"] = mc[metric] - obs[metric]
            rows_out.append(row)
    summary: list[dict[str, Any]] = []
    for control in sorted({r["control"] for r in rows_out}):
        rr = [r for r in rows_out if r["control"] == control]
        row = {"dataset": dataset, "control": control, "folds": len(rr)}
        for metric in METRICS:
            row[metric] = _mean(r[metric] for r in rr)
            row[f"delta_vs_observed_{metric}"] = _mean(r[f"delta_vs_observed_{metric}"] for r in rr)
        summary.append(row)
    _write_csv(out / "rich_randomized_controls_folds.csv", rows_out)
    _write_csv(out / "rich_randomized_controls_summary.csv", summary)
    _write_md(out / "rich_randomized_controls.md", summary, ["dataset", "control", "mcc", "delta_vs_observed_mcc", "sensitivity", "delta_vs_observed_sensitivity", "brier_score"])
    return 0


def run_source_compatibility_matrix(passthrough: list[str]) -> int:
    out = _out_dir(passthrough)
    args = _compare_args(
        passthrough,
        out,
        "xgb,extratrees,ebm_terms,rulefit,figs,tabpfn_distill_xgb_soft,tabpfn_distill_ebm_terms",
        "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_bayes_llr_posneg,pp_theta_post_ebm_residual_mcc",
    )
    rc = run_compare_datasets(args)
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "source_compatibility")
    _write_md(out / "source_compatibility_matrix.md", summary, ["rule_source", "variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score", "trace_fraction"])
    return 0


def run_extended_trace_curve(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    selected = SELECTED[dataset]
    out = _out_dir(passthrough)
    all_summaries: list[dict[str, Any]] = []
    old_env = {k: os.environ.get(k) for k in ("PPPOST_RULE_FAMILY_TOPK", "PPPOST_FAMILY_UTILITY_TOPK", "PPPOST_EBM_RESIDUAL_TOPK", "PPPOST_BAYES_LLR_TOPK")}
    try:
        for fraction in TRACE_FRACTIONS:
            topk = max(1, int(round(384 * fraction)))
            for key in old_env:
                os.environ[key] = str(topk)
            frac_dir = out / f"fraction_{fraction:.3f}".replace(".", "p")
            rc = run_compare_datasets(_compare_args(passthrough, frac_dir, selected["source"], "source_native," + selected["variant"], "none") + ["--top-k-ratio", str(fraction), "--sparse-logit-top-k", str(topk)])
            if rc != 0:
                return rc
            summary = _summarize_pairwise_csv(_latest_compare_csv(frac_dir), frac_dir / "trace_curve")
            for row in summary:
                row["requested_fraction"] = fraction
                row["requested_topk"] = topk
                row["dataset_key"] = dataset
                all_summaries.append(row)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    _write_csv(out / "extended_trace_curve_summary.csv", all_summaries)
    _write_md(out / "extended_trace_curve.md", all_summaries, ["dataset_key", "requested_fraction", "trace_fraction", "variant", "ppost_mcc", "delta_mcc", "delta_sensitivity", "delta_brier_score"])
    return 0


def run_native_wrong_correction(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows_out: list[dict[str, Any]] = []
    for base, pp in _selected_pairs(dataset):
        lb, lp = _load_prediction(base), _load_prediction(pp)
        if lb is None or lp is None:
            continue
        y, pb = lb; y2, ppred = lp
        if not np.array_equal(y, y2):
            continue
        base_pred = np.argmax(pb, axis=1); pp_pred = np.argmax(ppred, axis=1)
        native_wrong = base_pred != y
        native_wrong_ppost_right = native_wrong & (pp_pred == y)
        native_wrong_positive = native_wrong & (y == 1)
        native_wrong_positive_ppost_right = native_wrong_positive & (pp_pred == y)
        correction_rate = float(native_wrong_ppost_right.sum() / max(native_wrong.sum(), 1))
        positive_correction_rate = float(native_wrong_positive_ppost_right.sum() / max(native_wrong_positive.sum(), 1))
        base_unc = 1.0 - np.abs(2.0 * pb[:, 1] - 1.0) if pb.shape[1] == 2 else 1.0 - np.max(pb, axis=1)
        shift = ppred[:, 1] - pb[:, 1] if pb.shape[1] == 2 else np.max(np.abs(ppred - pb), axis=1)
        masks = {
            "all": np.ones(len(y), dtype=bool),
            "mortality_positive": y == 1,
            "native_wrong": native_wrong,
            "native_wrong_ppost_right": native_wrong_ppost_right,
            "native_wrong_mortality_positive": native_wrong_positive,
            "native_wrong_positive_ppost_right": native_wrong_positive_ppost_right,
            "native_uncertain_top20": base_unc >= np.quantile(base_unc, 0.80),
            "large_ppost_shift_top20": np.abs(shift) >= np.quantile(np.abs(shift), 0.80),
        }
        for subset, mask in masks.items():
            mb, mp = _mask_metrics(y, pb, mask), _mask_metrics(y, ppred, mask)
            row = {
                "dataset": dataset,
                "fold": pp.get("fold", ""),
                "subset": subset,
                "n": int(mask.sum()),
                "coverage": float(mask.mean()),
                "native_wrong_count": int(native_wrong.sum()),
                "native_wrong_ppost_right_count": int(native_wrong_ppost_right.sum()),
                "native_wrong_positive_count": int(native_wrong_positive.sum()),
                "native_wrong_positive_ppost_right_count": int(native_wrong_positive_ppost_right.sum()),
                "correction_rate": correction_rate,
                "positive_correction_rate": positive_correction_rate,
            }
            for metric in METRICS:
                row[f"native_{metric}"] = mb[metric]
                row[f"ppost_{metric}"] = mp[metric]
                row[f"delta_{metric}"] = mp[metric] - mb[metric]
            row["mean_risk_shift"] = float(np.mean(shift[mask])) if np.any(mask) else float("nan")
            rows_out.append(row)
    summary: list[dict[str, Any]] = []
    for subset in sorted({r["subset"] for r in rows_out}):
        rr = [r for r in rows_out if r["subset"] == subset]
        total_wrong = sum(int(r.get("native_wrong_count", 0)) for r in rr)
        total_fixed = sum(int(r.get("native_wrong_ppost_right_count", 0)) for r in rr)
        total_pos_wrong = sum(int(r.get("native_wrong_positive_count", 0)) for r in rr)
        total_pos_fixed = sum(int(r.get("native_wrong_positive_ppost_right_count", 0)) for r in rr)
        row = {
            "dataset": dataset,
            "subset": subset,
            "folds": len(rr),
            "mean_n": _mean(r["n"] for r in rr),
            "coverage": _mean(r["coverage"] for r in rr),
            "mean_risk_shift": _mean(r["mean_risk_shift"] for r in rr),
            "native_wrong_count": total_wrong,
            "native_wrong_ppost_right_count": total_fixed,
            "native_wrong_positive_count": total_pos_wrong,
            "native_wrong_positive_ppost_right_count": total_pos_fixed,
            "correction_rate": float(total_fixed / max(total_wrong, 1)),
            "positive_correction_rate": float(total_pos_fixed / max(total_pos_wrong, 1)),
        }
        for metric in METRICS:
            row[f"native_{metric}"] = _mean(r[f"native_{metric}"] for r in rr)
            row[f"ppost_{metric}"] = _mean(r[f"ppost_{metric}"] for r in rr)
            row[f"delta_{metric}"] = _mean(r[f"delta_{metric}"] for r in rr)
        summary.append(row)
    _write_csv(out / "native_wrong_correction_folds.csv", rows_out)
    _write_csv(out / "native_wrong_correction_summary.csv", summary)
    _write_md(out / "native_wrong_correction.md", summary, ["dataset", "subset", "mean_n", "delta_mcc", "delta_sensitivity", "delta_brier_score", "correction_rate", "positive_correction_rate", "mean_risk_shift"])
    return 0


def run_operating_point_separation(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    selected = SELECTED[dataset]
    out = _out_dir(passthrough)
    variants = "source_native,pp_theta_post_operating_calibrated,pp_theta_post_operating_mcc,pp_theta_post_operating_sens90,pp_theta_post_operating_sens92,pp_theta_post_operating_sens95,pp_theta_post_teacher_anchored,pp_theta_post_teacher_calibrated"
    rc = run_compare_datasets(_compare_args(passthrough, out, selected["source"], variants, "ebm,tabpfn"))
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "operating_point_separation")
    _write_md(out / "operating_point_separation.md", summary, ["variant", "folds", "ppost_mcc", "delta_mcc", "ppost_sensitivity", "delta_sensitivity", "ppost_brier_score", "delta_brier_score", "ppost_ece_10"])
    return 0


def run_case_trace_candidates(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    rows_out: list[dict[str, Any]] = []
    for base, pp in _selected_pairs(dataset):
        lb, lp = _load_prediction(base), _load_prediction(pp)
        if lb is None or lp is None:
            continue
        y, pb = lb; y2, ppred = lp
        if not np.array_equal(y, y2):
            continue
        base_pred = np.argmax(pb, axis=1); pp_pred = np.argmax(ppred, axis=1)
        shift = ppred[:, 1] - pb[:, 1] if pb.shape[1] == 2 else np.max(np.abs(ppred - pb), axis=1)
        priority = 3.0 * ((base_pred != y) & (pp_pred == y)).astype(float) + 2.0 * (y == 1).astype(float) + np.abs(shift)
        for rank, idx in enumerate(np.argsort(-priority)[:30], start=1):
            rows_out.append({
                "dataset": dataset,
                "fold": pp.get("fold", ""),
                "rank": rank,
                "patient_index_in_fold": int(idx),
                "y_true": int(y[idx]),
                "native_pred": int(base_pred[idx]),
                "ppost_pred": int(pp_pred[idx]),
                "native_p_mortality": float(pb[idx, 1]) if pb.shape[1] == 2 else float(np.max(pb[idx])),
                "ppost_p_mortality": float(ppred[idx, 1]) if ppred.shape[1] == 2 else float(np.max(ppred[idx])),
                "ppost_minus_native": float(shift[idx]),
                "native_wrong_ppost_right": int((base_pred[idx] != y[idx]) and (pp_pred[idx] == y[idx])),
                "mortality_positive": int(y[idx] == 1),
                "source": SELECTED[dataset]["source"],
                "variant": SELECTED[dataset]["variant"],
                "prediction_artifact": pp.get("prediction_artifact", ""),
            })
    _write_csv(out / "case_trace_candidates.csv", rows_out)
    _write_md(out / "case_trace_candidates.md", rows_out[:20], ["dataset", "fold", "rank", "y_true", "native_pred", "ppost_pred", "native_p_mortality", "ppost_p_mortality", "ppost_minus_native", "native_wrong_ppost_right"])
    return 0


def run_external_tabular_sanity(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    if dataset != "eicu":
        _write_csv(out / "external_tabular_sanity_skip.csv", [{"dataset_axis": dataset, "status": "skipped", "reason": "external sanity check is dataset-independent and runs on the eicu axis only"}])
        return 0
    stripped = _strip_options(passthrough, {"--datasets", "--output-dir", "--rule-sources", "--variants", "--baselines", "--folds"})
    args = [
        *stripped,
        "--datasets", "sklearn:breast_cancer",
        "--folds", "5",
        "--output-dir", str(out),
        "--rule-sources", "xgb,extratrees,ebm_terms,rulefit,figs",
        "--variants", "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_bayes_llr_posneg",
        "--baselines", "ebm,figs,rulefit",
        "--rule-selection", "diverse",
        "--rule-budget", "256",
        "--save-predictions",
    ]
    rc = run_compare_datasets(args)
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "external_tabular_sanity")
    _write_md(out / "external_tabular_sanity.md", summary, ["rule_source", "variant", "folds", "delta_mcc", "delta_sensitivity", "delta_brier_score", "trace_fraction"])
    return 0


def run_component_ablation(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    selected = SELECTED[dataset]
    out = _out_dir(passthrough)
    variants = "source_native,pp_theta_post_frozen,pp_theta_post_shrink_theta,pp_theta_post_signed_logit,pp_theta_post_sparse_logit,pp_theta_post_rule_family,pp_theta_post_rule_family_calibrated,pp_theta_post_family_utility_pruned_topk,pp_theta_post_bayes_llr,pp_theta_post_bayes_llr_beta,pp_theta_post_bayes_llr_posneg"
    rc = run_compare_datasets(_compare_args(passthrough, out, selected["source"], variants, "none"))
    if rc != 0:
        return rc
    summary = _summarize_pairwise_csv(_latest_compare_csv(out), out / "component_ablation")
    _write_md(out / "component_ablation.md", summary, ["variant", "folds", "ppost_mcc", "delta_mcc", "delta_sensitivity", "delta_brier_score", "trace_fraction"])
    return 0


def run_statistical_summary(passthrough: list[str]) -> int:
    dataset = _dataset_key(passthrough)
    out = _out_dir(passthrough)
    base_root = out.parent
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(base_root.glob("*/**/*summary*.csv")):
        if csv_path.name.startswith("compare_datasets"):
            continue
        try:
            data = _read_csv(csv_path)
        except Exception:
            continue
        rows.append({"dataset": dataset, "artifact": str(csv_path), "rows": len(data), "columns": ",".join(data[0].keys()) if data else ""})
    _write_csv(out / "statistical_summary_manifest.csv", rows)
    _write_md(out / "statistical_summary_manifest.md", rows, ["dataset", "rows", "artifact"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=(
        "paired_utility_ci",
        "rich_randomized_controls",
        "source_compatibility_matrix",
        "extended_trace_curve",
        "native_wrong_correction",
        "operating_point_separation",
        "case_trace_candidates",
        "external_tabular_sanity",
        "component_ablation",
        "statistical_summary",
    ))
    known, passthrough = parser.parse_known_args(argv)
    return {
        "paired_utility_ci": run_paired_utility_ci,
        "rich_randomized_controls": run_rich_randomized_controls,
        "source_compatibility_matrix": run_source_compatibility_matrix,
        "extended_trace_curve": run_extended_trace_curve,
        "native_wrong_correction": run_native_wrong_correction,
        "operating_point_separation": run_operating_point_separation,
        "case_trace_candidates": run_case_trace_candidates,
        "external_tabular_sanity": run_external_tabular_sanity,
        "component_ablation": run_component_ablation,
        "statistical_summary": run_statistical_summary,
    }[known.experiment](passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
