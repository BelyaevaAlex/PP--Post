#!/usr/bin/env python3
"""Paper Section 28: patient-level bootstrap and calibration from artifacts.

Consumes compare_datasets CSV rows that contain ``prediction_artifact`` paths
created by ``--save-predictions``. This script intentionally depends only on
NumPy and the Python standard library, so it can rerun patient-level bootstrap
and calibration analyses without importing the full training stack.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "28_prediction_artifact_metrics"
DEFAULT_METRICS = "accuracy,mcc,balanced_accuracy,f1_macro,roc_auc_ovr,auprc_ovr,log_loss,brier_score,ece_10,sensitivity,specificity,net_benefit_0_10,net_benefit_0_20"


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    proba = np.clip(proba, 0.0, None)
    denom = proba.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return proba / denom


def _confusion(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y.astype(int), pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1.0
    return cm


def _balanced_accuracy(cm: np.ndarray) -> float:
    recalls = []
    for k in range(cm.shape[0]):
        den = cm[k, :].sum()
        if den > 0:
            recalls.append(cm[k, k] / den)
    return float(np.mean(recalls)) if recalls else float("nan")


def _f1_macro(cm: np.ndarray) -> float:
    vals = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        den = 2 * tp + fp + fn
        vals.append(float(2 * tp / den) if den > 0 else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _mcc_multiclass(cm: np.ndarray) -> float:
    s = cm.sum()
    if s <= 0:
        return float("nan")
    c = np.trace(cm)
    t = cm.sum(axis=1)
    p = cm.sum(axis=0)
    numerator = c * s - np.dot(t, p)
    denom = math.sqrt(max((s * s - np.dot(p, p)) * (s * s - np.dot(t, t)), 0.0))
    return float(numerator / denom) if denom > 0 else 0.0


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))


def _binary_auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    y_bin = np.asarray(y_bin, dtype=int)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[y_bin == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision_binary(y_bin: np.ndarray, score: np.ndarray) -> float:
    y_bin = np.asarray(y_bin, dtype=int)
    n_pos = int(y_bin.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y_bin[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def _ovr_metric(y: np.ndarray, p: np.ndarray, fn) -> float:
    vals, weights = [], []
    for cls in range(p.shape[1]):
        target = (y == cls).astype(int)
        support = int(target.sum())
        if support == 0 or support == len(target):
            continue
        value = float(fn(target, p[:, cls]))
        if value == value:
            vals.append(value)
            weights.append(support)
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=np.asarray(weights, dtype=float)))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (pred == y).astype(float)
    out = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi) if hi == 1.0 else (conf >= lo) & (conf < hi)
        if np.any(mask):
            out += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(out)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    if p.shape[1] == 2:
        return float(np.mean((p[:, 1] - (y == 1).astype(float)) ** 2))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y.astype(int)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    if p.shape[1] != 2:
        return float("nan")
    pred_pos = p[:, 1] >= threshold
    y_pos = y == 1
    tp = float(np.logical_and(pred_pos, y_pos).sum())
    fp = float(np.logical_and(pred_pos, ~y_pos).sum())
    n = float(len(y))
    return float(tp / n - fp / n * threshold / (1.0 - threshold))


def compute_metrics(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict[str, float]:
    p = normalize_proba(p)
    pred = np.argmax(p, axis=1)
    cm = _confusion(y, pred, n_classes)
    out = {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": _balanced_accuracy(cm),
        "f1_macro": _f1_macro(cm),
        "mcc": _mcc_multiclass(cm),
        "log_loss": _log_loss(y, p),
        "auprc_ovr": _average_precision_binary((y == 1).astype(int), p[:, 1]) if n_classes == 2 else _ovr_metric(y, p, _average_precision_binary),
        "brier_score": _brier(y, p),
        "ece_10": _ece(y, p, 10),
        "roc_auc_ovr": _binary_auc((y == 1).astype(int), p[:, 1]) if n_classes == 2 else _ovr_metric(y, p, _binary_auc),
        "net_benefit_0_10": _net_benefit(y, p, 0.10),
        "net_benefit_0_20": _net_benefit(y, p, 0.20),
    }
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    else:
        out["sensitivity"] = float("nan")
        out["specificity"] = float("nan")
    return out


def _read_rows(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            with path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row = dict(row)
                    row["_source_csv"] = str(path)
                    rows.append(row)
    return rows


def _method_name(row: dict[str, Any]) -> str:
    return str(row.get("label") or f"{row.get('rule_source')}+{row.get('variant')}")


def _load_artifact(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    path = row.get("prediction_artifact")
    if not path:
        return None
    artifact = Path(str(path))
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if not artifact.exists():
        return None
    data = np.load(artifact, allow_pickle=False)
    return np.asarray(data["y_true"], dtype=int), normalize_proba(np.asarray(data["proba"], dtype=float))


def _bootstrap_metric(y: np.ndarray, p: np.ndarray, metric: str, n_bootstrap: int, seed: int, alpha: float) -> tuple[float, float, float]:
    n_classes = p.shape[1]
    base = compute_metrics(y, p, n_classes).get(metric, float("nan"))
    if len(y) == 0 or n_bootstrap <= 0:
        return float(base), float("nan"), float("nan")
    rng = random.Random(seed)
    vals: list[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = np.asarray([rng.randrange(n) for _ in range(n)], dtype=int)
        val = compute_metrics(y[idx], p[idx], n_classes).get(metric, float("nan"))
        if val == val:
            vals.append(float(val))
    vals.sort()
    if not vals:
        return float(base), float("nan"), float("nan")
    lo = vals[max(0, int((alpha / 2) * len(vals)))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return float(base), float(lo), float(hi)



def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int((seed + int.from_bytes(digest, "little")) % (2**32 - 1))


def _bootstrap_metrics_once(
    y: np.ndarray,
    p: np.ndarray,
    metrics: list[str],
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap all requested metrics in one pass over resamples."""
    n_classes = p.shape[1]
    base = compute_metrics(y, p, n_classes)
    if len(y) == 0 or n_bootstrap <= 0:
        return {metric: (float(base.get(metric, float("nan"))), float("nan"), float("nan")) for metric in metrics}

    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {metric: [] for metric in metrics}
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        boot = compute_metrics(y[idx], p[idx], n_classes)
        for metric in metrics:
            value = float(boot.get(metric, float("nan")))
            if value == value:
                vals[metric].append(value)

    out: dict[str, tuple[float, float, float]] = {}
    for metric in metrics:
        base_value = float(base.get(metric, float("nan")))
        metric_vals = sorted(vals[metric])
        if not metric_vals:
            out[metric] = (base_value, float("nan"), float("nan"))
            continue
        lo = metric_vals[max(0, int((alpha / 2) * len(metric_vals)))]
        hi = metric_vals[min(len(metric_vals) - 1, int((1 - alpha / 2) * len(metric_vals)))]
        out[metric] = (base_value, float(lo), float(hi))
    return out


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & (conf <= hi) if hi == 1 else (conf >= lo) & (conf < hi)
        if not np.any(mask):
            out.append({"bin": idx, "lo": lo, "hi": hi, "n": 0, "accuracy": "", "confidence": ""})
            continue
        out.append({"bin": idx, "lo": lo, "hi": hi, "n": int(mask.sum()), "accuracy": float(correct[mask].mean()), "confidence": float(conf[mask].mean())})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _read_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _artifact_path(row: dict[str, Any]) -> str:
    value = str(row.get("prediction_artifact", "") or "")
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--metrics", default=DEFAULT_METRICS)
    p.add_argument("--method-contains", default=None)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--resume", action="store_true", help="Reuse existing output rows and skip completed artifact/metric pairs.")
    p.add_argument("--checkpoint-every", type=int, default=1, help="Write output CSVs after this many newly computed artifacts.")
    p.add_argument("--max-artifacts", type=int, default=0, help="Optional batch cap for this invocation; 0 means all pending artifacts.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = _read_rows(args.csv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    out_dir = Path(args.output_dir)
    metric_csv = out_dir / "patient_bootstrap_metrics.csv"
    cal_csv = out_dir / "calibration_bins.csv"

    metric_rows: list[dict[str, Any]] = _read_existing_csv(metric_csv) if args.resume else []
    cal_rows: list[dict[str, Any]] = _read_existing_csv(cal_csv) if args.resume else []
    done_metrics = {
        (str(row.get("prediction_artifact", "")), str(row.get("metric", "")))
        for row in metric_rows
        if row.get("prediction_artifact") and row.get("metric")
    }

    artifact_rows: list[dict[str, Any]] = []
    seen_artifacts: set[str] = set()
    for row in rows:
        method = _method_name(row)
        if args.method_contains and args.method_contains not in method:
            continue
        artifact = _artifact_path(row)
        if not artifact or artifact in seen_artifacts:
            continue
        seen_artifacts.add(artifact)
        artifact_rows.append(row)

    total = len(artifact_rows)
    computed = 0
    skipped = 0
    for artifact_index, row in enumerate(artifact_rows, start=1):
        artifact = _artifact_path(row)
        pending_metrics = [metric for metric in metrics if (artifact, metric) not in done_metrics]
        if not pending_metrics:
            skipped += 1
            continue
        if args.max_artifacts and computed >= args.max_artifacts:
            break
        loaded = _load_artifact(row)
        if loaded is None:
            print(f"[skip missing] {artifact}", flush=True)
            continue
        y, p = loaded
        method = _method_name(row)
        print(
            f"[artifact {artifact_index}/{total}] dataset={row.get('dataset', '')} fold={row.get('fold', '')} "
            f"method={method} metrics={len(pending_metrics)} n={len(y)} boot={args.n_bootstrap}",
            flush=True,
        )
        boot = _bootstrap_metrics_once(
            y,
            p,
            pending_metrics,
            args.n_bootstrap,
            _stable_seed(args.seed, artifact),
            args.alpha,
        )
        for metric in pending_metrics:
            mean, lo, hi = boot[metric]
            metric_rows.append({
                "dataset": row.get("dataset", ""),
                "fold": row.get("fold", ""),
                "method": method,
                "metric": metric,
                "n_patients": int(len(y)),
                "value": mean,
                "ci_low": lo,
                "ci_high": hi,
                "prediction_artifact": row.get("prediction_artifact", ""),
            })
            done_metrics.add((artifact, metric))
        for bin_row in _calibration_bins(y, p):
            bin_row.update({
                "dataset": row.get("dataset", ""),
                "fold": row.get("fold", ""),
                "method": method,
                "prediction_artifact": row.get("prediction_artifact", ""),
            })
            cal_rows.append(bin_row)
        computed += 1
        if args.checkpoint_every > 0 and computed % args.checkpoint_every == 0:
            _write_csv(metric_csv, metric_rows)
            _write_csv(cal_csv, cal_rows)
            print(f"[checkpoint] computed={computed} skipped={skipped} metric_rows={len(metric_rows)}", flush=True)

    _write_csv(metric_csv, metric_rows)
    _write_csv(cal_csv, cal_rows)
    print(
        f"artifacts={len(set(r.get('prediction_artifact', '') for r in metric_rows))} "
        f"computed={computed} skipped={skipped} metric_rows={len(metric_rows)} output_dir={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
