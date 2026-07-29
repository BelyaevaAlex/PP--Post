#!/usr/bin/env python3
"""Paper Section 27: uncertainty intervals and non-inferiority checks.

Post-processes one or more compare_datasets CSV files. Without a comparator it
reports mean/std/bootstrap CI per method. With ``--method-contains`` and
``--comparator-contains`` it performs paired bootstrap over matched
(dataset, fold) keys and reports difference CIs plus a non-inferiority check.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "27_uncertainty_noninferiority"
DEFAULT_METRICS = "accuracy,mcc,balanced_accuracy,f1_macro,roc_auc_ovr,auprc_ovr,log_loss,brier_score,ece_10"


def _read_csvs(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = dict(row)
                    row["_source_csv"] = str(path)
                    rows.append(row)
    return rows


def _f(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except Exception:
        return float("nan")
    return value


def _method_name(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "").strip()
    if label:
        return label
    return f"{row.get('rule_source')}+{row.get('variant')}"


def _ci(values: list[float], n_bootstrap: int, seed: int, alpha: float) -> tuple[float, float]:
    values = [v for v in values if v == v]
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[max(0, int((alpha / 2) * len(means)))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return float(lo), float(hi)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], metrics: list[str], n_bootstrap: int, seed: int, alpha: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_method_name(row)].append(row)
    out: list[dict[str, Any]] = []
    for method, sub in sorted(grouped.items()):
        for metric in metrics:
            vals = [_f(r, metric) for r in sub]
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            lo, hi = _ci(vals, n_bootstrap, seed, alpha)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            out.append({
                "method": method,
                "metric": metric,
                "n": len(vals),
                "mean": mean,
                "std": math.sqrt(var),
                "ci_low": lo,
                "ci_high": hi,
            })
    return out


def paired(rows: list[dict[str, Any]], metrics: list[str], method_contains: str, comparator_contains: str, n_bootstrap: int, seed: int, alpha: float, ni_margin: float) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"method": [], "comp": []})
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("fold")))
        name = _method_name(row)
        if method_contains in name:
            by_key[key]["method"].append(row)
        if comparator_contains in name:
            by_key[key]["comp"].append(row)
    out: list[dict[str, Any]] = []
    for metric in metrics:
        diffs = []
        for key, pair in by_key.items():
            if not pair["method"] or not pair["comp"]:
                continue
            m = max(pair["method"], key=lambda r: _f(r, metric))
            c = max(pair["comp"], key=lambda r: _f(r, metric))
            diff = _f(m, metric) - _f(c, metric)
            if diff == diff:
                diffs.append(diff)
        if not diffs:
            continue
        lo, hi = _ci(diffs, n_bootstrap, seed, alpha)
        mean = sum(diffs) / len(diffs)
        out.append({
            "method_contains": method_contains,
            "comparator_contains": comparator_contains,
            "metric": metric,
            "n_pairs": len(diffs),
            "mean_diff": mean,
            "ci_low": lo,
            "ci_high": hi,
            "noninferiority_margin": ni_margin,
            "noninferior_by_ci": int(lo > -abs(ni_margin)) if metric in {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc_ovr", "auprc_ovr"} else "",
        })
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", nargs="+", required=True, help="CSV path(s) or glob(s)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--metrics", default=DEFAULT_METRICS)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--method-contains", default=None)
    p.add_argument("--comparator-contains", default=None)
    p.add_argument("--noninferiority-margin", type=float, default=0.01)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = _read_csvs(args.csv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    out_dir = Path(args.output_dir)
    summary = summarize(rows, metrics, args.n_bootstrap, args.seed, args.alpha)
    _write_csv(out_dir / "method_metric_ci.csv", summary)
    if args.method_contains and args.comparator_contains:
        paired_rows = paired(rows, metrics, args.method_contains, args.comparator_contains, args.n_bootstrap, args.seed, args.alpha, args.noninferiority_margin)
        _write_csv(out_dir / "paired_noninferiority_ci.csv", paired_rows)
    print(f"rows={len(rows)} output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
