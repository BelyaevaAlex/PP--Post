#!/usr/bin/env python3
"""Aggregate Rahmatullaev improvement sweep into mortality comparison tables.

The script scans compare_datasets CSV files under output/mortality_paper_jobs,
keeps complete 3-fold tabular runs, and writes compact leaderboards comparing
new rule-source/support/aggregation/calibration experiments with previous best
mortality tabular results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = ROOT / "output" / "mortality_paper_jobs"
NEW_SWEEP = "rahmatullaev_improvements_mortality_rule_improve_v1"
DATASET_MAP = {
    "mimic3_mortality_48h_tabular": "mimic3",
    "mimic4_mortality_48h_tabular": "mimic4",
    "eicu_mortality_48h_tabular": "eicu",
}
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "mcc",
    "cohen_kappa",
    "log_loss",
    "auprc_ovr",
    "brier_score",
    "ece_10",
    "ece_20",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "net_benefit_0_10",
    "net_benefit_0_20",
    "roc_auc_ovr",
]
LOWER_IS_BETTER = {"log_loss", "brier_score", "ece_10", "ece_20"}


def _csv_paths(root: Path) -> Iterable[Path]:
    yield from root.glob("*/**/compare_datasets*.csv")


def _path_meta(path: Path, root: Path) -> dict[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    sweep = parts[0] if len(parts) > 0 else ""
    dataset_axis = parts[1] if len(parts) > 1 else ""
    stage = parts[2] if len(parts) > 2 else ""
    return {
        "sweep": sweep,
        "dataset_axis": dataset_axis,
        "stage": stage,
        "csv_path": str(path),
    }


def load_rows(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(_csv_paths(root)):
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] unreadable {path}: {exc}")
            continue
        if df.empty or "dataset" not in df or "variant" not in df or "fold" not in df:
            continue
        meta = _path_meta(path, root)
        for k, v in meta.items():
            df[k] = v
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no compare_datasets CSV files found under {root}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["dataset_short"] = out["dataset"].map(DATASET_MAP).fillna(out["dataset_axis"])
    out["is_new"] = out["sweep"].eq(NEW_SWEEP)
    out["model_key"] = (
        out["label"].fillna(out["variant"].astype(str))
        + " | "
        + out["stage"].astype(str)
        + " | "
        + out["rule_source"].fillna("").astype(str)
    )
    for col in ["fold", "n_test", "n_branches", *METRICS]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize_runs(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset_short",
        "sweep",
        "stage",
        "csv_path",
        "is_new",
        "rule_source",
        "variant",
        "label",
    ]
    metric_cols = [m for m in METRICS if m in rows.columns]
    agg_spec = {m: ["mean", "std"] for m in metric_cols}
    agg_spec.update({"fold": "nunique", "n_test": "sum", "n_branches": "mean"})
    summary = rows.groupby(group_cols, dropna=False).agg(agg_spec)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.reset_index().rename(columns={
        "fold_nunique": "n_folds",
        "n_test_sum": "n_test_total",
        "n_branches_mean": "n_branches_mean",
    })
    summary = summary[summary["n_folds"].ge(3)].copy()
    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in summary:
            summary[mean_col] = summary[mean_col].astype(float)
        if std_col in summary:
            summary[std_col] = summary[std_col].fillna(0.0).astype(float)
    return summary


def _best_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = f"{metric}_mean"
    if col not in df:
        return pd.DataFrame()
    rows = []
    ascending = metric in LOWER_IS_BETTER
    for dataset, sub in df.dropna(subset=[col]).groupby("dataset_short"):
        if sub.empty:
            continue
        best = sub.sort_values(col, ascending=ascending).iloc[0].copy()
        best["selection_metric"] = metric
        rows.append(best)
    return pd.DataFrame(rows)


def build_best_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    previous = summary[~summary["is_new"]].copy()
    new = summary[summary["is_new"]].copy()
    rows = []
    for metric in ["mcc", "auprc_ovr", "balanced_accuracy", "sensitivity", "brier_score", "ece_10"]:
        prev_best = _best_by_metric(previous, metric)
        new_best = _best_by_metric(new, metric)
        if prev_best.empty or new_best.empty:
            continue
        prev_best = prev_best.set_index("dataset_short")
        new_best = new_best.set_index("dataset_short")
        for dataset in sorted(set(prev_best.index) & set(new_best.index)):
            p = prev_best.loc[dataset]
            n = new_best.loc[dataset]
            metric_col = f"{metric}_mean"
            delta = n[metric_col] - p[metric_col]
            if metric in LOWER_IS_BETTER:
                delta = p[metric_col] - n[metric_col]
            row = {
                "dataset": dataset,
                "selection_metric": metric,
                "previous_label": p["label"],
                "previous_sweep": p["sweep"],
                "previous_stage": p["stage"],
                "previous_rule_source": p["rule_source"],
                "new_label": n["label"],
                "new_stage": n["stage"],
                "new_rule_source": n["rule_source"],
                "previous_metric": p[metric_col],
                "new_metric": n[metric_col],
                "delta_positive_is_better": delta,
            }
            for extra in ["mcc", "auprc_ovr", "balanced_accuracy", "sensitivity", "specificity", "brier_score", "ece_10", "net_benefit_0_10"]:
                c = f"{extra}_mean"
                if c in summary:
                    row[f"previous_{extra}"] = p.get(c, np.nan)
                    row[f"new_{extra}"] = n.get(c, np.nan)
                    row[f"delta_{extra}"] = n.get(c, np.nan) - p.get(c, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)




def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    safe = df.loc[:, cols].copy()
    headers = list(safe.columns)
    rows = [[_format_value(v) for v in row] for row in safe.itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"
    out = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(fmt_row(row) for row in rows)
    return "\n".join(out)

def write_markdown(summary: pd.DataFrame, comparison: pd.DataFrame, out_path: Path) -> None:
    new = summary[summary["is_new"]].copy()
    lines: list[str] = []
    lines.append("# Rahmatullaev Improvement Sweep Leaderboard")
    lines.append("")
    lines.append(f"New sweep: `{NEW_SWEEP}`")
    lines.append("")
    lines.append("## Best New Models By Dataset (MCC)")
    lines.append("")
    cols = ["dataset_short", "label", "stage", "rule_source", "mcc_mean", "auprc_ovr_mean", "balanced_accuracy_mean", "sensitivity_mean", "brier_score_mean", "ece_10_mean"]
    best_new = _best_by_metric(new, "mcc")
    if not best_new.empty:
        lines.append(_markdown_table(best_new[cols].sort_values("dataset_short"), cols))
    lines.append("")
    lines.append("## Previous Best vs New Best")
    lines.append("")
    if not comparison.empty:
        mcc_cmp = comparison[comparison["selection_metric"].eq("mcc")].copy()
        show = [
            "dataset", "previous_label", "previous_mcc", "new_label", "new_mcc", "delta_mcc",
            "previous_auprc_ovr", "new_auprc_ovr", "previous_brier_score", "new_brier_score",
        ]
        lines.append(_markdown_table(mcc_cmp[show].sort_values("dataset"), show))
    lines.append("")
    lines.append("## Top New Rows")
    lines.append("")
    top_new = new.sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).groupby("dataset_short").head(10)
    lines.append(_markdown_table(top_new[cols], cols))
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_JOBS_ROOT / NEW_SWEEP / "summary_tables")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.jobs_root)
    summary = summarize_runs(rows)
    comparison = build_best_comparison(summary)

    all_path = args.output_dir / "mortality_all_tabular_run_summary.csv"
    new_path = args.output_dir / "rahmatullaev_new_run_summary.csv"
    cmp_path = args.output_dir / "previous_best_vs_rahmatullaev_new.csv"
    top_path = args.output_dir / "dataset_top20_mcc_with_new.csv"
    md_path = args.output_dir / "RAHMATULLAEV_IMPROVEMENT_LEADERBOARD.md"

    summary.sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).to_csv(all_path, index=False)
    summary[summary["is_new"]].sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).to_csv(new_path, index=False)
    comparison.sort_values(["selection_metric", "dataset"]).to_csv(cmp_path, index=False)
    top20 = summary.sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).groupby("dataset_short").head(20)
    top20.to_csv(top_path, index=False)
    write_markdown(summary, comparison, md_path)

    print(f"rows_raw={len(rows)} complete_run_rows={len(summary)} new_complete_run_rows={int(summary['is_new'].sum())}")
    print(f"wrote={all_path}")
    print(f"wrote={new_path}")
    print(f"wrote={cmp_path}")
    print(f"wrote={top_path}")
    print(f"wrote={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
