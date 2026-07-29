#!/usr/bin/env python3
"""Aggregate constrained dual-residual PPtheta-Post mortality jobs.

Scans all compare_datasets CSVs under output/mortality_paper_jobs and writes
leaderboards for the section 43 dual-residual sweep against prior PPtheta runs
and TabPFN/EBM baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = ROOT / "output" / "mortality_paper_jobs"
NEW_SWEEP = "rahmatullaev_dual_residual_ppost_mortality_dual_residual_ppost_v1"
OUT_DIR = JOBS_ROOT / "common_tables" / "rahmatullaev_dual_residual_ppost_v1"
DATASET_MAP = {
    "mimic3_mortality_48h_tabular": "mimic3",
    "mimic4_mortality_48h_tabular": "mimic4",
    "eicu_mortality_48h_tabular": "eicu",
}
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_weighted",
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
COMPARE_METRICS = [
    "mcc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auprc_ovr",
    "brier_score",
    "ece_10",
    "log_loss",
    "net_benefit_0_10",
]


def _csv_paths(root: Path):
    yield from root.glob("*/**/compare_datasets*.csv")


def _path_meta(path: Path, root: Path) -> dict[str, str]:
    parts = path.relative_to(root).parts
    return {
        "sweep": parts[0] if len(parts) > 0 else "",
        "dataset_axis": parts[1] if len(parts) > 1 else "",
        "stage": parts[2] if len(parts) > 2 else "",
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
        if df.empty or not {"dataset", "fold", "variant", "label"}.issubset(df.columns):
            continue
        meta = _path_meta(path, root)
        for key, value in meta.items():
            df[key] = value
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No compare_datasets CSVs under {root}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["dataset_short"] = out["dataset"].map(DATASET_MAP).fillna(out["dataset_axis"])
    out["is_dual_sweep"] = out["sweep"].eq(NEW_SWEEP)
    label = out["label"].fillna(out["variant"].astype(str)).astype(str)
    variant = out["variant"].fillna("").astype(str)
    out["is_dual_method"] = out["is_dual_sweep"] & (
        label.str.contains("DualResidual", case=False, regex=False)
        | variant.str.contains("dual_residual", case=False, regex=False)
    )
    out["is_ppost"] = label.str.contains("PPtheta", case=False, regex=False) | variant.str.contains("pp_theta", case=False, regex=False)
    out["is_baseline_tabpfn"] = label.eq("TabPFN")
    out["is_baseline_ebm"] = label.eq("EBM")
    for col in ["fold", "n_test", "n_branches", *METRICS]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [m for m in METRICS if m in rows.columns]
    group_cols = [
        "dataset_short",
        "sweep",
        "stage",
        "csv_path",
        "is_dual_sweep",
        "is_dual_method",
        "is_ppost",
        "is_baseline_tabpfn",
        "is_baseline_ebm",
        "rule_source",
        "variant",
        "label",
    ]
    agg = {m: ["mean", "std"] for m in metric_cols}
    agg.update({"fold": "nunique", "n_test": "sum", "n_branches": "mean"})
    summary = rows.groupby(group_cols, dropna=False).agg(agg)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.reset_index().rename(
        columns={
            "fold_nunique": "n_folds",
            "n_test_sum": "n_test_total",
            "n_branches_mean": "n_branches_mean",
        }
    )
    summary = summary[summary["n_folds"].ge(3)].copy()
    for m in metric_cols:
        for suffix in ["mean", "std"]:
            col = f"{m}_{suffix}"
            if col in summary:
                summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = f"{metric}_mean"
    if col not in df.columns:
        return pd.DataFrame()
    asc = metric in LOWER_IS_BETTER
    rows = []
    for dataset, sub in df.dropna(subset=[col]).groupby("dataset_short"):
        if sub.empty:
            continue
        rows.append(sub.sort_values(col, ascending=asc).iloc[0].copy())
    return pd.DataFrame(rows)


def comparison(summary: pd.DataFrame) -> pd.DataFrame:
    dual = summary[summary["is_dual_method"]].copy()
    previous_ppost = summary[summary["is_ppost"] & ~summary["is_dual_sweep"]].copy()
    tabpfn = summary[summary["is_baseline_tabpfn"]].copy()
    ebm = summary[summary["is_baseline_ebm"]].copy()
    cohorts = [
        ("previous_ppost", previous_ppost),
        ("tabpfn", tabpfn),
        ("ebm", ebm),
    ]
    rows = []
    for metric in COMPARE_METRICS:
        dual_best = best_by_metric(dual, metric)
        if dual_best.empty:
            continue
        dual_best = dual_best.set_index("dataset_short")
        for cohort_name, cohort_df in cohorts:
            other_best = best_by_metric(cohort_df, metric)
            if other_best.empty:
                continue
            other_best = other_best.set_index("dataset_short")
            for dataset in sorted(set(dual_best.index) & set(other_best.index)):
                d = dual_best.loc[dataset]
                o = other_best.loc[dataset]
                col = f"{metric}_mean"
                if metric in LOWER_IS_BETTER:
                    delta = o[col] - d[col]
                else:
                    delta = d[col] - o[col]
                row = {
                    "dataset": dataset,
                    "selection_metric": metric,
                    "reference": cohort_name,
                    "reference_label": o["label"],
                    "reference_sweep": o["sweep"],
                    "reference_stage": o["stage"],
                    "reference_rule_source": o["rule_source"],
                    "dual_label": d["label"],
                    "dual_stage": d["stage"],
                    "dual_rule_source": d["rule_source"],
                    "reference_metric": o[col],
                    "dual_metric": d[col],
                    "delta_positive_is_better": delta,
                }
                for extra in COMPARE_METRICS:
                    c = f"{extra}_mean"
                    if c in summary.columns:
                        row[f"reference_{extra}"] = o.get(c, np.nan)
                        row[f"dual_{extra}"] = d.get(c, np.nan)
                        row[f"delta_{extra}"] = d.get(c, np.nan) - o.get(c, np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    d = df.loc[:, cols].copy()
    headers = list(d.columns)
    vals = [[fmt(v) for v in row] for row in d.itertuples(index=False, name=None)]
    widths = [len(h) for h in headers]
    for row in vals:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"

    return "\n".join([line(headers), "| " + " | ".join("-" * w for w in widths) + " |", *[line(v) for v in vals]])


def write_md(summary: pd.DataFrame, comp: pd.DataFrame, out: Path) -> None:
    dual = summary[summary["is_dual_method"]].copy()
    lines: list[str] = []
    lines += ["# Dual-Residual PPtheta Leaderboard", "", f"New sweep: `{NEW_SWEEP}`", ""]
    metric_cols = [
        "dataset_short",
        "label",
        "stage",
        "rule_source",
        "mcc_mean",
        "balanced_accuracy_mean",
        "sensitivity_mean",
        "specificity_mean",
        "auprc_ovr_mean",
        "brier_score_mean",
        "ece_10_mean",
        "log_loss_mean",
    ]
    lines += ["## Best Dual-Residual By MCC", ""]
    lines.append(md_table(best_by_metric(dual, "mcc").sort_values("dataset_short"), metric_cols))
    lines += ["", "## Dual vs Previous PPtheta Best By MCC", ""]
    show = [
        "dataset",
        "reference_label",
        "reference_mcc",
        "dual_label",
        "dual_mcc",
        "delta_mcc",
        "reference_balanced_accuracy",
        "dual_balanced_accuracy",
        "reference_sensitivity",
        "dual_sensitivity",
        "reference_brier_score",
        "dual_brier_score",
        "reference_ece_10",
        "dual_ece_10",
    ]
    pp = comp[(comp["selection_metric"].eq("mcc")) & (comp["reference"].eq("previous_ppost"))].sort_values("dataset")
    lines.append(md_table(pp, show))
    lines += ["", "## Dual vs TabPFN/EBM By MCC", ""]
    base_show = ["dataset", "reference", "reference_label", "reference_mcc", "dual_label", "dual_mcc", "delta_mcc", "reference_brier_score", "dual_brier_score", "reference_ece_10", "dual_ece_10"]
    base = comp[(comp["selection_metric"].eq("mcc")) & (comp["reference"].isin(["tabpfn", "ebm"]))].sort_values(["dataset", "reference"])
    lines.append(md_table(base, base_show))
    for metric in ["balanced_accuracy", "sensitivity", "brier_score", "ece_10", "log_loss"]:
        lines += ["", f"## Best Dual-Residual By {metric}", ""]
        lines.append(md_table(best_by_metric(dual, metric).sort_values("dataset_short"), metric_cols))
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-root", type=Path, default=JOBS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.jobs_root)
    summary = summarize(rows)
    comp = comparison(summary)
    dual = summary[summary["is_dual_method"]].copy()

    summary.sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).to_csv(args.output_dir / "mortality_all_with_dual_residual_summary.csv", index=False)
    dual.sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).to_csv(args.output_dir / "dual_residual_new_summary.csv", index=False)
    comp.sort_values(["selection_metric", "reference", "dataset"]).to_csv(args.output_dir / "dual_vs_references_by_metric.csv", index=False)
    top = summary.sort_values(["dataset_short", "mcc_mean"], ascending=[True, False]).groupby("dataset_short").head(30)
    top.to_csv(args.output_dir / "dataset_top30_mcc_with_dual_residual.csv", index=False)
    write_md(summary, comp, args.output_dir / "DUAL_RESIDUAL_LEADERBOARD.md")

    print(f"raw_rows={len(rows)} complete_runs={len(summary)} dual_complete={len(dual)}")
    for name in [
        "mortality_all_with_dual_residual_summary.csv",
        "dual_residual_new_summary.csv",
        "dual_vs_references_by_metric.csv",
        "dataset_top30_mcc_with_dual_residual.csv",
        "DUAL_RESIDUAL_LEADERBOARD.md",
    ]:
        print(args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
