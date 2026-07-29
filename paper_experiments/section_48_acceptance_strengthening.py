#!/usr/bin/env python3
"""Section 48: acceptance-strengthening experiments for PPtheta-Post.

This section turns the AAAI reviewer-strengthening plan into reproducible
artifacts. It deliberately keeps the main claim narrow: PPtheta-Post is tested
as a prediction-sufficient posterior evidence interface, not as a universal
replacement for TabPFN or EBM.
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
from paper_experiments.section_28_prediction_artifact_metrics import (  # noqa: E402
    compute_metrics,
    normalize_proba,
)

PROOF_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_ppost_proof_mortality_ppost_proof_local_v1"

SELECTED = {
    "eicu": {
        "source": "tabpfn_distill_xgb_soft",
        "variant": "pp_theta_post_ebm_bounded_residual_gate",
        "stage": "rahmatullaev_proof_selective_utility",
        "claim": "Audit signal, compact trace, no MCC gain",
    },
    "mimic3": {
        "source": "xgb",
        "variant": "pp_theta_post_rule_family_calibrated",
        "stage": "rahmatullaev_proof_evidence_ablation",
        "claim": "Utility plus compact teacher-free trace",
    },
    "mimic4": {
        "source": "tabpfn_distill_xgb_soft",
        "variant": "pp_theta_post_ebm_residual_mcc",
        "stage": "rahmatullaev_proof_strong_base_repair",
        "claim": "Utility plus sensitivity gain",
    },
}

FRACTIONS = (0.01, 0.05, 0.10, 0.20, 1.00)
METRICS = ("mcc", "balanced_accuracy", "sensitivity", "specificity", "auprc_ovr", "roc_auc_ovr", "log_loss", "brier_score", "ece_10")


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _infer_dataset_key(dataset_arg: str) -> str:
    text = dataset_arg.lower()
    if "mimic4" in text:
        return "mimic4"
    if "mimic3" in text:
        return "mimic3"
    if "eicu" in text:
        return "eicu"
    raise ValueError(f"Cannot infer dataset key from {dataset_arg!r}")


def _extract_option(args: list[str], option: str, default: str | None = None) -> str | None:
    for i, value in enumerate(args[:-1]):
        if value == option:
            return args[i + 1]
    return default


def _strip_options(args: list[str], options: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        value = args[i]
        if value in options:
            i += 2
            continue
        out.append(value)
        i += 1
    return out


def _latest_compare_csv(out_dir: Path) -> Path:
    csvs = sorted(p for p in out_dir.glob("compare_datasets_*.csv") if not p.name.startswith("ppost_"))
    if not csvs:
        raise FileNotFoundError(f"No compare_datasets_*.csv in {out_dir}")
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _artifact_path(row: dict[str, str]) -> Path | None:
    raw = row.get("prediction_artifact", "")
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


def _load_artifact(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
    path = _artifact_path(row)
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["y_true"], dtype=int), normalize_proba(np.asarray(data["proba"], dtype=float))


def _selected_from_passthrough(passthrough: list[str]) -> tuple[str, dict[str, str], Path]:
    dataset_arg = _extract_option(passthrough, "--datasets", "") or ""
    dataset_key = _infer_dataset_key(dataset_arg)
    selected = SELECTED[dataset_key]
    out_dir = Path(_extract_option(passthrough, "--output-dir", str(ROOT / "output/paper/48_acceptance_strengthening")) or "")
    return dataset_key, selected, out_dir


def _compare_base_args(passthrough: list[str], selected: dict[str, str], out_dir: Path) -> list[str]:
    stripped = _strip_options(
        passthrough,
        {
            "--output-dir",
            "--rule-sources",
            "--variants",
            "--baselines",
            "--rule-selection",
            "--rule-budget",
            "--top-k-ratio",
            "--top-k-min",
            "--top-k-max",
            "--sparse-logit-top-k",
        },
    )
    return [
        *stripped,
        "--output-dir", str(out_dir),
        "--rule-sources", selected["source"],
        "--variants", "source_native," + selected["variant"],
        "--baselines", "none",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--top-k-min", "1",
        "--top-k-max", "100000",
        "--save-predictions",
    ]


def _summarize_compare(csv_path: Path, fraction: float, selected: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_csv(csv_path)
    native = [r for r in rows if r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"]
    ppost = [r for r in rows if r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]]
    native_by_fold = {r.get("fold", ""): r for r in native}
    fold_rows: list[dict[str, Any]] = []
    for row in ppost:
        fold = row.get("fold", "")
        base = native_by_fold.get(fold)
        if base is None:
            continue
        item: dict[str, Any] = {
            "fraction": fraction,
            "fold": fold,
            "dataset": row.get("dataset", ""),
            "rule_source": selected["source"],
            "variant": selected["variant"],
            "n_branches": _float(row.get("n_branches")),
            "top_k": _float(row.get("top_k")),
            "trace_fraction": _float(row.get("top_k")) / _float(row.get("n_branches")) if _float(row.get("n_branches")) > 0 else float("nan"),
        }
        for metric in METRICS:
            item[f"native_{metric}"] = _float(base.get(metric))
            item[f"ppost_{metric}"] = _float(row.get(metric))
            item[f"delta_{metric}"] = item[f"ppost_{metric}"] - item[f"native_{metric}"]
        fold_rows.append(item)
    summary: dict[str, Any] = {
        "fraction": fraction,
        "folds": len(fold_rows),
        "rule_source": selected["source"],
        "variant": selected["variant"],
        "mean_trace_fraction": _mean(r["trace_fraction"] for r in fold_rows),
    }
    for metric in METRICS:
        summary[f"native_{metric}"] = _mean(r[f"native_{metric}"] for r in fold_rows)
        summary[f"ppost_{metric}"] = _mean(r[f"ppost_{metric}"] for r in fold_rows)
        summary[f"delta_{metric}"] = _mean(r[f"delta_{metric}"] for r in fold_rows)
    return fold_rows, summary


def run_trace_sufficiency_curve(passthrough: list[str]) -> int:
    dataset_key, selected, out_dir = _selected_from_passthrough(passthrough)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_fold_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        frac_dir = out_dir / f"fraction_{fraction:.2f}".replace(".", "p")
        family_top_k = max(1, int(round(384 * fraction)))
        old_family = os.environ.get("PPPOST_RULE_FAMILY_TOPK")
        old_residual = os.environ.get("PPPOST_EBM_RESIDUAL_TOPK")
        os.environ["PPPOST_RULE_FAMILY_TOPK"] = str(family_top_k)
        os.environ["PPPOST_EBM_RESIDUAL_TOPK"] = str(family_top_k)
        args = _compare_base_args(passthrough, selected, frac_dir) + [
            "--top-k-ratio", str(fraction),
            "--sparse-logit-top-k", str(family_top_k),
        ]
        print(f"[section48] trace_sufficiency dataset={dataset_key} fraction={fraction} family_top_k={family_top_k}")
        rc = run_compare_datasets(args)
        if old_family is None:
            os.environ.pop("PPPOST_RULE_FAMILY_TOPK", None)
        else:
            os.environ["PPPOST_RULE_FAMILY_TOPK"] = old_family
        if old_residual is None:
            os.environ.pop("PPPOST_EBM_RESIDUAL_TOPK", None)
        else:
            os.environ["PPPOST_EBM_RESIDUAL_TOPK"] = old_residual
        if rc != 0:
            return rc
        fold_rows, summary = _summarize_compare(_latest_compare_csv(frac_dir), fraction, selected)
        all_fold_rows.extend(fold_rows)
        summaries.append(summary)
    full = next((s for s in summaries if abs(float(s["fraction"]) - 1.0) < 1e-9), summaries[-1])
    full_mcc = _float(full.get("ppost_mcc"))
    full_delta = _float(full.get("delta_mcc"))
    for row in summaries:
        row["mcc_retained_vs_full"] = _float(row.get("ppost_mcc")) / full_mcc if full_mcc and math.isfinite(full_mcc) else float("nan")
        row["delta_mcc_retained_vs_full"] = _float(row.get("delta_mcc")) / full_delta if full_delta and math.isfinite(full_delta) and abs(full_delta) > 1e-12 else float("nan")
        row["claim"] = selected["claim"]
    _write_csv(out_dir / "trace_sufficiency_curve_folds.csv", all_fold_rows)
    _write_csv(out_dir / "trace_sufficiency_curve_summary.csv", summaries)
    _write_trace_md(out_dir / "trace_sufficiency_curve.md", dataset_key, selected, summaries)
    return 0


def _write_trace_md(path: Path, dataset_key: str, selected: dict[str, str], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Trace Sufficiency Curve: {dataset_key}",
        "",
        f"Source: `{selected['source']}`",
        f"Variant: `{selected['variant']}`",
        f"Claim: {selected['claim']}",
        "",
        "| Evidence budget | Trace fraction | +PPtheta MCC | dMCC | dSens | dBrier | dECE | MCC retained |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {fraction:.0%} | {trace} | {mcc} | {dmcc} | {dsens} | {dbrier} | {dece} | {ret} |".format(
                fraction=float(row["fraction"]),
                trace=_fmt(_float(row.get("mean_trace_fraction"))),
                mcc=_fmt(_float(row.get("ppost_mcc"))),
                dmcc=_fmt(_float(row.get("delta_mcc"))),
                dsens=_fmt(_float(row.get("delta_sensitivity"))),
                dbrier=_fmt(_float(row.get("delta_brier_score"))),
                dece=_fmt(_float(row.get("delta_ece_10"))),
                ret=_fmt(_float(row.get("mcc_retained_vs_full"))),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _proof_rows_for_dataset(dataset_key: str, selected: dict[str, str]) -> list[dict[str, str]]:
    path = PROOF_ROOT / dataset_key / selected["stage"] / "ppost_proof_pairwise.csv"
    rows = _read_csv(path)
    return [
        r for r in rows
        if r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]
    ]


def run_proof_statistics(passthrough: list[str]) -> int:
    dataset_key, selected, out_dir = _selected_from_passthrough(passthrough)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _proof_rows_for_dataset(dataset_key, selected)
    out_rows = []
    for metric in ("mcc", "sensitivity", "brier_score", "ece_10"):
        deltas = np.array([_float(r.get(f"delta_{metric}")) for r in rows], dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            continue
        signs = np.sum(deltas > 0)
        # Exact two-sided sign test under p=0.5, ignoring zeros.
        n = int(np.sum(deltas != 0))
        k = int(np.sum(deltas[deltas != 0] > 0))
        tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n) if n else float("nan")
        p_two = min(1.0, 2.0 * tail) if math.isfinite(tail) else float("nan")
        out_rows.append({
            "dataset": dataset_key,
            "source": selected["source"],
            "variant": selected["variant"],
            "metric": metric,
            "folds": int(deltas.size),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "positive_folds": int(signs),
            "negative_folds": int(np.sum(deltas < 0)),
            "sign_test_p_two_sided": p_two,
            "claim": selected["claim"],
        })
    _write_csv(out_dir / "proof_statistics.csv", out_rows)
    return 0


def run_case_study_trace_candidates(passthrough: list[str]) -> int:
    dataset_key, selected, out_dir = _selected_from_passthrough(passthrough)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise = _proof_rows_for_dataset(dataset_key, selected)
    candidate_rows: list[dict[str, Any]] = []
    for pair in pairwise:
        fold = pair.get("fold", "")
        base_art = _artifact_path({"prediction_artifact": pair.get("base_prediction_artifact", "")})
        # Pairwise tables from section 46 do not store base artifact paths; recover them from compare CSV.
        compare_path = PROOF_ROOT / dataset_key / selected["stage"]
        csvs = sorted(compare_path.glob("compare_datasets_*.csv"))
        if not csvs:
            continue
        rows = _read_csv(max(csvs, key=lambda p: p.stat().st_mtime))
        base = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"), None)
        pp = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]), None)
        if base is None or pp is None:
            continue
        loaded_base = _load_artifact(base)
        loaded_pp = _load_artifact(pp)
        if loaded_base is None or loaded_pp is None:
            continue
        y, p_base = loaded_base
        y_pp, p_pp = loaded_pp
        if not np.array_equal(y, y_pp):
            continue
        pred_base = np.argmax(p_base, axis=1)
        pred_pp = np.argmax(p_pp, axis=1)
        improvement = (pred_base != y) & (pred_pp == y)
        risk_shift = p_pp[:, 1] - p_base[:, 1] if p_pp.shape[1] == 2 else np.max(p_pp - p_base, axis=1)
        score = np.abs(risk_shift) + improvement.astype(float)
        for rank, idx in enumerate(np.argsort(-score)[:10], start=1):
            candidate_rows.append({
                "dataset": dataset_key,
                "fold": fold,
                "rank": rank,
                "patient_index_in_fold": int(idx),
                "y_true": int(y[idx]),
                "native_pred": int(pred_base[idx]),
                "ppost_pred": int(pred_pp[idx]),
                "native_p_mortality": float(p_base[idx, 1]) if p_base.shape[1] == 2 else float(np.max(p_base[idx])),
                "ppost_p_mortality": float(p_pp[idx, 1]) if p_pp.shape[1] == 2 else float(np.max(p_pp[idx])),
                "risk_shift": float(risk_shift[idx]),
                "native_wrong_ppost_right": int(bool(improvement[idx])),
                "source": selected["source"],
                "variant": selected["variant"],
                "claim": selected["claim"],
            })
    _write_csv(out_dir / "case_study_trace_candidates.csv", candidate_rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("trace_sufficiency_curve", "proof_statistics", "case_study_trace_candidates"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = build_parser()
    known, passthrough = parser.parse_known_args(argv)
    if known.experiment == "trace_sufficiency_curve":
        return run_trace_sufficiency_curve(passthrough)
    if known.experiment == "proof_statistics":
        return run_proof_statistics(passthrough)
    if known.experiment == "case_study_trace_candidates":
        return run_case_study_trace_candidates(passthrough)
    raise AssertionError(known.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
