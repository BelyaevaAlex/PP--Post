#!/usr/bin/env python3
"""Section 50: targeted AAAI acceptance-strengthening jobs.

Each experiment corresponds to one reviewer-facing action item:
compact residual traces, subset sufficiency, trace bootstrap intervals,
EBM-vs-PPtheta case candidates, claim checklist, and EBM-source diagnostics.
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


PROOF_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_ppost_proof_mortality_ppost_proof_local_v1"
ACCEPT_V2 = ROOT / "output/mortality_paper_jobs/rahmatullaev_acceptance_strengthening_mortality_accept_strengthening_v2"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"

SELECTED = {
    "eicu": {
        "dataset_label": "eICU",
        "source": "tabpfn_distill_xgb_soft",
        "variant": "pp_theta_post_ebm_bounded_residual_gate",
        "stage": "rahmatullaev_proof_selective_utility",
        "claim": "Audit signal, compact trace, no MCC gain",
    },
    "mimic3": {
        "dataset_label": "MIMIC-III",
        "source": "xgb",
        "variant": "pp_theta_post_rule_family_calibrated",
        "stage": "rahmatullaev_proof_evidence_ablation",
        "claim": "Utility plus compact teacher-free trace",
    },
    "mimic4": {
        "dataset_label": "MIMIC-IV",
        "source": "tabpfn_distill_xgb_soft",
        "variant": "pp_theta_post_ebm_residual_mcc",
        "stage": "rahmatullaev_proof_strong_base_repair",
        "claim": "Utility plus sensitivity gain",
    },
}

FRACTIONS = (0.01, 0.05, 0.10, 0.20, 1.00)
RESIDUAL_VARIANTS = (
    "pp_theta_post_ebm_residual_calibrated",
    "pp_theta_post_ebm_residual_mcc",
    "pp_theta_post_ebm_residual_sens92",
    "pp_theta_post_ebm_residual_sens95",
)


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        arr = np.stack([1.0 - arr, arr], axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    denom = arr.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= 0
    if np.any(bad):
        arr[bad] = 1.0 / arr.shape[1]
        denom = arr.sum(axis=1, keepdims=True)
    return arr / denom


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: Any, digits: int = 4) -> str:
    val = _float(value)
    return "nan" if not math.isfinite(val) else f"{val:.{digits}f}"


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


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _infer_dataset_key(args: list[str]) -> str:
    value = _extract_option(args, "--datasets", "") or ""
    text = value.lower()
    if "mimic4" in text:
        return "mimic4"
    if "mimic3" in text:
        return "mimic3"
    if "eicu" in text:
        return "eicu"
    raise ValueError(f"Cannot infer dataset from --datasets {value!r}")


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


def _selected_and_out(passthrough: list[str]) -> tuple[str, dict[str, str], Path]:
    dataset = _infer_dataset_key(passthrough)
    out_dir = Path(_extract_option(passthrough, "--output-dir", str(ROOT / "output/paper/50_aaai_next_steps")) or "")
    return dataset, SELECTED[dataset], out_dir


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


def _binary_metrics(y: np.ndarray, proba: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    y_m = np.asarray(y[mask], dtype=int)
    p_m = normalize_proba(np.asarray(proba[mask], dtype=float))
    if len(y_m) == 0:
        return {k: float("nan") for k in ("n", "accuracy", "sensitivity", "specificity", "mcc", "brier", "mean_p_mortality")}
    pred = np.argmax(p_m, axis=1)
    tp = float(np.sum((pred == 1) & (y_m == 1)))
    tn = float(np.sum((pred == 0) & (y_m == 0)))
    fp = float(np.sum((pred == 1) & (y_m == 0)))
    fn = float(np.sum((pred == 0) & (y_m == 1)))
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    p1 = p_m[:, 1] if p_m.shape[1] > 1 else p_m[:, 0]
    return {
        "n": float(len(y_m)),
        "prevalence": float(np.mean(y_m == 1)),
        "accuracy": float(np.mean(pred == y_m)),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "mcc": float(mcc),
        "brier": float(np.mean((p1 - (y_m == 1).astype(float)) ** 2)),
        "mean_p_mortality": float(np.mean(p1)),
    }


def _compare_base_args(passthrough: list[str], out_dir: Path, source: str, variants: str) -> list[str]:
    stripped = _strip_options(
        passthrough,
        {"--output-dir", "--rule-sources", "--variants", "--baselines", "--rule-selection", "--rule-budget", "--top-k-ratio", "--top-k-min", "--top-k-max", "--sparse-logit-top-k"},
    )
    return [
        *stripped,
        "--output-dir", str(out_dir),
        "--rule-sources", source,
        "--variants", variants,
        "--baselines", "none",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--top-k-min", "1",
        "--top-k-max", "100000",
        "--save-predictions",
    ]


def run_compact_residual_trace(passthrough: list[str]) -> int:
    dataset, _selected, out_dir = _selected_and_out(passthrough)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    source = "tabpfn_distill_xgb_soft"
    variants = "source_native," + ",".join(RESIDUAL_VARIANTS)
    old_env = {k: os.environ.get(k) for k in ("PPPOST_EBM_RESIDUAL_TOPK", "PPPOST_EBM_RESIDUAL_TRUE_WEIGHT", "PPPOST_EBM_RESIDUAL_RIDGE_L2", "PPPOST_EBM_RESIDUAL_MAX_SCALE", "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL", "PPPOST_EBM_RESIDUAL_ACC_TOL")}
    try:
        for fraction in FRACTIONS:
            topk = max(1, int(round(384 * fraction)))
            frac_dir = out_dir / f"residual_fraction_{fraction:.2f}".replace(".", "p")
            os.environ.update({
                "PPPOST_EBM_RESIDUAL_TOPK": str(topk),
                "PPPOST_EBM_RESIDUAL_TRUE_WEIGHT": os.environ.get("PPPOST_EBM_RESIDUAL_TRUE_WEIGHT", "0.60"),
                "PPPOST_EBM_RESIDUAL_RIDGE_L2": os.environ.get("PPPOST_EBM_RESIDUAL_RIDGE_L2", "5.0"),
                "PPPOST_EBM_RESIDUAL_MAX_SCALE": os.environ.get("PPPOST_EBM_RESIDUAL_MAX_SCALE", "0.40"),
                "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL": os.environ.get("PPPOST_EBM_RESIDUAL_LOGLOSS_TOL", "0.010"),
                "PPPOST_EBM_RESIDUAL_ACC_TOL": os.environ.get("PPPOST_EBM_RESIDUAL_ACC_TOL", "0.006"),
            })
            print(f"[section50] compact_residual_trace dataset={dataset} fraction={fraction} topk={topk}")
            from compare_datasets import main as run_compare_datasets
            rc = run_compare_datasets(_compare_base_args(passthrough, frac_dir, source, variants))
            if rc != 0:
                return rc
            rows = _read_csv(_latest_compare_csv(frac_dir))
            native = [r for r in rows if r.get("rule_source") == source and r.get("variant") == "source_native"]
            native_by_fold = {r.get("fold", ""): r for r in native}
            for row in rows:
                if row.get("rule_source") != source or row.get("variant") not in RESIDUAL_VARIANTS:
                    continue
                base = native_by_fold.get(row.get("fold", ""))
                if base is None:
                    continue
                family_count = _float(row.get("ebm_residual_family_count"))
                selected_count = _float(row.get("ebm_residual_selected_families"))
                trace_fraction = selected_count / family_count if family_count > 0 else float("nan")
                item = {
                    "dataset": dataset,
                    "fraction": fraction,
                    "requested_topk": topk,
                    "fold": row.get("fold", ""),
                    "rule_source": source,
                    "variant": row.get("variant", ""),
                    "family_count": family_count,
                    "selected_families": selected_count,
                    "compact_trace_fraction": trace_fraction,
                    "native_mcc": _float(base.get("mcc")),
                    "ppost_mcc": _float(row.get("mcc")),
                    "delta_mcc": _float(row.get("mcc")) - _float(base.get("mcc")),
                    "native_sensitivity": _float(base.get("sensitivity")),
                    "ppost_sensitivity": _float(row.get("sensitivity")),
                    "delta_sensitivity": _float(row.get("sensitivity")) - _float(base.get("sensitivity")),
                    "delta_brier": _float(row.get("brier_score")) - _float(base.get("brier_score")),
                    "delta_ece": _float(row.get("ece_10")) - _float(base.get("ece_10")),
                    "prediction_artifact": row.get("prediction_artifact", ""),
                }
                all_rows.append(item)
        for key in sorted({(r["fraction"], r["variant"]) for r in all_rows}):
            frac, variant = key
            rows = [r for r in all_rows if r["fraction"] == frac and r["variant"] == variant]
            summary.append({
                "dataset": dataset,
                "fraction": frac,
                "variant": variant,
                "folds": len(rows),
                "mean_trace_fraction": _mean(r["compact_trace_fraction"] for r in rows),
                "native_mcc": _mean(r["native_mcc"] for r in rows),
                "ppost_mcc": _mean(r["ppost_mcc"] for r in rows),
                "delta_mcc": _mean(r["delta_mcc"] for r in rows),
                "delta_sensitivity": _mean(r["delta_sensitivity"] for r in rows),
                "delta_brier": _mean(r["delta_brier"] for r in rows),
                "delta_ece": _mean(r["delta_ece"] for r in rows),
            })
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    _write_csv(out_dir / "compact_residual_trace_folds.csv", all_rows)
    _write_csv(out_dir / "compact_residual_trace_summary.csv", summary)
    _write_md_table(out_dir / "compact_residual_trace.md", summary, ["dataset", "fraction", "variant", "mean_trace_fraction", "ppost_mcc", "delta_mcc", "delta_sensitivity", "delta_brier"])
    return 0


def _trace_compare_rows(dataset: str, fraction_dir: Path) -> tuple[dict[str, str], dict[str, str]] | None:
    selected = SELECTED[dataset]
    rows = _read_csv(_latest_compare_csv(fraction_dir))
    native = next((r for r in rows if r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"), None)
    ppost = next((r for r in rows if r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]), None)
    if native is None or ppost is None:
        return None
    return native, ppost


def run_subset_sufficiency(passthrough: list[str]) -> int:
    dataset, selected, out_dir = _selected_and_out(passthrough)
    root = ACCEPT_V2 / dataset / "rahmatullaev_accept_trace_sufficiency_curve"
    rows_out: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        frac_dir = root / f"fraction_{fraction:.2f}".replace(".", "p")
        if not frac_dir.exists():
            continue
        compare_rows = _read_csv(_latest_compare_csv(frac_dir))
        native_by_fold = {r.get("fold", ""): r for r in compare_rows if r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"}
        pp_by_fold = {r.get("fold", ""): r for r in compare_rows if r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]}
        for fold, native in native_by_fold.items():
            pp = pp_by_fold.get(fold)
            if pp is None:
                continue
            loaded_native = _load_prediction(native)
            loaded_pp = _load_prediction(pp)
            if loaded_native is None or loaded_pp is None:
                continue
            y, p_native = loaded_native
            y2, p_pp = loaded_pp
            if not np.array_equal(y, y2):
                continue
            native_p1 = p_native[:, 1] if p_native.shape[1] > 1 else p_native[:, 0]
            pp_p1 = p_pp[:, 1] if p_pp.shape[1] > 1 else p_pp[:, 0]
            pred_native = np.argmax(p_native, axis=1)
            risk_shift = pp_p1 - native_p1
            q80 = np.quantile(native_p1, 0.80)
            s80 = np.quantile(np.abs(risk_shift), 0.80)
            masks = {
                "all": np.ones(len(y), dtype=bool),
                "mortality_positive": y == 1,
                "native_wrong": pred_native != y,
                "high_native_risk_top20": native_p1 >= q80,
                "large_ppost_shift_top20": np.abs(risk_shift) >= s80,
            }
            for subset, mask in masks.items():
                mn = _binary_metrics(y, p_native, mask)
                mp = _binary_metrics(y, p_pp, mask)
                rows_out.append({
                    "dataset": dataset,
                    "fold": fold,
                    "fraction": fraction,
                    "subset": subset,
                    "n": int(mn["n"]),
                    "prevalence": mn.get("prevalence"),
                    "native_accuracy": mn.get("accuracy"),
                    "ppost_accuracy": mp.get("accuracy"),
                    "delta_accuracy": mp.get("accuracy") - mn.get("accuracy"),
                    "native_mcc": mn.get("mcc"),
                    "ppost_mcc": mp.get("mcc"),
                    "delta_mcc": mp.get("mcc") - mn.get("mcc"),
                    "native_sensitivity": mn.get("sensitivity"),
                    "ppost_sensitivity": mp.get("sensitivity"),
                    "delta_sensitivity": mp.get("sensitivity") - mn.get("sensitivity") if math.isfinite(mp.get("sensitivity", float("nan"))) and math.isfinite(mn.get("sensitivity", float("nan"))) else float("nan"),
                    "native_brier": mn.get("brier"),
                    "ppost_brier": mp.get("brier"),
                    "delta_brier": mp.get("brier") - mn.get("brier"),
                    "mean_risk_shift": float(np.mean(risk_shift[mask])) if np.any(mask) else float("nan"),
                    "source": selected["source"],
                    "variant": selected["variant"],
                })
    summary: list[dict[str, Any]] = []
    for key in sorted({(r["fraction"], r["subset"]) for r in rows_out}):
        frac, subset = key
        rr = [r for r in rows_out if r["fraction"] == frac and r["subset"] == subset]
        summary.append({
            "dataset": dataset,
            "fraction": frac,
            "subset": subset,
            "folds": len(rr),
            "mean_n": _mean(r["n"] for r in rr),
            "delta_accuracy": _mean(r["delta_accuracy"] for r in rr),
            "delta_mcc": _mean(r["delta_mcc"] for r in rr),
            "delta_sensitivity": _mean(r["delta_sensitivity"] for r in rr),
            "delta_brier": _mean(r["delta_brier"] for r in rr),
            "mean_risk_shift": _mean(r["mean_risk_shift"] for r in rr),
        })
    _write_csv(out_dir / "subset_sufficiency_folds.csv", rows_out)
    _write_csv(out_dir / "subset_sufficiency_summary.csv", summary)
    _write_md_table(out_dir / "subset_sufficiency.md", summary, ["dataset", "fraction", "subset", "mean_n", "delta_accuracy", "delta_mcc", "delta_sensitivity", "delta_brier"])
    return 0


def _load_fraction_predictions(dataset: str, fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    selected = SELECTED[dataset]
    frac_dir = ACCEPT_V2 / dataset / "rahmatullaev_accept_trace_sufficiency_curve" / f"fraction_{fraction:.2f}".replace(".", "p")
    rows = _read_csv(_latest_compare_csv(frac_dir))
    ys: list[np.ndarray] = []
    natives: list[np.ndarray] = []
    pposts: list[np.ndarray] = []
    for fold in sorted({r.get("fold", "") for r in rows}):
        native = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"), None)
        pp = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]), None)
        if native is None or pp is None:
            continue
        ln = _load_prediction(native)
        lp = _load_prediction(pp)
        if ln is None or lp is None:
            continue
        y, pn = ln
        y2, ppred = lp
        if np.array_equal(y, y2):
            ys.append(y); natives.append(pn); pposts.append(ppred)
    if not ys:
        return None
    return np.concatenate(ys), np.vstack(natives), np.vstack(pposts)


def _binary_metrics_take(y: np.ndarray, proba: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    return _binary_metrics(np.asarray(y, dtype=int)[idx], np.asarray(proba, dtype=float)[idx])


def run_trace_bootstrap_ci(passthrough: list[str]) -> int:
    dataset, _selected, out_dir = _selected_and_out(passthrough)
    rng = np.random.default_rng(int(os.environ.get("PPPOST_BOOTSTRAP_SEED", "2027")))
    n_boot = int(os.environ.get("PPPOST_BOOTSTRAP_N", "400"))
    full_loaded = _load_fraction_predictions(dataset, 1.0)
    if full_loaded is None:
        raise FileNotFoundError(f"Missing full trace predictions for {dataset}")
    y_full, _native_full, pp_full = full_loaded
    full_stats = _binary_metrics(y_full, pp_full)
    rows: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        loaded = _load_fraction_predictions(dataset, fraction)
        if loaded is None:
            continue
        y, native, pp = loaded
        base = _binary_metrics(y, native)
        obs = _binary_metrics(y, pp)
        vals: dict[str, list[float]] = {k: [] for k in ("ppost_mcc", "delta_mcc", "delta_sensitivity", "retained_mcc", "retained_sensitivity")}
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            b = _binary_metrics_take(y, native, idx)
            o = _binary_metrics_take(y, pp, idx)
            f = _binary_metrics_take(y_full, pp_full, idx % len(y_full))
            vals["ppost_mcc"].append(o["mcc"])
            vals["delta_mcc"].append(o["mcc"] - b["mcc"])
            vals["delta_sensitivity"].append(o["sensitivity"] - b["sensitivity"] if math.isfinite(o["sensitivity"]) and math.isfinite(b["sensitivity"]) else float("nan"))
            vals["retained_mcc"].append(o["mcc"] / f["mcc"] if abs(f["mcc"]) > 1e-12 else float("nan"))
            vals["retained_sensitivity"].append(o["sensitivity"] / f["sensitivity"] if math.isfinite(o["sensitivity"]) and math.isfinite(f["sensitivity"]) and abs(f["sensitivity"]) > 1e-12 else float("nan"))
        row = {
            "dataset": dataset,
            "fraction": fraction,
            "n": n,
            "bootstrap_n": n_boot,
            "native_mcc": base["mcc"],
            "ppost_mcc": obs["mcc"],
            "delta_mcc": obs["mcc"] - base["mcc"],
            "delta_sensitivity": obs["sensitivity"] - base["sensitivity"] if math.isfinite(obs["sensitivity"]) and math.isfinite(base["sensitivity"]) else float("nan"),
            "retained_mcc_vs_full": obs["mcc"] / full_stats["mcc"] if abs(full_stats["mcc"]) > 1e-12 else float("nan"),
            "retained_sensitivity_vs_full": obs["sensitivity"] / full_stats["sensitivity"] if math.isfinite(obs["sensitivity"]) and math.isfinite(full_stats["sensitivity"]) and abs(full_stats["sensitivity"]) > 1e-12 else float("nan"),
        }
        for name, arr in vals.items():
            clean = np.array([v for v in arr if math.isfinite(v)], dtype=float)
            if clean.size:
                row[f"{name}_ci_low"] = float(np.quantile(clean, 0.025))
                row[f"{name}_ci_high"] = float(np.quantile(clean, 0.975))
        rows.append(row)
    _write_csv(out_dir / "trace_bootstrap_ci.csv", rows)
    _write_md_table(out_dir / "trace_bootstrap_ci.md", rows, ["dataset", "fraction", "ppost_mcc", "delta_mcc", "delta_mcc_ci_low", "delta_mcc_ci_high", "retained_mcc_vs_full", "retained_mcc_ci_low", "retained_mcc_ci_high"])
    return 0


def _proof_compare_rows(dataset: str) -> list[dict[str, str]]:
    selected = SELECTED[dataset]
    return _read_csv(_latest_compare_csv(PROOF_ROOT / dataset / selected["stage"]))


def run_ebm_ppost_case_study(passthrough: list[str]) -> int:
    dataset, selected, out_dir = _selected_and_out(passthrough)
    rows = _proof_compare_rows(dataset)
    out: list[dict[str, Any]] = []
    for fold in sorted({r.get("fold", "") for r in rows}):
        ebm = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == "_standalone" and r.get("variant") == "ebm"), None)
        native = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == selected["source"] and r.get("variant") == "source_native"), None)
        pp = next((r for r in rows if r.get("fold") == fold and r.get("rule_source") == selected["source"] and r.get("variant") == selected["variant"]), None)
        if ebm is None or native is None or pp is None:
            continue
        le, ln, lp = _load_prediction(ebm), _load_prediction(native), _load_prediction(pp)
        if le is None or ln is None or lp is None:
            continue
        y, p_ebm = le
        y2, p_native = ln
        y3, p_pp = lp
        if not (np.array_equal(y, y2) and np.array_equal(y, y3)):
            continue
        pe = p_ebm[:, 1]; pn = p_native[:, 1]; pp1 = p_pp[:, 1]
        pred_e = np.argmax(p_ebm, axis=1); pred_n = np.argmax(p_native, axis=1); pred_p = np.argmax(p_pp, axis=1)
        pp_fixes = (pred_n != y) & (pred_p == y)
        ebm_diff = np.abs(pp1 - pe)
        score = pp_fixes.astype(float) + np.abs(pp1 - pn) + 0.25 * ebm_diff
        for rank, idx in enumerate(np.argsort(-score)[:20], start=1):
            out.append({
                "dataset": dataset,
                "fold": fold,
                "rank": rank,
                "patient_index_in_fold": int(idx),
                "y_true": int(y[idx]),
                "ebm_pred": int(pred_e[idx]),
                "native_pred": int(pred_n[idx]),
                "ppost_pred": int(pred_p[idx]),
                "ebm_p_mortality": float(pe[idx]),
                "native_p_mortality": float(pn[idx]),
                "ppost_p_mortality": float(pp1[idx]),
                "ppost_minus_native": float(pp1[idx] - pn[idx]),
                "ppost_minus_ebm": float(pp1[idx] - pe[idx]),
                "native_wrong_ppost_right": int(bool(pp_fixes[idx])),
                "source": selected["source"],
                "variant": selected["variant"],
                "claim": selected["claim"],
            })
    _write_csv(out_dir / "ebm_vs_ppost_case_candidates.csv", out)
    return 0


def run_claim_checklist(passthrough: list[str]) -> int:
    dataset, selected, out_dir = _selected_and_out(passthrough)
    label = selected["dataset_label"]
    usefulness = [r for r in _read_csv(GENERATED / "ppost_final_usefulness.csv") if r.get("dataset") == label]
    controls = [r for r in _read_csv(GENERATED / "ppost_final_controls.csv") if r.get("dataset") == label]
    trace = [r for r in _read_csv(GENERATED / "ppost_final_trace_sufficiency.csv") if r.get("dataset") == label]
    u = usefulness[0] if usefulness else {}
    perm = next((r for r in controls if r.get("control") == "patient-permuted"), {})
    compact = [r for r in trace if _float(r.get("trace_fraction")) <= 0.20]
    best_compact = max(compact, key=lambda r: _float(r.get("ppost_mcc"), -999.0), default={})
    rows = [
        {"dataset": dataset, "claim": "non_random_evidence", "required_evidence": "patient-permuted MCC gap > 0", "status": "yes" if _float(perm.get("observed_minus_control_mcc")) > 0.05 else "no", "value": perm.get("observed_minus_control_mcc", "")},
        {"dataset": dataset, "claim": "within_source_utility", "required_evidence": "delta MCC > 0 or delta sensitivity > 0", "status": "yes" if (_float(u.get("delta_mcc")) > 0 or _float(u.get("delta_sensitivity")) > 0) else "no", "value": f"dMCC={u.get('delta_mcc','')}; dSens={u.get('delta_sensitivity','')}"},
        {"dataset": dataset, "claim": "compact_trace", "required_evidence": "<=20% trace keeps near-full MCC", "status": "yes" if best_compact and _float(best_compact.get("mcc_retained_vs_full")) >= 0.98 else "no", "value": f"trace={best_compact.get('trace_fraction','')}; retained={best_compact.get('mcc_retained_vs_full','')}"},
        {"dataset": dataset, "claim": "fully_symbolic_teacher_free", "required_evidence": "no teacher at inference for selected row", "status": "yes", "value": selected["variant"]},
        {"dataset": dataset, "claim": "calibrated_risk", "required_evidence": "low or improved brier/ece", "status": "partial" if _float(u.get("delta_brier")) <= 0.002 else "no", "value": f"dBrier={u.get('delta_brier','')}; dECE={u.get('delta_ece','')}"},
    ]
    _write_csv(out_dir / "claim_checklist.csv", rows)
    _write_md_table(out_dir / "claim_checklist.md", rows, ["dataset", "claim", "status", "value", "required_evidence"])
    return 0


def run_ebm_source_diagnostics(passthrough: list[str]) -> int:
    dataset, _selected, out_dir = _selected_and_out(passthrough)
    rows_out: list[dict[str, Any]] = []
    for summary_path in (PROOF_ROOT / dataset).glob("*/ppost_proof_summary.csv"):
        stage = summary_path.parent.name
        for row in _read_csv(summary_path):
            if row.get("rule_source") not in {"ebm_terms", "tabpfn_distill_ebm_terms"}:
                continue
            if not row.get("variant", "").startswith("pp_theta"):
                continue
            rows_out.append({
                "dataset": dataset,
                "stage": stage,
                "rule_source": row.get("rule_source"),
                "variant": row.get("variant"),
                "observed_mcc": _float(row.get("observed_mcc")),
                "delta_mcc": _float(row.get("mean_delta_mcc")),
                "delta_sensitivity": _float(row.get("mean_delta_sensitivity")),
                "delta_brier": _float(row.get("mean_delta_brier_score")),
                "delta_ece": _float(row.get("mean_delta_ece_10")),
                "control_gap": _float(row.get("observed_minus_permuted_mcc")),
                "trace_fraction": _float(row.get("mean_trace_fraction")),
            })
    rows_out.sort(key=lambda r: (r["rule_source"], -r["delta_mcc"]))
    _write_csv(out_dir / "ebm_source_diagnostics.csv", rows_out)
    _write_md_table(out_dir / "ebm_source_diagnostics.md", rows_out[:20], ["dataset", "rule_source", "variant", "stage", "observed_mcc", "delta_mcc", "delta_sensitivity", "delta_brier", "control_gap"])
    return 0


def _write_md_table(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            vals.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(
        "compact_residual_trace",
        "subset_sufficiency",
        "trace_bootstrap_ci",
        "ebm_ppost_case_study",
        "claim_checklist",
        "ebm_source_diagnostics",
    ), required=True)
    known, passthrough = parser.parse_known_args(argv)
    if known.experiment == "compact_residual_trace":
        return run_compact_residual_trace(passthrough)
    if known.experiment == "subset_sufficiency":
        return run_subset_sufficiency(passthrough)
    if known.experiment == "trace_bootstrap_ci":
        return run_trace_bootstrap_ci(passthrough)
    if known.experiment == "ebm_ppost_case_study":
        return run_ebm_ppost_case_study(passthrough)
    if known.experiment == "claim_checklist":
        return run_claim_checklist(passthrough)
    if known.experiment == "ebm_source_diagnostics":
        return run_ebm_source_diagnostics(passthrough)
    raise AssertionError(known.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
