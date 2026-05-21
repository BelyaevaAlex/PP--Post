"""§6.1 — temporal ablations for PPtheta-Post.

Main track: fix the per-timestep RuleNetwork backbone and
sweep over temporal aggregation modes and their hyper-parameters.

Unlike :mod:`compare_temporal`, this driver does **not** vary the
backbone (no L1 / L2 / L3 / extractor changes) — it trains a single
:class:`PPThetaPostTemporal` once per fold and then evaluates the same
``z_per_time`` tensor through every aggregation × hyper-parameter
combination registered in :data:`ABLATION_VARIANTS`.

Ablation axes
-------------
* aggregation mode      ∈ {mean, max, exists, forall, k_of_t, last, attention}
* ``k_of_t`` threshold  ∈ {0.1, 0.25, 0.5, 0.75}
* ``top_k_time`` filter ∈ {None, 0.05, 0.1, 0.25}
* ``attention_mode``    ∈ {shared, per_branch, multi_head}
* head                   ∈ {noisy_or, weighted_mean}

Output
------
* Markdown table at ``output/temporal/ablations_<timestamp>.md``
* CSV at ``output/temporal/ablations_<timestamp>.csv`` (one row per
  variant × fold).
* Final aggregated CSV ``output/temporal/ablations_<timestamp>_summary.csv``
  with mean ± std per variant.

Optional feature-teacher track
------------------------------
Pass ``--include-ts-feature-teacher`` to add L2T/L3T variants where
TabPFN-style forecasting/residual features are distilled into the flat
PPtheta-Post student.  This mirrors the tabular TabPFN-distill recipe:
the teacher guides the feature space used to grow branches, while the
RuleNetwork/PPtheta-Post head is trained against the original labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from problog_inference import build_theta_matrix  # noqa: E402  pylint: disable=wrong-import-position

from .compare_temporal import (  # noqa: E402
    AggregatedRow,
    FoldResult,
    _aggregate,
    _evaluate,
    run_l2_ts_teacher,
    run_l3_ts_teacher,
)
from .datasets import load_temporal_dataset  # noqa: E402
from .pp_theta_post_temporal import PPThetaPostTemporal  # noqa: E402
from .temporal_inference import TemporalProbLogClassifier  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Ablation grid
# ─────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS: List[dict] = []

# Pure aggregation modes (no extra hyper-parameters):
for mode in ("mean", "max", "last"):
    ABLATION_VARIANTS.append({
        "name": f"agg_{mode}_wmean",
        "temporal_mode": mode,
        "head": "weighted_mean",
    })

# k-of-T sweep:
for k in (0.10, 0.25, 0.50, 0.75):
    ABLATION_VARIANTS.append({
        "name": f"agg_kOfT-{int(k * 100)}_wmean",
        "temporal_mode": "k_of_t",
        "k": k,
        "head": "weighted_mean",
    })

# Existential / forall + top-k-time filter sweep:
for top_k in (None, 0.05, 0.10, 0.25):
    suffix = "all" if top_k is None else f"top{int(top_k * 100)}"
    ABLATION_VARIANTS.append({
        "name": f"agg_exists-{suffix}_noisy_or",
        "temporal_mode": "exists",
        "head": "noisy_or",
        "top_k_time": top_k,
    })

# Attention modes (only `shared` available without n_branches; per-branch
# and multi-head require fitting):
for attn_mode in ("shared", "per_branch", "multi_head"):
    ABLATION_VARIANTS.append({
        "name": f"agg_attn-{attn_mode}_wmean",
        "temporal_mode": "attention",
        "attention_mode": attn_mode,
        "head": "weighted_mean",
    })


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class AblationFoldResult:
    fold: int
    variant: str
    fold_result: FoldResult


def run_l4_ablation_one_fold(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    var_names: Sequence[str], n_classes: int, seed: int, epochs: int,
) -> Dict[str, FoldResult]:
    """Train a single PPThetaPostTemporal and evaluate every ablation
    variant on the same held-out validation set.
    """
    out: Dict[str, FoldResult] = {}
    t0 = time.time()
    tbn = PPThetaPostTemporal(
        var_names=var_names, n_classes=n_classes,
        seed=seed, epochs=epochs,
    ).fit(
        X_train_ts, mask_train, y_train,
        x_val=(X_val_ts, mask_val, y_val),
    )
    fit_secs = time.time() - t0

    z_val = tbn.predict_branch_probs_per_time(X_val_ts, mask_val)
    theta = build_theta_matrix(tbn.branches, n_classes)
    attention_cache: Dict[str, np.ndarray] = {}

    for variant in ABLATION_VARIANTS:
        if variant["temporal_mode"] == "attention":
            attn_mode = variant.get("attention_mode", "shared")
            if attn_mode not in attention_cache:
                tbn.fit_attention(
                    X_train_ts, mask_train, y_train, theta=theta,
                    mode=attn_mode, epochs=200, lr=0.05,
                )
                attention_cache[attn_mode] = tbn.attention.weights()
            attn_w = attention_cache[attn_mode]
        else:
            attn_w = None

        clf = TemporalProbLogClassifier(
            branches=tbn.branches,
            n_classes=n_classes,
            head=variant.get("head", "weighted_mean"),
            temporal_mode=variant["temporal_mode"],
            k=variant.get("k"),
            top_k_time=variant.get("top_k_time"),
            theta=theta,
        )
        t1 = time.time()
        proba = clf.predict_proba(z_val, attention_weights=attn_w)
        out[variant["name"]] = _evaluate(
            y_val, proba, n_classes, fit_secs, time.time() - t1,
        )
    return out


def _format_md(rows: Dict[str, AggregatedRow]) -> str:
    out = StringIO()
    out.write("\n| Variant | Acc | F1 | MCC | ROC AUC | PR AUC | fit (s) | pred (s) |\n")
    out.write("|---|---|---|---|---|---|---|---|\n")
    for name, agg in rows.items():
        m = agg.metric_means
        s = agg.metric_stds
        out.write(
            f"| {name} "
            f"| {m['accuracy']:.3f}±{s['accuracy']:.3f} "
            f"| {m['f1_weighted']:.3f}±{s['f1_weighted']:.3f} "
            f"| {m['mcc']:.3f}±{s['mcc']:.3f} "
            f"| {m['roc_auc']:.3f}±{s['roc_auc']:.3f} "
            f"| {m['pr_auc']:.3f}±{s['pr_auc']:.3f} "
            f"| {m['fit_seconds']:.1f} "
            f"| {m['predict_seconds']:.2f} "
            "|\n"
        )
    return out.getvalue()


def _write_csv(per_fold_rows: List[AblationFoldResult], path: str) -> None:
    keys = ["accuracy", "f1_weighted", "mcc", "roc_auc", "pr_auc",
            "fit_seconds", "predict_seconds"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "variant"] + keys)
        for r in per_fold_rows:
            row = [r.fold, r.variant] + [
                getattr(r.fold_result, k) for k in keys
            ]
            w.writerow(row)


def _write_summary_csv(rows: Dict[str, AggregatedRow], path: str) -> None:
    keys = ["accuracy", "f1_weighted", "mcc", "roc_auc", "pr_auc",
            "fit_seconds", "predict_seconds"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["variant"]
        for k in keys:
            header += [f"{k}_mean", f"{k}_std"]
        w.writerow(header)
        for name, agg in rows.items():
            row = [name]
            for k in keys:
                row += [agg.metric_means[k], agg.metric_stds[k]]
            w.writerow(row)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L4 PPθ-Post-Temporal ablations")
    p.add_argument("--datasets", nargs="+",
                   default=["pam"],
                   help="Subset of registered temporal datasets to run.")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include-ts-feature-teacher",
        action="store_true",
        help="Append L2T/L3T TabPFN-style temporal feature-teacher "
             "ablation rows alongside the fixed-L4 aggregation sweep.",
    )
    p.add_argument(
        "--ts-teacher-levels", nargs="+", default=["L2T", "L3T"],
        help="Feature-teacher levels to run when "
             "--include-ts-feature-teacher is set.",
    )
    p.add_argument(
        "--ts-teacher-backend",
        choices=["auto", "tabpfn_ts", "tabpfn", "extratrees"],
        default="tabpfn_ts",
        help="Backend for L2T/L3T feature-teacher variants.",
    )
    p.add_argument(
        "--ts-teacher-max-rows", type=int, default=4096,
        help="Maximum transition rows used by the ExtraTrees teacher.",
    )
    p.add_argument(
        "--ts-teacher-model-path", default=None,
        help="Path to the downloaded TabPFN-TS checkpoint.",
    )
    p.add_argument(
        "--ts-teacher-device", default="cpu",
        help="Device passed to TabPFN-TS LOCAL mode.",
    )
    p.add_argument(
        "--ts-teacher-n-estimators", type=int, default=8,
        help="Number of TabPFN estimators used by the TabPFN-TS teacher.",
    )
    p.add_argument(
        "--ts-teacher-workers", type=int, default=1,
        help="CPU worker count inside tabpfn_time_series.",
    )
    p.add_argument(
        "--n-windows", type=int, default=4,
        help="Number of windows for L2T when feature-teacher rows run.",
    )
    p.add_argument(
        "--n-intervals", type=int, default=10,
        help="Number of intervals for L3T when feature-teacher rows run.",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(THIS_DIR, "..", "output", "temporal"),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(args.output_dir, f"ablations_{timestamp}.md")
    csv_path = os.path.join(args.output_dir, f"ablations_{timestamp}.csv")
    summary_path = os.path.join(
        args.output_dir, f"ablations_{timestamp}_summary.csv",
    )

    md = StringIO()
    md.write(f"# PPθ-Post-Temporal L4 ablations — {timestamp}\n\n")
    md.write(
        "**Backbone fixed**: per-timestep RuleNetwork trained once per fold; "
        "only the temporal aggregation / hyper-parameters vary.\n\n"
    )
    md.write(f"Datasets: {args.datasets} | folds: {args.folds} | "
             f"epochs: {args.epochs}\n")
    if args.include_ts_feature_teacher:
        md.write(
            f"\nFeature-teacher rows: {args.ts_teacher_levels} | "
            f"backend: {args.ts_teacher_backend} | "
            f"max_rows: {args.ts_teacher_max_rows}\n"
        )

    all_per_fold: List[AblationFoldResult] = []

    for ds_name in args.datasets:
        print(f"\n=== dataset: {ds_name} ===")
        X_ts, mask, y, var_names, dataset_name = load_temporal_dataset(ds_name)
        n_classes = int(np.unique(y).size)
        skf = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed,
        )

        per_variant: Dict[str, List[FoldResult]] = {}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_ts, y)):
            print(f"  fold {fold_idx + 1}/{args.folds}")
            X_tr_ts, mask_tr, y_tr = (
                X_ts[train_idx], mask[train_idx], y[train_idx]
            )
            X_va_ts, mask_va, y_va = (
                X_ts[val_idx], mask[val_idx], y[val_idx]
            )
            results = run_l4_ablation_one_fold(
                X_tr_ts, mask_tr, y_tr,
                X_va_ts, mask_va, y_va,
                var_names=var_names, n_classes=n_classes,
                seed=args.seed, epochs=args.epochs,
            )
            if args.include_ts_feature_teacher:
                teacher_levels = {
                    str(level).upper() for level in args.ts_teacher_levels
                }
                if "L2T" in teacher_levels:
                    try:
                        results.update(run_l2_ts_teacher(
                            X_tr_ts, mask_tr, y_tr,
                            X_va_ts, mask_va, y_va,
                            n_classes=n_classes, seed=args.seed,
                            epochs=args.epochs, n_windows=args.n_windows,
                            teacher_backend=args.ts_teacher_backend,
                            teacher_max_rows=args.ts_teacher_max_rows,
                            teacher_model_path=args.ts_teacher_model_path,
                            teacher_device=args.ts_teacher_device,
                            teacher_n_estimators=args.ts_teacher_n_estimators,
                            teacher_num_workers=args.ts_teacher_workers,
                        ))
                    except (RuntimeError, ImportError) as exc:
                        print(f"    [skipped] L2T feature teacher: {exc}")
                if "L3T" in teacher_levels:
                    try:
                        results.update(run_l3_ts_teacher(
                            X_tr_ts, mask_tr, y_tr,
                            X_va_ts, mask_va, y_va,
                            var_names=var_names, n_classes=n_classes,
                            seed=args.seed, epochs=args.epochs,
                            n_intervals=args.n_intervals,
                            teacher_backend=args.ts_teacher_backend,
                            teacher_max_rows=args.ts_teacher_max_rows,
                            teacher_model_path=args.ts_teacher_model_path,
                            teacher_device=args.ts_teacher_device,
                            teacher_n_estimators=args.ts_teacher_n_estimators,
                            teacher_num_workers=args.ts_teacher_workers,
                        ))
                    except (RuntimeError, ImportError) as exc:
                        print(f"    [skipped] L3T feature teacher: {exc}")
            for variant_name, fold_result in results.items():
                per_variant.setdefault(variant_name, []).append(fold_result)
                all_per_fold.append(AblationFoldResult(
                    fold=fold_idx, variant=f"{dataset_name}__{variant_name}",
                    fold_result=fold_result,
                ))

        rows = {name: _aggregate(fl) for name, fl in per_variant.items()}
        md.write(f"\n## {dataset_name}\n")
        md.write(_format_md(rows))
        _write_summary_csv(rows, summary_path)

    _write_csv(all_per_fold, csv_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.getvalue())
    print(f"\nMarkdown report: {md_path}")
    print(f"Per-fold CSV:    {csv_path}")
    print(f"Summary CSV:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
