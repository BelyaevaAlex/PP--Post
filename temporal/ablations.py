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

Optional TabPFN-TS distillation track
-------------------------------------
Pass ``--include-tabpfn-ts-distill`` to add temporal analogues of the
tabular TabPFN-distill rule sources: a black-box TabPFN-TS teacher
produces soft labels, then XGB / ExtraTrees / CatBoost students are
trained on ordinary L2/L3 temporal features and converted into
PPtheta-Post branches.
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
    _cleanup_memory,
    _evaluate,
    run_baseline,
    run_tabpfn_ts_distill,
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


def _log(message: str) -> None:
    print(f"[temporal_ablations {time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def _short_metrics(result: FoldResult) -> str:
    return (
        f"acc={result.accuracy:.4f} f1w={result.f1_weighted:.4f} "
        f"mcc={result.mcc:.4f} roc={result.roc_auc:.4f} "
        f"pr={result.pr_auc:.4f} fit={result.fit_seconds:.1f}s "
        f"pred={result.predict_seconds:.2f}s"
    )


def _stratified_attention_subset(
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    n = len(y)
    cap = int(max_samples)
    if cap <= 0 or n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    pieces = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(round(cap * len(cls_idx) / n)))
        take = min(take, len(cls_idx))
        pieces.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(pieces)
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    rng.shuffle(idx)
    return idx


def run_l4_ablation_one_fold(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    var_names: Sequence[str], n_classes: int, seed: int, epochs: int,
    log_prefix: str = "",
    l4_batch_size: int = 256,
    attention_max_samples: int = 2048,
) -> Dict[str, FoldResult]:
    """Train a single PPThetaPostTemporal and evaluate every ablation
    variant on the same held-out validation set.
    """
    prefix = f"{log_prefix} " if log_prefix else ""
    out: Dict[str, FoldResult] = {}
    _log(
        f"{prefix}backbone fit start train={X_train_ts.shape} "
        f"val={X_val_ts.shape} epochs={epochs} batch={l4_batch_size}"
    )
    t0 = time.time()
    tbn = PPThetaPostTemporal(
        var_names=var_names, n_classes=n_classes,
        seed=seed, epochs=epochs,
    ).fit(
        X_train_ts, mask_train, y_train,
        x_val=(X_val_ts, mask_val, y_val),
    )
    fit_secs = time.time() - t0
    _log(
        f"{prefix}backbone fit done fit={fit_secs:.1f}s "
        f"branches={len(tbn.branches)}"
    )

    theta = build_theta_matrix(tbn.branches, n_classes)
    _log(f"{prefix}theta built shape={theta.shape}; variants={len(ABLATION_VARIANTS)}")
    attention_cache: Dict[str, np.ndarray] = {}

    try:
        for idx, variant in enumerate(ABLATION_VARIANTS, start=1):
            variant_name = variant["name"]
            _log(f"{prefix}variant {idx}/{len(ABLATION_VARIANTS)} {variant_name} start")
            if variant["temporal_mode"] == "attention":
                attn_mode = variant.get("attention_mode", "shared")
                if attn_mode not in attention_cache:
                    attn_idx = _stratified_attention_subset(
                        y_train, attention_max_samples, seed + idx,
                    )
                    _log(
                        f"{prefix}attention fit start mode={attn_mode} "
                        f"samples={len(attn_idx)}/{len(y_train)}"
                    )
                    t_attn = time.time()
                    tbn.fit_attention(
                        X_train_ts[attn_idx], mask_train[attn_idx], y_train[attn_idx],
                        theta=theta, mode=attn_mode, epochs=200, lr=0.05,
                    )
                    attention_cache[attn_mode] = tbn.attention.weights()
                    _log(
                        f"{prefix}attention fit done mode={attn_mode} "
                        f"elapsed={time.time() - t_attn:.1f}s"
                    )
                    del attn_idx
                    _cleanup_memory()
                attn_w = attention_cache[attn_mode]
            else:
                attn_w = None

            t1 = time.time()
            proba = tbn.predict_temporal_proba_batched(
                X_val_ts,
                mask_val,
                theta=theta,
                n_classes=n_classes,
                head=variant.get("head", "weighted_mean"),
                temporal_mode=variant["temporal_mode"],
                k=variant.get("k"),
                top_k_time=variant.get("top_k_time"),
                attention_weights=attn_w,
                batch_size=l4_batch_size,
            )
            out[variant_name] = _evaluate(
                y_val, proba, n_classes, fit_secs, time.time() - t1,
            )
            del proba
            _cleanup_memory()
            _log(f"{prefix}variant {variant_name} done {_short_metrics(out[variant_name])}")
        _log(f"{prefix}ablation fold done rows={len(out)}")
        return out
    finally:
        del attention_cache, theta, tbn
        _cleanup_memory()


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
        "--l4-batch-size",
        type=int,
        default=int(os.environ.get("TEMPORAL_L4_BATCH_SIZE", "256")),
        help="Patient batch size for memory-bounded L4 validation inference.",
    )
    p.add_argument(
        "--attention-max-samples",
        type=int,
        default=int(os.environ.get("TEMPORAL_ATTENTION_MAX_SAMPLES", "2048")),
        help="Max train patients used to fit attention weights safely.",
    )
    p.add_argument(
        "--include-tabpfn-ts-distill",
        action="store_true",
        help="Append TabPFN-TS -> XGB/ET/CB rule-student rows.",
    )
    p.add_argument(
        "--include-tabpfn-ts-baseline",
        action="store_true",
        help="Append standalone black-box TabPFN-TS baseline row.",
    )
    p.add_argument(
        "--tabpfn-ts-distill-levels", nargs="+", default=["L2", "L3"],
        help="Ordinary temporal feature levels used by distill students.",
    )
    p.add_argument(
        "--tabpfn-ts-distill-students", nargs="+",
        default=["xgb", "et", "cb"],
        help="Rule students for TabPFN-TS distillation: xgb et cb.",
    )
    p.add_argument(
        "--tabpfn-ts-teacher-head",
        choices=["tabpfn", "xgb", "extratrees", "logreg"],
        default="tabpfn",
        help="Classifier head used on TabPFN-TS representation to form soft labels.",
    )
    p.add_argument(
        "--tabpfn-classifier-model-path", default=None,
        help="Path to TabPFN classifier checkpoint for teacher head=tabpfn.",
    )
    p.add_argument(
        "--ts-teacher-backend",
        choices=["auto", "tabpfn_ts", "tabpfn", "extratrees"],
        default="tabpfn_ts",
        help="Backend for the black-box TabPFN-TS representation.",
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
        help="Number of windows for L2 distill students.",
    )
    p.add_argument(
        "--n-intervals", type=int, default=10,
        help="Number of intervals for L3 distill students.",
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
    if args.include_tabpfn_ts_distill:
        md.write(
            f"\nTabPFN-TS distill rows: {args.tabpfn_ts_distill_levels} | "
            f"students: {args.tabpfn_ts_distill_students} | "
            f"backend: {args.ts_teacher_backend} | "
            f"head: {args.tabpfn_ts_teacher_head}\n"
        )
    if args.include_tabpfn_ts_baseline:
        md.write("\nStandalone baseline: TabPFN-TS black-box classifier\n")

    all_per_fold: List[AblationFoldResult] = []

    for ds_name in args.datasets:
        _log(f"dataset start {ds_name}")
        print(f"\n=== dataset: {ds_name} ===", flush=True)
        X_ts, mask, y, var_names, dataset_name = load_temporal_dataset(ds_name)
        _log(
            f"dataset loaded {dataset_name} X_ts={X_ts.shape} "
            f"mask={mask.shape} y={y.shape} vars={len(var_names)}"
        )
        n_classes = int(np.unique(y).size)
        skf = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed,
        )

        per_variant: Dict[str, List[FoldResult]] = {}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_ts, y)):
            print(f"  fold {fold_idx + 1}/{args.folds}", flush=True)
            _log(f"{dataset_name} fold {fold_idx + 1}/{args.folds} split start")
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
                log_prefix=f"{dataset_name} fold {fold_idx + 1}/{args.folds}",
            )
            if args.include_tabpfn_ts_distill:
                _log(f"{dataset_name} fold {fold_idx + 1}/{args.folds} TabPFN-TS distill block start")
                for distill_level in args.tabpfn_ts_distill_levels:
                    for student in args.tabpfn_ts_distill_students:
                        _log(
                            f"{dataset_name} fold {fold_idx + 1}/{args.folds} "
                            f"distill {distill_level}/{student} start"
                        )
                        try:
                            before = set(results)
                            results.update(run_tabpfn_ts_distill(
                                X_tr_ts, mask_tr, y_tr,
                                X_va_ts, mask_va, y_va,
                                var_names=var_names, n_classes=n_classes,
                                seed=args.seed, epochs=args.epochs,
                                level=distill_level,
                                student=student,
                                n_windows=args.n_windows,
                                n_intervals=args.n_intervals,
                                teacher_backend=args.ts_teacher_backend,
                                teacher_max_rows=args.ts_teacher_max_rows,
                                teacher_model_path=args.ts_teacher_model_path,
                                teacher_device=args.ts_teacher_device,
                                teacher_n_estimators=args.ts_teacher_n_estimators,
                                teacher_num_workers=args.ts_teacher_workers,
                                teacher_head=args.tabpfn_ts_teacher_head,
                                classifier_model_path=(
                                    args.tabpfn_classifier_model_path
                                ),
                            ))
                            new_rows = sorted(set(results) - before)
                            _log(
                                f"{dataset_name} fold {fold_idx + 1}/{args.folds} "
                                f"distill {distill_level}/{student} done rows={new_rows}"
                            )
                        except (RuntimeError, ImportError, ValueError) as exc:
                            print(
                                "    [skipped] TabPFN-TS distill "
                                f"{distill_level}/{student}: {exc}",
                                flush=True,
                            )
                            _log(
                                f"{dataset_name} fold {fold_idx + 1}/{args.folds} "
                                f"distill {distill_level}/{student} skipped: {exc}"
                            )
                _log(f"{dataset_name} fold {fold_idx + 1}/{args.folds} TabPFN-TS distill block done")
            if args.include_tabpfn_ts_baseline:
                _log(f"{dataset_name} fold {fold_idx + 1}/{args.folds} TabPFN-TS baseline start")
                try:
                    before = set(results)
                    results.update(run_baseline(
                        "tabpfn_ts",
                        X_tr_ts, mask_tr, y_tr,
                        X_va_ts, mask_va, y_va,
                        n_classes=n_classes, seed=args.seed,
                        epochs=args.epochs,
                        ts_backend=args.ts_teacher_backend,
                        ts_max_rows=args.ts_teacher_max_rows,
                        ts_model_path=args.ts_teacher_model_path,
                        ts_device=args.ts_teacher_device,
                        ts_n_estimators=args.ts_teacher_n_estimators,
                        ts_num_workers=args.ts_teacher_workers,
                        head=args.tabpfn_ts_teacher_head,
                        classifier_model_path=args.tabpfn_classifier_model_path,
                        classifier_device=args.ts_teacher_device,
                        classifier_n_estimators=args.ts_teacher_n_estimators,
                    ))
                    _log(
                        f"{dataset_name} fold {fold_idx + 1}/{args.folds} "
                        f"TabPFN-TS baseline done rows={sorted(set(results) - before)}"
                    )
                except (RuntimeError, ImportError, ValueError) as exc:
                    print(f"    [skipped] TabPFN-TS baseline: {exc}", flush=True)
                    _log(
                        f"{dataset_name} fold {fold_idx + 1}/{args.folds} "
                        f"TabPFN-TS baseline skipped: {exc}"
                    )
            for variant_name, fold_result in results.items():
                per_variant.setdefault(variant_name, []).append(fold_result)
                all_per_fold.append(AblationFoldResult(
                    fold=fold_idx, variant=f"{dataset_name}__{variant_name}",
                    fold_result=fold_result,
                ))
            _write_csv(all_per_fold, csv_path)
            _log(
                f"{dataset_name} fold {fold_idx + 1}/{args.folds} "
                f"recorded rows={len(results)} partial_csv={csv_path}"
            )

        rows = {name: _aggregate(fl) for name, fl in per_variant.items()}
        md.write(f"\n## {dataset_name}\n")
        md.write(_format_md(rows))
        _write_summary_csv(rows, summary_path)
        _log(f"dataset done {dataset_name} variants={len(rows)} summary_csv={summary_path}")

    _write_csv(all_per_fold, csv_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.getvalue())
    print(f"\nMarkdown report: {md_path}", flush=True)
    print(f"Per-fold CSV:    {csv_path}", flush=True)
    print(f"Summary CSV:     {summary_path}", flush=True)
    _log(f"all done md={md_path} csv={csv_path} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
