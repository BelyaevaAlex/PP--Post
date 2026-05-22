"""§5.1 — intra-method ablation driver across the four temporal levels
of PPθ-Post.

Scope
-----
This is an **intra-method comparison only**: the four PPθ-Post variants
(L1 / L2 / L3 / L4) are evaluated against each other on the same
temporal benchmark.  External baselines (LR / XGB / GRU-D /
Transformer / SAnD / mTAN / Raindrop / SeFT / CAMELOT /
InterpGN) are added via the `--baselines` flag; SOTA rows use the
*authors' original code* from `temporal/vendor/*` (vendored is the
default and only track for SOTA — see :mod:`temporal.baselines_vendored`
and :mod:`temporal.baselines_vendored_tf`).

Sanity-check disclaimer
-----------------------
Numbers reported on the bundled synthetic loaders (``p12`` / ``pam`` /
``mimic3``) are a **smoke-level sanity check** — they confirm that the
pipeline works end-to-end and that variant ranking is sensible on
toy data.  **For paper-quality numbers, run on credentialed real
datasets (PhysioNet/2012, PAMAP2, MIMIC-III/IV) using the same CLI
flags.**  See ``PAPER_LAYOUT.md`` for the full reproduction plan.

Usage
-----
::

    python -m temporal.compare_temporal \\
        --datasets p12 pam mimic3 \\
        --levels L1 L2 L3 L4 \\
        --include-tabpfn-ts-distill \\
        --folds 3 --epochs 80

The TabPFN-TS distillation rows are explicit teacher-student rule-source
ablations: the black-box TabPFN-TS teacher produces soft labels, then
XGB / ExtraTrees / CatBoost students are trained on ordinary L2/L3
temporal features and converted into PPθ-Post branches.

The script writes a Markdown summary table to
``output/temporal/<datestamp>.md`` and a per-row CSV with mean ± std.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from rule_network_model import RuleNetworkModel  # noqa: E402
from problog_inference import (  # noqa: E402
    ProbLogClassifier,
    aggregate_weighted_mean,
    build_theta_matrix,
)

from .datasets import load_temporal_dataset  # noqa: E402
from .interval_forest import (  # noqa: E402
    fit_interval_forest,
    IntervalFeatureExtractor,
)
from .baselines import (  # noqa: E402
    BASELINE_REGISTRY,
    BaselineBase,
    make_baseline,
)
from .baselines_vendored import (  # noqa: E402
    VENDORED_REGISTRY,
    VendoredInterpGNBaseline,
    make_vendored,
)
from .baselines_vendored_tf import (  # noqa: E402
    VENDORED_TF_REGISTRY,
    make_vendored_tf,
)
from .pp_theta_post_temporal import PPThetaPostTemporal  # noqa: E402
from .tabpfn_ts_distill import (  # noqa: E402
    TabPFNTSClassifierTeacher,
    fit_distilled_rule_student,
    student_label,
)
from .tabularize import multi_window_flatten, summary_flatten  # noqa: E402
from .temporal_inference import (  # noqa: E402
    DEFAULT_TEMPORAL_VARIANTS,
    TemporalProbLogClassifier,
)


# ─────────────────────────────────────────────────────────────────────────
# Metric utilities
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    accuracy: float = 0.0
    f1_weighted: float = 0.0
    mcc: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        scores = []
        for k in range(n_classes):
            mask = (y_true == k).astype(int)
            if mask.sum() == 0:
                continue
            scores.append(average_precision_score(mask, proba[:, k]))
        return float(np.mean(scores)) if scores else float("nan")
    except Exception:
        return float("nan")


def _evaluate(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
    fit_seconds: float,
    predict_seconds: float,
) -> FoldResult:
    pred = np.argmax(proba, axis=1)
    return FoldResult(
        accuracy=float(accuracy_score(y_true, pred)),
        f1_weighted=float(f1_score(y_true, pred, average="weighted",
                                    zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, pred)),
        roc_auc=_safe_roc_auc(y_true, proba, n_classes),
        pr_auc=_safe_pr_auc(y_true, proba, n_classes),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-level pipeline runners
# ─────────────────────────────────────────────────────────────────────────

def _train_rule_network_static(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    n_classes: int,
) -> RuleNetworkModel:
    log2_d = int(np.floor(np.log2(max(X_train.shape[1], 2))))
    n_est = max(2, n_classes + log2_d)
    max_leaves = 2 ** (log2_d + 4)
    forest = ExtraTreesClassifier(
        n_estimators=n_est,
        max_leaf_nodes=max_leaves,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    model = RuleNetworkModel()
    model.build_model_from_ensemble(forest)
    model.fit(
        X_train.astype(np.float32),
        y_train.astype(np.int64),
        X_val.astype(np.float32),
        y_val.astype(np.int64),
        learning_rate=0.01,
        epochs=epochs,
    )
    return model


def _run_static_levels_on_features(
    label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
) -> Dict[str, FoldResult]:
    """Train RuleNetwork on a flat feature matrix and report results for
    the standard PPθ-Post inference variants used in this comparison.
    """
    out: Dict[str, FoldResult] = {}
    t0 = time.time()
    model = _train_rule_network_static(
        X_train, y_train, X_val, y_val, seed, epochs, n_classes,
    )
    fit_secs = time.time() - t0

    bp_val = model.predict_branch_proba(X_val).numpy()
    branches = model.branches

    t0 = time.time()
    proba_neural = model.predict_proba(X_val).numpy()
    out[f"{label}_Neural"] = _evaluate(
        y_val, proba_neural, n_classes, fit_secs, time.time() - t0,
    )

    for variant in (
        ("PL-fast", "fast"),
        ("PL-full", "full"),
    ):
        name, mode = variant
        clf = ProbLogClassifier(
            branches=branches, n_classes=n_classes, mode=mode,
        )
        t0 = time.time()
        proba = clf.predict_proba(bp_val, X_val, verbose=False)
        out[f"{label}_{name}"] = _evaluate(
            y_val, proba, n_classes, fit_secs, time.time() - t0,
        )

    theta = build_theta_matrix(branches, n_classes)
    t0 = time.time()
    proba_wmean = aggregate_weighted_mean(bp_val, theta)
    out[f"{label}_PL-wmean"] = _evaluate(
        y_val, proba_wmean, n_classes, fit_secs, time.time() - t0,
    )
    return out


def run_l1(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, seed: int, epochs: int,
) -> Dict[str, FoldResult]:
    X_train = summary_flatten(X_train_ts, mask_train)
    X_val = summary_flatten(X_val_ts, mask_val)
    return _run_static_levels_on_features(
        "L1", X_train, y_train, X_val, y_val, n_classes, seed, epochs,
    )


def run_l2(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, seed: int, epochs: int, n_windows: int,
) -> Dict[str, FoldResult]:
    X_train = multi_window_flatten(X_train_ts, mask_train, n_windows=n_windows)
    X_val = multi_window_flatten(X_val_ts, mask_val, n_windows=n_windows)
    return _run_static_levels_on_features(
        "L2", X_train, y_train, X_val, y_val, n_classes, seed, epochs,
    )


def run_l3(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    var_names: Sequence[str], n_classes: int, seed: int, epochs: int,
    n_intervals: int,
) -> Dict[str, FoldResult]:
    extractor = IntervalFeatureExtractor(
        var_names=var_names,
        T=X_train_ts.shape[1],
        n_intervals=n_intervals,
        seed=seed,
    )
    X_train = extractor.transform(X_train_ts, mask_train)
    X_val = extractor.transform(X_val_ts, mask_val)
    return _run_static_levels_on_features(
        "L3", X_train, y_train, X_val, y_val, n_classes, seed, epochs,
    )


def _run_static_levels_on_branches(
    label: str,
    branches_per_tree,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    native_model=None,
) -> Dict[str, FoldResult]:
    out: Dict[str, FoldResult] = {}
    t0 = time.time()
    model = RuleNetworkModel()
    model.build_model_from_branches(
        branches_per_tree,
        in_features=int(X_train.shape[1]),
        out_features=int(n_classes),
    )
    model.fit(
        X_train.astype(np.float32),
        y_train.astype(np.int64),
        X_val.astype(np.float32),
        y_val.astype(np.int64),
        learning_rate=0.01,
        epochs=epochs,
    )
    fit_secs = time.time() - t0

    if native_model is not None and hasattr(native_model, "predict_proba"):
        t0 = time.time()
        proba_native = np.asarray(native_model.predict_proba(X_val), dtype=np.float64)
        if proba_native.ndim == 1:
            proba_native = np.column_stack([1.0 - proba_native, proba_native])
        if proba_native.shape[1] < n_classes:
            full = np.zeros((proba_native.shape[0], n_classes), dtype=np.float64)
            classes = getattr(native_model, "classes_", np.arange(proba_native.shape[1]))
            for col, cls in enumerate(classes):
                if 0 <= int(cls) < n_classes:
                    full[:, int(cls)] = proba_native[:, col]
            proba_native = full
        out[f"{label}_NativeStudent"] = _evaluate(
            y_val, proba_native, n_classes, fit_secs, time.time() - t0,
        )

    bp_val = model.predict_branch_proba(X_val).numpy()
    branches = model.branches

    t0 = time.time()
    proba_neural = model.predict_proba(X_val).numpy()
    out[f"{label}_Neural"] = _evaluate(
        y_val, proba_neural, n_classes, fit_secs, time.time() - t0,
    )

    for name, mode in (("PL-fast", "fast"), ("PL-full", "full")):
        clf = ProbLogClassifier(
            branches=branches, n_classes=n_classes, mode=mode,
        )
        t0 = time.time()
        proba = clf.predict_proba(bp_val, X_val, verbose=False)
        out[f"{label}_{name}"] = _evaluate(
            y_val, proba, n_classes, fit_secs, time.time() - t0,
        )

    theta = build_theta_matrix(branches, n_classes)
    t0 = time.time()
    proba_wmean = aggregate_weighted_mean(bp_val, theta)
    out[f"{label}_PL-wmean"] = _evaluate(
        y_val, proba_wmean, n_classes, fit_secs, time.time() - t0,
    )
    return out


def _temporal_student_features(
    level: str,
    X_train_ts: np.ndarray,
    mask_train: np.ndarray,
    X_val_ts: np.ndarray,
    mask_val: np.ndarray,
    *,
    var_names: Sequence[str],
    seed: int,
    n_windows: int,
    n_intervals: int,
) -> Tuple[np.ndarray, np.ndarray]:
    key = level.upper()
    if key == "L2":
        return (
            multi_window_flatten(X_train_ts, mask_train, n_windows=n_windows),
            multi_window_flatten(X_val_ts, mask_val, n_windows=n_windows),
        )
    if key == "L3":
        extractor = IntervalFeatureExtractor(
            var_names=var_names,
            T=X_train_ts.shape[1],
            n_intervals=n_intervals,
            seed=seed,
        )
        return (
            extractor.transform(X_train_ts, mask_train),
            extractor.transform(X_val_ts, mask_val),
        )
    raise ValueError("TabPFN-TS distillation supports student levels L2 and L3")


def run_tabpfn_ts_distill(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    var_names: Sequence[str], n_classes: int, seed: int, epochs: int,
    level: str, student: str, n_windows: int, n_intervals: int,
    teacher_backend: str = "tabpfn_ts",
    teacher_max_rows: int = 4096,
    teacher_model_path: Optional[str] = None,
    teacher_device: str = "cpu",
    teacher_n_estimators: int = 8,
    teacher_num_workers: int = 1,
    teacher_head: str = "tabpfn",
    classifier_model_path: Optional[str] = None,
) -> Dict[str, FoldResult]:
    """Temporal analogue of tabular TabPFN-distill.

    The black-box TabPFN-TS teacher produces soft labels from the raw time
    series.  The rule student is trained on ordinary L2/L3 temporal
    features, then converted into PPtheta-Post branches.
    """
    X_train, X_val = _temporal_student_features(
        level,
        X_train_ts,
        mask_train,
        X_val_ts,
        mask_val,
        var_names=var_names,
        seed=seed,
        n_windows=n_windows,
        n_intervals=n_intervals,
    )
    teacher = TabPFNTSClassifierTeacher(
        n_classes=n_classes,
        seed=seed,
        ts_backend=teacher_backend,
        ts_max_rows=teacher_max_rows,
        ts_model_path=teacher_model_path,
        ts_device=teacher_device,
        ts_n_estimators=teacher_n_estimators,
        ts_num_workers=teacher_num_workers,
        head=teacher_head,
        classifier_model_path=classifier_model_path,
        classifier_device=teacher_device,
        classifier_n_estimators=teacher_n_estimators,
    ).fit(X_train_ts, mask_train, y_train)
    p_soft = teacher.predict_proba(X_train_ts, mask_train)
    fitted_student = fit_distilled_rule_student(
        X_train,
        y_train,
        p_soft,
        n_classes=n_classes,
        seed=seed,
        student=student,
    )
    label = (
        f"{level.upper()}-TabPFNTS-"
        f"Distill{student_label(student)}"
    )
    return _run_static_levels_on_branches(
        label,
        fitted_student.branches_per_tree,
        X_train,
        y_train,
        X_val,
        y_val,
        n_classes,
        seed,
        epochs,
        native_model=fitted_student.native_model,
    )


def run_l4(
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    var_names: Sequence[str], n_classes: int, seed: int, epochs: int,
    variants: Sequence[dict],
) -> Dict[str, FoldResult]:
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

    # Cache attention weights per ``attention_mode`` so that variants
    # sharing the same mode are not retrained from scratch.
    attention_cache: Dict[str, np.ndarray] = {}

    for variant in variants:
        if variant["temporal_mode"] == "attention":
            attn_mode = variant.get("attention_mode", "shared")
            if attn_mode not in attention_cache:
                tbn.fit_attention(
                    X_train_ts, mask_train, y_train, theta=theta,
                    mode=attn_mode, epochs=200, lr=0.05,
                )
                attention_cache[attn_mode] = (
                    tbn.attention.weights() if tbn.attention is not None else None
                )
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
        t0 = time.time()
        proba = clf.predict_proba(z_val, attention_weights=attn_w)
        out[f"L4_{variant['name']}"] = _evaluate(
            y_val, proba, n_classes, fit_secs, time.time() - t0,
        )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class AggregatedRow:
    metric_means: Dict[str, float] = field(default_factory=dict)
    metric_stds: Dict[str, float] = field(default_factory=dict)


def _aggregate(per_fold: List[FoldResult]) -> AggregatedRow:
    keys = ["accuracy", "f1_weighted", "mcc", "roc_auc", "pr_auc",
            "fit_seconds", "predict_seconds"]
    means = {k: float(np.mean([getattr(r, k) for r in per_fold])) for k in keys}
    stds = {k: float(np.std([getattr(r, k) for r in per_fold])) for k in keys}
    return AggregatedRow(metric_means=means, metric_stds=stds)


def _format_table(
    dataset_name: str,
    rows: Dict[str, AggregatedRow],
) -> str:
    out = StringIO()
    out.write(f"\n## {dataset_name}\n\n")
    out.write("| Variant | Acc | F1 | MCC | ROC AUC | PR AUC | fit (s) | pred (s) |\n")
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


UNIFIED_BASELINE_REGISTRY: Dict[str, type] = {
    **BASELINE_REGISTRY,        # local rows incl. TabPFN-TS black-box baseline
    **VENDORED_REGISTRY,        # 5 PyTorch vendored (sand, mtan, gru_d, raindrop, interp_gn)
    **VENDORED_TF_REGISTRY,     # 2 TF vendored (seft, camelot)
}


def make_unified_baseline(
    name: str, n_classes: int, **kwargs,
) -> BaselineBase:
    """Single-entry-point factory that prefers vendored adapters
    (authors' original code) and falls back to the local
    local baseline registry only for rows that have no upstream worth
    vendoring, plus the standalone ``tabpfn_ts`` black-box baseline.

    Raises :class:`RuntimeError` if a vendored SOTA adapter cannot be
    initialised (missing TensorFlow / CUDA / ``torch_geometric`` / …) —
    the driver in :func:`main` catches this and skips the row, so the
    final report only contains rows that actually ran.
    """
    key = name.lower()
    if key in VENDORED_REGISTRY:
        return make_vendored(key, n_classes=n_classes, **kwargs)
    if key in VENDORED_TF_REGISTRY:
        return make_vendored_tf(key, n_classes=n_classes, **kwargs)
    if key in BASELINE_REGISTRY:
        return make_baseline(key, n_classes=n_classes, **kwargs)
    raise KeyError(
        f"unknown baseline {name!r}; choose from "
        f"{sorted(UNIFIED_BASELINE_REGISTRY)}"
    )


def run_baseline(
    name: str,
    X_train_ts: np.ndarray, mask_train: np.ndarray, y_train: np.ndarray,
    X_val_ts: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, seed: int, epochs: int,
    **baseline_kwargs,
) -> Dict[str, FoldResult]:
    """Train a single external baseline and evaluate on the held-out set.

    SOTA baselines are served exclusively from the vendored adapters
    (authors' original code); only ``lr`` / ``xgb`` / ``transformer``
    remain as local re-implementations.  If the vendored adapter cannot
    initialise (missing TF / CUDA / ``torch_geometric`` / configs) the
    caller (:func:`main`) catches the resulting :class:`RuntimeError`
    and emits a ``[skipped]`` line — there is *no* re-implementation
    fallback for SOTA baselines.

    Returns a one-key dict ``{label: FoldResult}`` so the result
    integrates seamlessly with the L1-L4 reporting pipeline.
    """
    cls = UNIFIED_BASELINE_REGISTRY[name]
    kwargs = {"n_classes": n_classes, "seed": seed}
    kwargs.update(baseline_kwargs)
    if "epochs" in getattr(cls, "__dataclass_fields__", {}):
        kwargs["epochs"] = epochs
    baseline = make_unified_baseline(name, **kwargs)

    t0 = time.time()
    if getattr(baseline, "needs_torch", False):
        try:
            baseline.fit(
                X_train_ts, mask_train, y_train,
                x_val=(X_val_ts, mask_val, y_val),
            )
        except TypeError:
            # Some adapters (e.g. CAMELOT / vendored ones) don't accept x_val.
            baseline.fit(X_train_ts, mask_train, y_train)
    else:
        baseline.fit(X_train_ts, mask_train, y_train)
    fit_secs = time.time() - t0

    t1 = time.time()
    proba = baseline.predict_proba(X_val_ts, mask_val)
    predict_secs = time.time() - t1

    label = f"BL_{baseline.name}"
    if isinstance(baseline, VendoredInterpGNBaseline):
        try:
            print(f"    [InterpGN] mean gate routing "
                  f"g={baseline.mean_gate_routing:.3f} "
                  f"(higher = more reliance on the opaque neural path)")
        except Exception:                                       # pragma: no cover
            pass

    return {label: _evaluate(y_val, proba, n_classes, fit_secs, predict_secs)}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPθ-Post temporal comparison driver "
                    "(intra-method ablation L1–L4 + optional external baselines)",
    )
    p.add_argument("--datasets", nargs="+",
                   default=["p12", "pam", "mimic3"],
                   help="Subset of registered temporal datasets to run.")
    p.add_argument("--levels", nargs="+",
                   default=["L1", "L2", "L3", "L4"],
                   help="Subset of PPtheta-Post temporal levels to evaluate.")
    p.add_argument(
        "--baselines", nargs="*", default=[],
        help="External baselines to add alongside L1-L4. "
             "Choices: " + ", ".join(sorted(UNIFIED_BASELINE_REGISTRY)) +
             ".  Pass `all` to enable every registered baseline.  SOTA "
             "rows use the *authors' original code* from "
             "temporal/vendor/* (vendored is the default and only "
             "track for SOTA).  Adapters that cannot initialise "
             "(missing TensorFlow / CUDA / torch_geometric / configs) "
             "are skipped — no re-implementation fallback exists.",
    )
    p.add_argument("--folds", type=int, default=3,
                   help="Number of stratified CV folds.")
    p.add_argument("--epochs", type=int, default=60,
                   help="RuleNetwork training epochs per fold.")
    p.add_argument("--n-windows", type=int, default=4,
                   help="Number of windows for L2.")
    p.add_argument("--n-intervals", type=int, default=10,
                   help="Number of random intervals for L3.")
    p.add_argument(
        "--ts-teacher-backend",
        choices=["auto", "tabpfn_ts", "tabpfn", "extratrees"],
        default="tabpfn_ts",
        help="Backend for the black-box TabPFN-TS representation. "
             "`tabpfn_ts` uses tabpfn_time_series.TabPFNTSPipeline "
             "with local Prior-Labs/tabpfn_3 weights; "
             "`extratrees` is a local smoke-test backend; `auto` tries "
             "TabPFN and falls back to ExtraTrees.",
    )
    p.add_argument(
        "--ts-teacher-max-rows", type=int, default=4096,
        help="Maximum lag-to-next transition rows used to fit the "
             "ExtraTrees smoke-test teacher per fold.",
    )
    p.add_argument(
        "--ts-teacher-model-path", default=None,
        help="Path to the downloaded TabPFN-TS checkpoint. Defaults to "
             "TABPFN_TS_MODEL_PATH or the download script cache path.",
    )
    p.add_argument(
        "--ts-teacher-device", default="cpu",
        help="Device passed to TabPFN-TS LOCAL mode (default: cpu).",
    )
    p.add_argument(
        "--ts-teacher-n-estimators", type=int, default=8,
        help="Number of TabPFN estimators used by the TabPFN-TS teacher.",
    )
    p.add_argument(
        "--ts-teacher-workers", type=int, default=1,
        help=(
            "CPU worker count inside tabpfn_time_series. Default 1 avoids "
            "loky/semaphore issues in sandboxed macOS runs; increase for "
            "paper-scale local runs if the machine supports it."
        ),
    )
    p.add_argument(
        "--include-tabpfn-ts-distill",
        action="store_true",
        help=(
            "Append temporal TabPFN-TS distillation rows: black-box "
            "TabPFN-TS teacher -> XGB/ET/CB rule student -> PPtheta-Post."
        ),
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
    p.add_argument("--n-l4-variants", type=int, default=4,
                   help="Number of L4 inference variants to evaluate "
                        "(taken from DEFAULT_TEMPORAL_VARIANTS).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=os.path.join(THIS_DIR, "..",
                                                          "output", "temporal"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    level_keys = {str(level).upper() for level in args.levels}
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(args.output_dir, f"compare_temporal_{timestamp}.md")

    if args.baselines == ["all"]:
        baseline_keys = list(UNIFIED_BASELINE_REGISTRY.keys())
    else:
        baseline_keys = [b.lower() for b in args.baselines]
        unknown = [b for b in baseline_keys
                   if b not in UNIFIED_BASELINE_REGISTRY]
        if unknown:
            raise SystemExit(
                f"unknown baseline(s): {unknown}; "
                f"choose from {sorted(UNIFIED_BASELINE_REGISTRY)}"
            )

    md_buf = StringIO()
    md_buf.write(f"# PPθ-Post temporal comparison — {timestamp}\n\n")
    if baseline_keys:
        md_buf.write(
            "> **Scope**: PPθ-Post L1 / L2 / L3 / L4 variants **plus "
            "external baselines**: "
            f"{', '.join(baseline_keys)}.  All baselines share the "
            "`(X_ts, mask, y)` interface; for InterpGN the entry is "
            "annotated with the average routing fraction `g(X)` — see "
            "§6.5 for the gate-opacity caveat.\n>\n"
        )
    else:
        md_buf.write(
            "> **Scope**: intra-method ablation — only PPθ-Post L1 / L2 / "
            "L3 / L4 variants.  Add `--baselines lr xgb transformer "
            "sand mtan gru_d seft raindrop camelot interp_gn` (or "
            "`--baselines all`) to include the external baselines.  "
            "SOTA rows always use the *authors' original code* from "
            "`temporal/vendor/*` — adapters that cannot initialise are "
            "skipped (no re-implementation fallback).\n>\n"
        )
    md_buf.write(
        "> **Sanity-check disclaimer**: results on synthetic loaders "
        "(`p12` / `pam` / `mimic3`) are smoke-level only.  Paper-quality "
        "numbers require credentialed real benchmarks (PhysioNet/2012, "
        "PAMAP2, MIMIC-III/IV) — see `PAPER_LAYOUT.md`.\n\n"
    )
    md_buf.write(f"Levels: {args.levels} | folds: {args.folds} | "
                 f"epochs: {args.epochs}")
    if args.include_tabpfn_ts_distill:
        md_buf.write(
            f" | tabpfn_ts_distill_levels: {args.tabpfn_ts_distill_levels}"
            f" | tabpfn_ts_distill_students: {args.tabpfn_ts_distill_students}"
            f" | ts_teacher_backend: {args.ts_teacher_backend}"
            f" | ts_teacher_head: {args.tabpfn_ts_teacher_head}"
        )
    if baseline_keys:
        md_buf.write(f" | baselines: {baseline_keys}")
    md_buf.write("\n")

    l4_variants = list(DEFAULT_TEMPORAL_VARIANTS)[: args.n_l4_variants]

    for ds_name in args.datasets:
        print(f"\n=== dataset: {ds_name} ===")
        X_ts, mask, y, var_names, dataset_name = load_temporal_dataset(ds_name)
        n_classes = int(np.unique(y).size)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.seed)

        per_variant: Dict[str, List[FoldResult]] = {}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_ts, y)):
            print(f"  fold {fold_idx + 1}/{args.folds}")
            X_tr_ts, mask_tr, y_tr = X_ts[train_idx], mask[train_idx], y[train_idx]
            X_va_ts, mask_va, y_va = X_ts[val_idx], mask[val_idx], y[val_idx]

            results: Dict[str, FoldResult] = {}
            if "L1" in level_keys:
                results.update(run_l1(
                    X_tr_ts, mask_tr, y_tr,
                    X_va_ts, mask_va, y_va,
                    n_classes=n_classes, seed=args.seed,
                    epochs=args.epochs,
                ))
            if "L2" in level_keys:
                results.update(run_l2(
                    X_tr_ts, mask_tr, y_tr,
                    X_va_ts, mask_va, y_va,
                    n_classes=n_classes, seed=args.seed,
                    epochs=args.epochs, n_windows=args.n_windows,
                ))
            if "L3" in level_keys:
                results.update(run_l3(
                    X_tr_ts, mask_tr, y_tr,
                    X_va_ts, mask_va, y_va,
                    var_names=var_names, n_classes=n_classes,
                    seed=args.seed, epochs=args.epochs,
                    n_intervals=args.n_intervals,
                ))
            if args.include_tabpfn_ts_distill:
                for distill_level in args.tabpfn_ts_distill_levels:
                    for student in args.tabpfn_ts_distill_students:
                        try:
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
                        except (RuntimeError, ImportError, ValueError) as exc:
                            print(
                                "    [skipped] TabPFN-TS distill "
                                f"{distill_level}/{student}: {exc}"
                            )
            if "L4" in level_keys:
                results.update(run_l4(
                    X_tr_ts, mask_tr, y_tr,
                    X_va_ts, mask_va, y_va,
                    var_names=var_names, n_classes=n_classes,
                    seed=args.seed, epochs=args.epochs,
                    variants=l4_variants,
                ))
            for bl_name in baseline_keys:
                try:
                    baseline_kwargs = {}
                    if bl_name == "tabpfn_ts":
                        baseline_kwargs = dict(
                            ts_backend=args.ts_teacher_backend,
                            ts_max_rows=args.ts_teacher_max_rows,
                            ts_model_path=args.ts_teacher_model_path,
                            ts_device=args.ts_teacher_device,
                            ts_n_estimators=args.ts_teacher_n_estimators,
                            ts_num_workers=args.ts_teacher_workers,
                            head=args.tabpfn_ts_teacher_head,
                            classifier_model_path=(
                                args.tabpfn_classifier_model_path
                            ),
                            classifier_device=args.ts_teacher_device,
                            classifier_n_estimators=args.ts_teacher_n_estimators,
                        )
                    results.update(run_baseline(
                        bl_name,
                        X_tr_ts, mask_tr, y_tr,
                        X_va_ts, mask_va, y_va,
                        n_classes=n_classes, seed=args.seed,
                        epochs=args.epochs,
                        **baseline_kwargs,
                    ))
                except (RuntimeError, ImportError) as exc:
                    print(f"    [skipped] baseline {bl_name!r}: {exc}")
                except Exception as exc:                            # pragma: no cover
                    print(f"    [warn]    baseline {bl_name!r} failed: {exc}")

            for variant_name, fold_result in results.items():
                per_variant.setdefault(variant_name, []).append(fold_result)

        rows: Dict[str, AggregatedRow] = {}
        for variant_name, fold_list in per_variant.items():
            rows[variant_name] = _aggregate(fold_list)

        table = _format_table(dataset_name, rows)
        md_buf.write(table)
        print(table)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_buf.getvalue())
    print(f"\nSaved comparison report to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
