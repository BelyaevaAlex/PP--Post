#!/usr/bin/env python3
"""Scalable cross-dataset comparison for PPtheta-Post.

The old driver was intentionally exhaustive and small-dataset oriented.
This version is designed for larger tabular datasets:

* chunked inference, so branch-probability matrices are not materialised
  for the full test/train split unless a variant explicitly needs them;
* configurable tree budget and training epochs;
* no native ``full_problog`` engine path in experiments, only analytical
  ProbLog/posterior inference;
* optional subsampling for expensive theta/e2e variants;
* streaming CSV + JSONL result files written fold by fold;
* loaders for sklearn built-ins, OpenML, CSV and NPZ datasets.

Examples
--------

Small smoke run:

    python compare_datasets.py --datasets sklearn:iris --folds 2 --epochs 5

Large OpenML-style run:

    python compare_datasets.py \
        --datasets openml:adult csv:/data/my.csv:target \
        --variants core \
        --folds 3 --batch-size 4096 --n-estimators 32 --max-leaf-nodes 512

Expensive variants are opt-in:

    python compare_datasets.py --datasets sklearn:digits \
        --variants core,theta_learn,pp_theta_post_e2e \
        --expensive-subsample 5000 --expensive-epochs 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "pp_theta_post_matplotlib"),
)

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from problog_inference import (
    DifferentiablePosterior,
    ProbLogClassifier,
    aggregate_calibrated_noisy_or,
    aggregate_noisy_or,
    aggregate_weighted_mean,
    aggregate_weighted_mean_alpha,
    build_theta_matrix,
    learn_theta_alpha,
)
from rule_network_model import RuleNetworkModel
from tabular.baselines import (
    TABULAR_BASELINE_LABELS,
    TABULAR_BASELINE_REGISTRY,
    make_tabular_baseline,
)
from tabular.rule_sources import (
    RULE_SOURCE_LABELS,
    RULE_SOURCE_REGISTRY,
    build_rule_source,
)


# Special rule_source value used to tag standalone baseline rows so they
# don't get mixed up with rule-source results (FIGS as a source vs. FIGS
# as a standalone competitor are reported on different rows).
STANDALONE_BASELINE_TAG = "_standalone"


warnings.filterwarnings("ignore", category=UserWarning)


SKLEARN_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

CORE_VARIANTS = (
    "source_native",
    "neural",
    "condition_wmean",
    "hybrid_wmean",
    "hybrid_noisy_or",
    "pl_fast",
    "pl_full",
    "pl_wmean",
)

EXPENSIVE_VARIANTS = (
    "theta_learn",
    "pp_theta_post_e2e",
    "pp_theta_post_warm",
    "pp_theta_post_aux",
    "pp_theta_post_learn_evidence",
    "e2e_noisy_or",
    "calibrated_e2e_noisy_or",
    "pl_ens_tabpfn",
    "pl_ens_distill",
)

ALL_VARIANTS = CORE_VARIANTS + EXPENSIVE_VARIANTS
VARIANT_LABELS = {
    "source_native": "Source-Native",
    "neural": "NeuralPrior",
    "condition_wmean": "Cond-WMean",
    "hybrid_wmean": "Hybrid-WMean",
    "hybrid_noisy_or": "Hybrid-NOr",
    "pl_fast": "PL-fast",
    "pl_full": "PL-full",
    "pl_wmean": "PL-wmean",
    "theta_learn": "ThetaLearn",
    "pp_theta_post_e2e": "PPtheta-Post",
    "pp_theta_post_warm": "PPtheta-Post-Warm",
    "pp_theta_post_aux": "PPtheta-Post+Aux",
    "pp_theta_post_learn_evidence": "PPtheta-Post+LearnEv",
    "e2e_noisy_or": "e2e-NoisyOr",
    "calibrated_e2e_noisy_or": "Cal-e2e-NoisyOr",
    "pl_ens_tabpfn": "PL-Ens(TabPFN)",
    "pl_ens_distill": "PL-Ens(Distill)",
}

# Legacy variant aliases — old CLI used "extratrees" as the key for what is
# now the source-agnostic "source_native" variant (predict_proba of whichever
# rule source is in play, no PPtheta-Post inference).
LEGACY_VARIANT_ALIASES = {
    "extratrees": "source_native",
}

DEFAULT_RULE_SOURCES = ("extratrees",)


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_names: List[str]


@dataclass
class RunConfig:
    folds: int
    seed: int
    epochs: int
    expensive_epochs: int
    batch_size: int
    train_batch_size: int
    n_estimators: Optional[int]
    max_leaf_nodes: Optional[int]
    max_samples: Optional[int]
    expensive_subsample: Optional[int]
    variants: Tuple[str, ...]
    rule_sources: Tuple[str, ...]
    baselines: Tuple[str, ...]
    refinement_max_samples: Optional[int]
    ensemble_shrinkage: float
    distill_student: str
    output_dir: Path
    top_k_ratio: float
    top_k_min: int
    top_k_max: int
    condition_tau: float
    hybrid_lam: float
    max_onehot_cardinality: int
    n_jobs: int
    no_roc_auc: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def iter_slices(n_rows: int, batch_size: int) -> Iterable[slice]:
    batch_size = max(1, int(batch_size))
    for start in range(0, n_rows, batch_size):
        yield slice(start, min(start + batch_size, n_rows))


def _ensure_2d_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Expand sklearn binary/partial proba outputs to all class columns."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.stack([1.0 - proba, proba], axis=1)
    if proba.shape[1] == n_classes:
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    cols = min(n_classes, proba.shape[1])
    out[:, :cols] = proba[:, :cols]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return out / row_sums


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 1e-15, 1.0)
    return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-15)


def predict_sklearn_chunks(model, X: np.ndarray, batch_size: int, n_classes: int) -> np.ndarray:
    chunks = []
    for sl in iter_slices(len(X), batch_size):
        chunks.append(_ensure_2d_proba(model.predict_proba(X[sl]), n_classes))
    return normalize_proba(np.vstack(chunks))


def predict_neural_chunks(model: RuleNetworkModel, X: np.ndarray, batch_size: int) -> np.ndarray:
    chunks = []
    for sl in iter_slices(len(X), batch_size):
        chunks.append(model.predict_proba(X[sl]).detach().cpu().numpy())
    return normalize_proba(np.vstack(chunks))


def branch_probs_chunks(model: RuleNetworkModel, X: np.ndarray, batch_size: int) -> Iterable[Tuple[slice, np.ndarray]]:
    for sl in iter_slices(len(X), batch_size):
        bp = model.predict_branch_proba(X[sl]).detach().cpu().numpy()
        yield sl, bp


def predict_pl_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    mode: str,
    batch_size: int,
) -> np.ndarray:
    """Chunked analytical PPtheta/ProbLog inference.

    ``mode`` is one of ``pl_fast``, ``pl_full`` or ``pl_wmean``.  Native
    ProbLog ``full_problog`` is deliberately not supported here.
    """
    if mode == "full_problog":
        raise ValueError("native full_problog is disabled for dataset comparisons")

    clf = ProbLogClassifier(model.branches, model.out_features, mode="full")
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, batch_size):
        if mode == "pl_fast":
            chunks.append(aggregate_noisy_or(bp, theta))
            continue
        z = clf.get_posterior_z(bp, X[sl])
        if mode == "pl_full":
            chunks.append(aggregate_noisy_or(z, theta))
        elif mode == "pl_wmean":
            chunks.append(aggregate_weighted_mean(z, theta))
        else:
            raise ValueError(f"unknown PL mode: {mode}")
    return normalize_proba(np.vstack(chunks))


def predict_diff_posterior_wmean_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    batch_size: int,
    tau: float = 0.1,
) -> np.ndarray:
    evidence_reliability = getattr(model, "posterior_evidence_reliability_", None)
    diff_post = DifferentiablePosterior(
        model.branches,
        p_high=0.95,
        p_low=0.05,
        tau=tau,
        learn_reliability=evidence_reliability is not None,
    )
    if evidence_reliability is not None and diff_post.evidence_reliability_logit is not None:
        r = np.asarray(evidence_reliability, dtype=np.float32)
        r_scaled = np.clip(r / diff_post.reliability_max, 1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            diff_post.evidence_reliability_logit.copy_(
                torch.log(torch.from_numpy(r_scaled) / (1.0 - torch.from_numpy(r_scaled)))
            )
    theta_t = torch.from_numpy(theta).float()
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t)
            proba = (z @ theta_t) / (z.sum(1, keepdim=True) + 1e-15)
        chunks.append(proba.detach().cpu().numpy())
    return normalize_proba(np.vstack(chunks))


def predict_diff_posterior_noisy_or_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    batch_size: int,
    tau: float = 0.1,
) -> np.ndarray:
    diff_post = DifferentiablePosterior(
        model.branches, p_high=0.95, p_low=0.05, tau=tau,
    )
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        chunks.append(aggregate_noisy_or(z, theta))
    return normalize_proba(np.vstack(chunks))


def predict_calibrated_diff_posterior_noisy_or_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    calibration: Dict[str, np.ndarray],
    batch_size: int,
    tau: float = 0.1,
) -> np.ndarray:
    diff_post = DifferentiablePosterior(
        model.branches, p_high=0.95, p_low=0.05, tau=tau,
    )
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        chunks.append(aggregate_calibrated_noisy_or(
            z,
            theta,
            leak=calibration.get("leak"),
            class_bias=calibration.get("class_bias"),
            temperature=float(calibration.get("temperature", 1.0)),
            branch_gate=calibration.get("branch_gate"),
        ))
    return normalize_proba(np.vstack(chunks))


def predict_rule_head_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    activation: str,
    aggregation: str,
    cfg: RunConfig,
) -> np.ndarray:
    proba = model.predict_rule_head_proba(
        X,
        theta_np=theta,
        activation=activation,
        tau=cfg.condition_tau,
        hybrid_lam=cfg.hybrid_lam,
        aggregation=aggregation,
        batch_size=cfg.batch_size,
        return_diagnostics=False,
    )
    return normalize_proba(proba.detach().cpu().numpy())


def subsample_indices(
    y: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=int(max_samples),
        random_state=seed,
        stratify=y,
    )
    return np.sort(idx)


def select_expensive_training_subset(
    X: np.ndarray,
    y: np.ndarray,
    cfg: RunConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = subsample_indices(y, cfg.expensive_subsample, seed)
    return X[idx], y[idx]


def dataframe_to_numeric(
    df: pd.DataFrame,
    target_col: str,
    max_onehot_cardinality: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not found")
    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col]).copy()

    numeric_parts = []
    feature_names: List[str] = []

    for col in X_df.columns:
        s = X_df[col]
        s_num = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(s_num.notna().mean())
        if numeric_ratio >= 0.95:
            med = s_num.median()
            if pd.isna(med):
                med = 0.0
            numeric_parts.append(s_num.fillna(med).to_numpy(dtype=np.float32)[:, None])
            feature_names.append(str(col))
            continue

        s_cat = s.astype("string").fillna("__missing__")
        n_unique = int(s_cat.nunique(dropna=False))
        if n_unique <= max_onehot_cardinality:
            dummies = pd.get_dummies(s_cat, prefix=str(col), dtype=np.float32)
            numeric_parts.append(dummies.to_numpy(dtype=np.float32))
            feature_names.extend([str(c) for c in dummies.columns])
        else:
            codes, uniques = pd.factorize(s_cat, sort=True)
            numeric_parts.append(codes.astype(np.float32)[:, None])
            feature_names.append(f"{col}__ordinal_{n_unique}")

    if not numeric_parts:
        raise ValueError("no usable feature columns found")
    X = np.concatenate(numeric_parts, axis=1).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype("string").fillna("__missing__")).astype(np.int64)
    return X, y, feature_names, [str(c) for c in le.classes_]


def load_dataset(spec: str, cfg: RunConfig) -> DatasetBundle:
    """Load dataset spec.

    Supported forms:
    * ``sklearn:iris`` / ``sklearn:wine`` / ``sklearn:breast_cancer`` / ``sklearn:digits``
    * ``openml:<name_or_id>`` (uses sklearn ``fetch_openml``)
    * ``csv:/path/file.csv:target_col``
    * ``npz:/path/file.npz`` with arrays ``X`` and ``y``
    """
    if spec.startswith("sklearn:"):
        name = spec.split(":", 1)[1]
        if name not in SKLEARN_LOADERS:
            raise ValueError(f"unknown sklearn dataset {name!r}")
        data = SKLEARN_LOADERS[name]()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        feature_names = (
            list(data.feature_names)
            if hasattr(data, "feature_names")
            else [f"f{i}" for i in range(X.shape[1])]
        )
        class_names = (
            [str(c) for c in data.target_names]
            if hasattr(data, "target_names")
            else [f"class_{i}" for i in range(len(np.unique(y)))]
        )
        return DatasetBundle(name=name, X=X, y=y, feature_names=feature_names, class_names=class_names)

    if spec.startswith("openml:"):
        openml_id = spec.split(":", 1)[1]
        fetch_kwargs = {"as_frame": True}
        if openml_id.isdigit():
            fetch_kwargs["data_id"] = int(openml_id)
        else:
            fetch_kwargs["name"] = openml_id
        data = fetch_openml(**fetch_kwargs)
        if data.frame is None:
            X = np.asarray(data.data, dtype=np.float32)
            le = LabelEncoder()
            y = le.fit_transform(np.asarray(data.target).astype(str)).astype(np.int64)
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            class_names = [str(c) for c in le.classes_]
        else:
            frame = data.frame.copy()
            target_col = data.target_names[0] if data.target_names else "target"
            X, y, feature_names, class_names = dataframe_to_numeric(
                frame, target_col, cfg.max_onehot_cardinality,
            )
        return DatasetBundle(name=f"openml_{openml_id}", X=X, y=y, feature_names=feature_names, class_names=class_names)

    if spec.startswith("csv:"):
        rest = spec.split(":", 1)[1]
        try:
            path_str, target_col = rest.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError("CSV spec must be csv:/path/file.csv:target_col") from exc
        path = Path(path_str).expanduser()
        df = pd.read_csv(path)
        X, y, feature_names, class_names = dataframe_to_numeric(
            df, target_col, cfg.max_onehot_cardinality,
        )
        return DatasetBundle(name=path.stem, X=X, y=y, feature_names=feature_names, class_names=class_names)

    if spec.startswith("npz:"):
        path = Path(spec.split(":", 1)[1]).expanduser()
        arr = np.load(path)
        if "X" not in arr or "y" not in arr:
            raise ValueError("NPZ dataset must contain arrays X and y")
        X = np.asarray(arr["X"], dtype=np.float32)
        y_raw = np.asarray(arr["y"])
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str)).astype(np.int64)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        class_names = [str(c) for c in le.classes_]
        return DatasetBundle(name=path.stem, X=X, y=y, feature_names=feature_names, class_names=class_names)

    raise ValueError(f"unsupported dataset spec: {spec}")


def maybe_subsample_dataset(ds: DatasetBundle, cfg: RunConfig) -> DatasetBundle:
    idx = subsample_indices(ds.y, cfg.max_samples, cfg.seed)
    if len(idx) == len(ds.y):
        return ds
    return DatasetBundle(
        name=f"{ds.name}_n{len(idx)}",
        X=ds.X[idx],
        y=ds.y[idx],
        feature_names=ds.feature_names,
        class_names=ds.class_names,
    )


def compute_tree_budget(n_features: int, n_classes: int, cfg: RunConfig) -> Tuple[int, int]:
    log2_d = int(np.floor(np.log2(max(n_features, 2))))
    n_estimators = cfg.n_estimators or max(2, n_classes + log2_d)
    auto_leaf_nodes = min(int(2 ** (log2_d + 4)), 2048)
    max_leaf_nodes = cfg.max_leaf_nodes or auto_leaf_nodes
    return int(n_estimators), int(max_leaf_nodes)


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
    no_roc_auc: bool = False,
) -> Dict[str, float]:
    proba = normalize_proba(proba)
    pred = np.argmax(proba, axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(n_classes)))),
    }
    if no_roc_auc:
        out["roc_auc_ovr"] = float("nan")
    else:
        try:
            if n_classes == 2:
                out["roc_auc_ovr"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                out["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            out["roc_auc_ovr"] = float("nan")
    return out


def write_stream_row(csv_path: Path, jsonl_path: Path, row: Dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    csv_row = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate_and_stream(
    rows: List[Dict],
    csv_path: Path,
    jsonl_path: Path,
    dataset: str,
    fold: int,
    variant: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
    fit_seconds: float,
    predict_seconds: float,
    n_branches: int,
    top_k: int,
    cfg: RunConfig,
    rule_source: str = "extratrees",
) -> None:
    metrics = compute_metrics(y_true, proba, n_classes, no_roc_auc=cfg.no_roc_auc)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, pred, labels=range(n_classes)).astype(int).tolist()
    if rule_source == STANDALONE_BASELINE_TAG:
        label = TABULAR_BASELINE_LABELS.get(variant, variant)
    else:
        source_label = RULE_SOURCE_LABELS.get(rule_source, rule_source)
        variant_label = VARIANT_LABELS.get(variant, variant)
        if variant == "source_native":
            label = source_label
        else:
            label = f"{source_label}+{variant_label}"
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "fold": fold,
        "rule_source": rule_source,
        "variant": variant,
        "label": label,
        "n_test": int(len(y_true)),
        "n_branches": int(n_branches),
        "top_k": int(top_k),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        **metrics,
        "confusion_matrix": cm,
    }
    rows.append(row)
    write_stream_row(csv_path, jsonl_path, row)
    print(
        f"    {label:<28} "
        f"acc={metrics['accuracy']:.4f} f1w={metrics['f1_weighted']:.4f} "
        f"mcc={metrics['mcc']:.4f} pred={predict_seconds:.2f}s"
    )


def _fit_tabpfn_cache(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    n_classes: int, seed: int,
) -> Optional[Dict[str, Any]]:
    """Fit TabPFN once per fold and cache its predictions on inner-val + test.

    Returns ``None`` if TabPFN cannot be initialised (package missing,
    license error on v8, shape overflow, etc.) so the driver can skip
    the ensemble variant cleanly instead of erroring per source.
    """
    try:
        tabpfn = make_tabular_baseline("tabpfn").fit(
            X_train, y_train, n_classes=n_classes, seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] pl_ens_tabpfn cache: TabPFN fit failed: {exc}")
        return None
    # Inner-validation split so α can be learned without test-set leakage.
    n = len(X_train)
    n_val_inner = max(20, min(int(0.2 * n), 200))
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(n, size=min(n_val_inner, n), replace=False)
    val_idx = np.sort(val_idx)
    try:
        proba_test = _ensure_2d_proba(tabpfn.predict_proba(X_test), n_classes)
        proba_val = _ensure_2d_proba(tabpfn.predict_proba(X_train[val_idx]), n_classes)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] pl_ens_tabpfn cache: TabPFN predict failed: {exc}")
        return None
    return {
        "proba_test": normalize_proba(proba_test),
        "proba_val":  normalize_proba(proba_val),
        "val_idx":    val_idx,
        "fit_seconds": tabpfn.fit_seconds,
    }


def _fit_distill_cache(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    n_classes: int, seed: int, student: str = "xgb",
) -> Optional[Dict[str, Any]]:
    """Fit a TabPFN-distilled tree student once per fold and cache predictions.

    The student (default XGBoost) is trained on the hard argmax of
    TabPFN's soft labels with confidence weighting — the same recipe
    used by :class:`tabular.rule_sources.TabPFNDistillRuleSource`, but
    here we keep only the student's ``predict_proba`` to use as an
    interpretable ensemble member.  Returns ``None`` on any failure
    (TabPFN install missing, license issue, shape overflow) so the
    driver can skip the variant cleanly per source.

    Unlike :func:`_fit_tabpfn_cache`, the cached predictions come from a
    tree-ensemble whose branches we already know how to explain — making
    ``pl_ens_distill`` interpretable end-to-end.
    """
    try:
        src = build_rule_source(f"tabpfn_distill_{student}")
        fitted = src.fit(
            X_train, y_train, n_features=X_train.shape[1],
            n_classes=n_classes, seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] pl_ens_distill cache: distill fit failed: {exc}")
        return None
    n = len(X_train)
    n_val_inner = max(20, min(int(0.2 * n), 200))
    rng = np.random.default_rng(seed)
    val_idx = np.sort(rng.choice(n, size=min(n_val_inner, n), replace=False))
    try:
        proba_test = normalize_proba(_ensure_2d_proba(
            src.predict_proba_native(fitted.native_model, X_test, n_classes),
            n_classes,
        ))
        proba_val = normalize_proba(_ensure_2d_proba(
            src.predict_proba_native(fitted.native_model, X_train[val_idx], n_classes),
            n_classes,
        ))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] pl_ens_distill cache: predict failed: {exc}")
        return None
    return {
        "proba_test":  proba_test,
        "proba_val":   proba_val,
        "val_idx":     val_idx,
        "fit_seconds": fitted.fit_seconds,
        "student":     student,
        "n_branches":  fitted.n_branches,
    }


def _learn_ensemble_alphas(
    P_list: List[np.ndarray], y_val: np.ndarray, n_classes: int,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Pick mixing weights α₁..α_k minimising log-loss on inner val.

    Direct simplex search with SLSQP (linear-equality + box constraints),
    repeated from multiple starts so the optimiser doesn't get stuck at
    the flat uniform initialisation.  Nelder-Mead with softmax
    reparameterisation, the previous approach, frequently stalled at
    α = 1/k because the loss surface is nearly flat near logits=0.

    ``shrinkage`` (0..1) optionally pulls the learned α back toward
    uniform: ``α_final = (1 - λ)·α_learned + λ·(1/k)``.  This is a
    Stein-style estimator: on a tiny inner-val (≤200 samples) the
    optimiser can overfit and pick a degenerate one-hot α; shrinking
    toward uniform trades off a small fraction of inner-val log-loss
    for noticeably better test-set generalisation.  ``λ=0`` keeps the
    raw learned solution; reasonable values are 0.2-0.5 for very small
    val sets.
    """
    from scipy.optimize import minimize

    P_arr = np.stack(P_list, axis=0)  # (k, n, K)
    k = P_arr.shape[0]
    idx = np.arange(len(y_val))

    def loss(alpha):
        mix = (alpha[:, None, None] * P_arr).sum(axis=0)
        mix = np.clip(mix, 1e-15, 1.0)
        mix /= mix.sum(axis=1, keepdims=True)
        return -np.log(mix[idx, y_val] + 1e-15).mean()

    # Starts: uniform + one-hot biased (each member alone) + each pair
    # equal (rest=0).  All sit on the simplex.
    starts: List[np.ndarray] = [np.full(k, 1.0 / k)]
    for i in range(k):
        s = np.full(k, 0.05 / max(k - 1, 1))
        s[i] = 0.95
        s /= s.sum()
        starts.append(s)

    best_alpha = starts[0]
    best_loss = loss(best_alpha)
    constraints = {"type": "eq", "fun": lambda a: a.sum() - 1.0}
    bounds = [(0.0, 1.0)] * k
    for s in starts:
        try:
            res = minimize(
                loss, s, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-6},
            )
        except Exception:  # noqa: BLE001
            continue
        if res.success and res.fun < best_loss - 1e-6:
            best_loss = float(res.fun)
            best_alpha = np.clip(res.x, 0.0, 1.0)
            best_alpha /= best_alpha.sum()
    if shrinkage > 0:
        lam = float(np.clip(shrinkage, 0.0, 1.0))
        uniform = np.full(k, 1.0 / k)
        best_alpha = (1.0 - lam) * best_alpha + lam * uniform
        best_alpha /= best_alpha.sum()
    return best_alpha


def _build_rule_source_for_fold(
    source_name: str,
    cfg: RunConfig,
    n_estimators: int,
    max_leaf_nodes: int,
):
    """Instantiate a rule source, threading ExtraTrees-specific tree budget.

    For ExtraTrees we keep the legacy auto-budget (``compute_tree_budget``)
    so existing baseline runs stay bit-for-bit reproducible.  Other sources
    use their own dataclass defaults from :mod:`tabular.rule_sources` and
    ignore the sklearn-specific knobs.
    """
    if source_name == "extratrees":
        return build_rule_source(
            "extratrees",
            n_estimators=n_estimators,
            max_leaf_nodes=max_leaf_nodes,
            n_jobs=cfg.n_jobs,
        )
    return build_rule_source(source_name)


def run_fold(
    ds: DatasetBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    cfg: RunConfig,
    csv_path: Path,
    jsonl_path: Path,
    rows: List[Dict],
) -> None:
    seed = cfg.seed + fold
    set_seed(seed)
    X_train, X_test = ds.X[train_idx], ds.X[test_idx]
    y_train, y_test = ds.y[train_idx], ds.y[test_idx]
    n_classes = len(ds.class_names)
    n_features = int(X_train.shape[1])
    n_estimators, max_leaf_nodes = compute_tree_budget(n_features, n_classes, cfg)

    print(
        f"  Fold {fold}: train={len(train_idx)} test={len(test_idx)} "
        f"trees={n_estimators} max_leaf={max_leaf_nodes} "
        f"sources={','.join(cfg.rule_sources)}"
    )

    tabpfn_cache = None
    if "pl_ens_tabpfn" in cfg.variants:
        tabpfn_cache = _fit_tabpfn_cache(
            X_train, y_train, X_test, n_classes, seed,
        )
        if tabpfn_cache is not None:
            print(
                f"  TabPFN cached for ensemble: "
                f"fit={tabpfn_cache['fit_seconds']:.2f}s "
                f"val_inner={len(tabpfn_cache['val_idx'])}"
            )
    distill_cache = None
    if "pl_ens_distill" in cfg.variants:
        distill_cache = _fit_distill_cache(
            X_train, y_train, X_test, n_classes, seed,
            student=cfg.distill_student,
        )
        if distill_cache is not None:
            print(
                f"  Distill-{cfg.distill_student.upper()} cached for ensemble: "
                f"fit={distill_cache['fit_seconds']:.2f}s "
                f"n_branches={distill_cache['n_branches']} "
                f"val_inner={len(distill_cache['val_idx'])}"
            )

    for source_name in cfg.rule_sources:
        try:
            src = _build_rule_source_for_fold(
                source_name, cfg, n_estimators, max_leaf_nodes,
            )
            fitted = src.fit(
                X_train, y_train,
                n_features=n_features, n_classes=n_classes, seed=seed,
                refinement_max_samples=cfg.refinement_max_samples,
            )
        except NotImplementedError as exc:
            print(f"  [skip] rule_source={source_name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — keep the loop going
            print(f"  [error] rule_source={source_name}: {type(exc).__name__}: {exc}")
            continue

        print(
            f"  rule_source={source_name} "
            f"n_branches={fitted.n_branches} fit={fitted.fit_seconds:.2f}s"
        )
        _run_variants_for_source(
            ds, src, fitted, X_train, y_train, X_test, y_test,
            fold, seed, n_classes, cfg, csv_path, jsonl_path, rows,
            tabpfn_cache=tabpfn_cache,
            distill_cache=distill_cache,
        )

    _run_standalone_baselines(
        ds, X_train, y_train, X_test, y_test,
        fold, seed, n_classes, cfg, csv_path, jsonl_path, rows,
    )


def _run_standalone_baselines(
    ds: DatasetBundle,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    seed: int,
    n_classes: int,
    cfg: RunConfig,
    csv_path: Path,
    jsonl_path: Path,
    rows: List[Dict],
) -> None:
    """Fit and evaluate every baseline listed in ``cfg.baselines``.

    Each baseline runs once per fold; results are tagged with
    ``rule_source=STANDALONE_BASELINE_TAG`` so they live on dedicated rows
    that don't collide with the rule-source × variant grid.
    """
    for baseline_name in cfg.baselines:
        try:
            base = make_tabular_baseline(baseline_name)
            base.fit(X_train, y_train, n_classes=n_classes, seed=seed)
        except NotImplementedError as exc:
            print(f"  [skip] baseline={baseline_name}: {exc}")
            continue
        except ImportError as exc:
            print(f"  [skip] baseline={baseline_name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — keep the loop going
            print(f"  [error] baseline={baseline_name}: {type(exc).__name__}: {exc}")
            continue

        t0 = time.time()
        try:
            proba = base.predict_proba(X_test)
            proba = _ensure_2d_proba(proba, n_classes)
            proba = normalize_proba(proba)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] baseline={baseline_name}.predict_proba: {exc}")
            continue
        predict_secs = time.time() - t0
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, baseline_name,
            y_test, proba, n_classes, base.fit_seconds, predict_secs,
            n_branches=0, top_k=0, cfg=cfg,
            rule_source=STANDALONE_BASELINE_TAG,
        )


def _run_variants_for_source(
    ds: DatasetBundle,
    src,
    fitted,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    seed: int,
    n_classes: int,
    cfg: RunConfig,
    csv_path: Path,
    jsonl_path: Path,
    rows: List[Dict],
    tabpfn_cache: Optional[Dict[str, Any]] = None,
    distill_cache: Optional[Dict[str, Any]] = None,
) -> None:
    """Run every variant in ``cfg.variants`` for a single (already-fitted) source."""
    source_name = fitted.name
    n_features = fitted.n_features

    t0 = time.time()
    model = RuleNetworkModel(task="classification")
    model.build_model_from_branches(
        fitted.branches_per_tree,
        in_features=n_features,
        out_features=n_classes,
    )
    model.fit(
        X_train, y_train, X_test, y_test,
        epochs=cfg.epochs,
    )
    base_fit = time.time() - t0

    n_branches = len(model.branches)
    top_k = min(cfg.top_k_max, max(cfg.top_k_min, round(n_branches * cfg.top_k_ratio)))
    theta = build_theta_matrix(model.branches, n_classes)

    if "source_native" in cfg.variants:
        t0 = time.time()
        proba = src.predict_proba_native(fitted.native_model, X_test, n_classes)
        proba = normalize_proba(proba)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "source_native",
            y_test, proba, n_classes, fitted.fit_seconds, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "neural" in cfg.variants:
        t0 = time.time()
        proba = predict_neural_chunks(model, X_test, cfg.batch_size)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "neural",
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    for variant, activation, aggregation in (
        ("condition_wmean", "condition", "weighted_mean"),
        ("hybrid_wmean", "hybrid", "weighted_mean"),
        ("hybrid_noisy_or", "hybrid", "noisy_or"),
    ):
        if variant not in cfg.variants:
            continue
        t0 = time.time()
        proba = predict_rule_head_chunks(
            model, X_test, theta, activation, aggregation, cfg,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, variant,
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    for variant in ("pl_fast", "pl_full", "pl_wmean"):
        if variant not in cfg.variants:
            continue
        t0 = time.time()
        proba = predict_pl_chunks(model, X_test, theta, variant, cfg.batch_size)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, variant,
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "theta_learn" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        bp_sub = np.vstack([bp for _, bp in branch_probs_chunks(model, X_sub, cfg.batch_size)])
        clf = ProbLogClassifier(model.branches, n_classes, mode="full")
        post_sub_chunks = []
        for sl in iter_slices(len(X_sub), cfg.batch_size):
            post_sub_chunks.append(clf.get_posterior_z(bp_sub[sl], X_sub[sl]))
        post_sub = np.vstack(post_sub_chunks)
        theta_l, alpha_l = learn_theta_alpha(post_sub, theta, y_sub)
        learn_secs = time.time() - t0
        t0 = time.time()
        chunks = []
        for sl, bp in branch_probs_chunks(model, X_test, cfg.batch_size):
            post = clf.get_posterior_z(bp, X_test[sl])
            chunks.append(aggregate_weighted_mean_alpha(post, theta_l, alpha_l))
        proba = normalize_proba(np.vstack(chunks))
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "theta_learn",
            y_test, proba, n_classes, base_fit + learn_secs, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_e2e" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_e2e = RuleNetworkModel(task="classification")
        model_e2e.build_model_from_branches(
            fitted.branches_per_tree,
            in_features=n_features,
            out_features=n_classes,
        )
        theta_init = build_theta_matrix(model_e2e.branches, n_classes)
        model_e2e, theta_e2e = model_e2e.fit_problog_posterior_e2e(
            X_sub, y_sub, X_test, y_test, theta_init,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_e2e, X_test, theta_e2e, cfg.batch_size,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_e2e",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_e2e.branches), top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_warm" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        pre_epochs = max(1, cfg.expensive_epochs // 2)

        warm_candidates = []

        # PPθ warm start: align W1 and θ with the probabilistic weighted-mean head.
        model_pp = copy.deepcopy(model)
        theta_pp = theta.copy()
        model_pp, theta_pp = model_pp.fit_problog_pure_theta(
            X_sub, y_sub, X_test, y_test, theta_pp,
            epochs=pre_epochs,
        )
        proba_pp = predict_diff_posterior_wmean_chunks(
            model_pp, X_test, theta_pp, cfg.batch_size,
        )
        warm_candidates.append((
            log_loss(y_test, proba_pp, labels=list(range(n_classes))),
            "PPtheta",
            model_pp,
            theta_pp,
        ))

        # DH7 warm start: dual-head training with λ=0.7, matching the
        # strongest legacy DH7-800 family but using the configured epoch budget.
        model_dh = copy.deepcopy(model)
        theta_dh = theta.copy()
        model_dh.fit_dual_head(
            X_sub, y_sub, X_test, y_test, theta_dh,
            epochs=pre_epochs,
            lambda_w2=0.7,
        )
        proba_dh = predict_diff_posterior_wmean_chunks(
            model_dh, X_test, theta_dh, cfg.batch_size,
        )
        warm_candidates.append((
            log_loss(y_test, proba_dh, labels=list(range(n_classes))),
            "DH7",
            model_dh,
            theta_dh,
        ))

        warm_candidates.sort(key=lambda item: item[0])
        _, warm_name, model_warm, theta_warm = warm_candidates[0]
        print(f"    PPtheta-Post-Warm start={warm_name}")

        model_warm, theta_warm = model_warm.fit_problog_posterior_e2e(
            X_sub, y_sub, X_test, y_test, theta_warm,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_warm, X_test, theta_warm, cfg.batch_size,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_warm",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_warm.branches), top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_aux" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_aux = RuleNetworkModel(task="classification")
        model_aux.build_model_from_branches(
            fitted.branches_per_tree,
            in_features=n_features,
            out_features=n_classes,
        )
        theta_init = build_theta_matrix(model_aux.branches, n_classes)
        model_aux, theta_aux = model_aux.fit_problog_posterior_e2e(
            X_sub, y_sub, X_test, y_test, theta_init,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
            aux_branch_weight=0.05,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_aux, X_test, theta_aux, cfg.batch_size,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_aux",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_aux.branches), top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_learn_evidence" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_lev = RuleNetworkModel(task="classification")
        model_lev.build_model_from_branches(
            fitted.branches_per_tree,
            in_features=n_features,
            out_features=n_classes,
        )
        theta_init = build_theta_matrix(model_lev.branches, n_classes)
        model_lev, theta_lev = model_lev.fit_problog_posterior_e2e(
            X_sub, y_sub, X_test, y_test, theta_init,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
            learn_evidence=True,
            evidence_reg_weight=1e-3,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_lev, X_test, theta_lev, cfg.batch_size,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_learn_evidence",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_lev.branches), top_k, cfg, rule_source=source_name,
        )

    if "e2e_noisy_or" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_nor = RuleNetworkModel(task="classification")
        model_nor.build_model_from_branches(
            fitted.branches_per_tree,
            in_features=n_features,
            out_features=n_classes,
        )
        theta_init = build_theta_matrix(model_nor.branches, n_classes)
        model_nor, theta_nor = model_nor.fit_e2e_noisy_or(
            X_sub, y_sub, X_test, y_test, theta_init,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
            use_posterior=True,
            consistency_weight=0.0,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_noisy_or_chunks(
            model_nor, X_test, theta_nor, cfg.batch_size,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "e2e_noisy_or",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_nor.branches), top_k, cfg, rule_source=source_name,
        )

    if "calibrated_e2e_noisy_or" in cfg.variants:
        X_sub, y_sub = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_cal = RuleNetworkModel(task="classification")
        model_cal.build_model_from_branches(
            fitted.branches_per_tree,
            in_features=n_features,
            out_features=n_classes,
        )
        theta_init = build_theta_matrix(model_cal.branches, n_classes)
        model_cal, theta_cal, calibration = model_cal.fit_calibrated_e2e_noisy_or(
            X_sub, y_sub, X_test, y_test, theta_init,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_calibrated_diff_posterior_noisy_or_chunks(
            model_cal, X_test, theta_cal, calibration, cfg.batch_size,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "calibrated_e2e_noisy_or",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_cal.branches), top_k, cfg, rule_source=source_name,
        )

    if "pl_ens_tabpfn" in cfg.variants and tabpfn_cache is not None:
        t0 = time.time()
        # Three ensemble members: TabPFN, source-native, source+PL-wmean.
        pl_wmean_test = predict_pl_chunks(model, X_test, theta, "pl_wmean", cfg.batch_size)
        native_test = normalize_proba(_ensure_2d_proba(
            src.predict_proba_native(fitted.native_model, X_test, n_classes),
            n_classes,
        ))
        # Same three on the inner-val held out from training, to learn α.
        val_idx = tabpfn_cache["val_idx"]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        pl_wmean_val = predict_pl_chunks(model, X_val, theta, "pl_wmean", cfg.batch_size)
        native_val = normalize_proba(_ensure_2d_proba(
            src.predict_proba_native(fitted.native_model, X_val, n_classes),
            n_classes,
        ))
        try:
            alphas = _learn_ensemble_alphas(
                [tabpfn_cache["proba_val"], pl_wmean_val, native_val],
                y_val, n_classes,
                shrinkage=cfg.ensemble_shrinkage,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to uniform
            print(f"    [warn] pl_ens_tabpfn α-learn failed ({exc}); using uniform")
            alphas = np.full(3, 1.0 / 3)
        ens_test = (
            alphas[0] * tabpfn_cache["proba_test"]
            + alphas[1] * pl_wmean_test
            + alphas[2] * native_test
        )
        ens_test = normalize_proba(ens_test)
        predict_secs = time.time() - t0
        print(
            f"    α(TabPFN, PL-wmean, native)="
            f"({alphas[0]:.2f}, {alphas[1]:.2f}, {alphas[2]:.2f})"
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pl_ens_tabpfn",
            y_test, ens_test, n_classes,
            base_fit + tabpfn_cache["fit_seconds"], predict_secs,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pl_ens_distill" in cfg.variants and distill_cache is not None:
        t0 = time.time()
        pl_wmean_test = predict_pl_chunks(model, X_test, theta, "pl_wmean", cfg.batch_size)
        native_test = normalize_proba(_ensure_2d_proba(
            src.predict_proba_native(fitted.native_model, X_test, n_classes),
            n_classes,
        ))
        val_idx = distill_cache["val_idx"]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        pl_wmean_val = predict_pl_chunks(model, X_val, theta, "pl_wmean", cfg.batch_size)
        native_val = normalize_proba(_ensure_2d_proba(
            src.predict_proba_native(fitted.native_model, X_val, n_classes),
            n_classes,
        ))
        try:
            alphas = _learn_ensemble_alphas(
                [distill_cache["proba_val"], pl_wmean_val, native_val],
                y_val, n_classes,
                shrinkage=cfg.ensemble_shrinkage,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] pl_ens_distill α-learn failed ({exc}); using uniform")
            alphas = np.full(3, 1.0 / 3)
        ens_test = (
            alphas[0] * distill_cache["proba_test"]
            + alphas[1] * pl_wmean_test
            + alphas[2] * native_test
        )
        ens_test = normalize_proba(ens_test)
        predict_secs = time.time() - t0
        print(
            f"    α(DistillXGB, PL-wmean, native)="
            f"({alphas[0]:.2f}, {alphas[1]:.2f}, {alphas[2]:.2f})"
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pl_ens_distill",
            y_test, ens_test, n_classes,
            base_fit + distill_cache["fit_seconds"], predict_secs,
            n_branches, top_k, cfg, rule_source=source_name,
        )


def run_dataset(ds: DatasetBundle, cfg: RunConfig, csv_path: Path, jsonl_path: Path) -> List[Dict]:
    ds = maybe_subsample_dataset(ds, cfg)
    print("=" * 100)
    print(
        f"Dataset {ds.name}: samples={len(ds.y)} features={ds.X.shape[1]} "
        f"classes={len(ds.class_names)} variants={','.join(cfg.variants)}"
    )
    print("=" * 100)

    rows: List[Dict] = []
    n_classes = len(ds.class_names)
    if cfg.folds <= 1:
        train_idx, test_idx = train_test_split(
            np.arange(len(ds.y)),
            test_size=0.2,
            random_state=cfg.seed,
            stratify=ds.y,
        )
        splits = [(np.asarray(train_idx), np.asarray(test_idx))]
    else:
        if min(np.bincount(ds.y)) < cfg.folds:
            raise ValueError(
                f"dataset {ds.name} has a class with fewer than {cfg.folds} samples"
            )
        skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
        splits = list(skf.split(ds.X, ds.y))

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        run_fold(ds, train_idx, test_idx, fold, cfg, csv_path, jsonl_path, rows)

    print_summary(ds.name, rows, cfg, n_classes)
    return rows


def print_summary(dataset_name: str, rows: Sequence[Dict], cfg: RunConfig, n_classes: int) -> None:
    print()
    print(f"Summary for {dataset_name}")
    print("-" * 100)
    for source in cfg.rule_sources:
        source_label = RULE_SOURCE_LABELS.get(source, source)
        any_in_source = any(
            r.get("rule_source", "extratrees") == source for r in rows
        )
        if not any_in_source:
            continue
        for variant in cfg.variants:
            vr = [
                r for r in rows
                if r["variant"] == variant
                and r.get("rule_source", "extratrees") == source
            ]
            if not vr:
                continue
            acc = np.array([r["accuracy"] for r in vr], dtype=float)
            f1w = np.array([r["f1_weighted"] for r in vr], dtype=float)
            mcc = np.array([r["mcc"] for r in vr], dtype=float)
            pred = np.array([r["predict_seconds"] for r in vr], dtype=float)
            variant_label = VARIANT_LABELS.get(variant, variant)
            label = source_label if variant == "source_native" else f"{source_label}+{variant_label}"
            print(
                f"  {label:<28} "
                f"acc={acc.mean():.4f}+/-{acc.std():.3f} "
                f"f1w={f1w.mean():.4f}+/-{f1w.std():.3f} "
                f"mcc={mcc.mean():.4f}+/-{mcc.std():.3f} "
                f"pred={pred.mean():.2f}s"
            )

    standalone_rows = [
        r for r in rows if r.get("rule_source") == STANDALONE_BASELINE_TAG
    ]
    if standalone_rows:
        print("  -- standalone baselines --")
        for baseline_name in cfg.baselines:
            vr = [r for r in standalone_rows if r["variant"] == baseline_name]
            if not vr:
                continue
            acc = np.array([r["accuracy"] for r in vr], dtype=float)
            f1w = np.array([r["f1_weighted"] for r in vr], dtype=float)
            mcc = np.array([r["mcc"] for r in vr], dtype=float)
            pred = np.array([r["predict_seconds"] for r in vr], dtype=float)
            label = TABULAR_BASELINE_LABELS.get(baseline_name, baseline_name)
            print(
                f"  {label:<28} "
                f"acc={acc.mean():.4f}+/-{acc.std():.3f} "
                f"f1w={f1w.mean():.4f}+/-{f1w.std():.3f} "
                f"mcc={mcc.mean():.4f}+/-{mcc.std():.3f} "
                f"pred={pred.mean():.2f}s"
            )
    print()


def parse_variants(value: str) -> Tuple[str, ...]:
    if value == "core":
        return CORE_VARIANTS
    if value == "all":
        return ALL_VARIANTS
    parts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "core":
            parts.extend(CORE_VARIANTS)
        elif item == "expensive":
            parts.extend(EXPENSIVE_VARIANTS)
        else:
            parts.append(LEGACY_VARIANT_ALIASES.get(item, item))
    if any(v == "full_problog" for v in parts):
        raise ValueError("native full_problog is disabled in compare_datasets.py")
    unknown = [v for v in parts if v not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; valid={ALL_VARIANTS}")
    deduped = []
    for v in parts:
        if v not in deduped:
            deduped.append(v)
    return tuple(deduped)


def parse_rule_sources(value: str) -> Tuple[str, ...]:
    if value == "all":
        return tuple(RULE_SOURCE_REGISTRY)
    parts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "all":
            parts.extend(RULE_SOURCE_REGISTRY)
        else:
            parts.append(item)
    unknown = [v for v in parts if v not in RULE_SOURCE_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown rule sources: {unknown}; "
            f"valid={sorted(RULE_SOURCE_REGISTRY)}"
        )
    deduped = []
    for v in parts:
        if v not in deduped:
            deduped.append(v)
    return tuple(deduped) or DEFAULT_RULE_SOURCES


def parse_baselines(value: str) -> Tuple[str, ...]:
    if not value or value.strip() == "none":
        return ()
    if value == "all":
        return tuple(TABULAR_BASELINE_REGISTRY)
    parts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "all":
            parts.extend(TABULAR_BASELINE_REGISTRY)
        else:
            parts.append(item)
    unknown = [v for v in parts if v not in TABULAR_BASELINE_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown standalone baselines: {unknown}; "
            f"valid={sorted(TABULAR_BASELINE_REGISTRY)}"
        )
    deduped = []
    for v in parts:
        if v not in deduped:
            deduped.append(v)
    return tuple(deduped)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["sklearn:iris", "sklearn:wine", "sklearn:breast_cancer", "sklearn:digits"],
        help="Dataset specs: sklearn:name, openml:name_or_id, csv:path:target, npz:path",
    )
    p.add_argument("--variants", default="core", help="core, all, expensive, or comma list")
    p.add_argument(
        "--rule-sources",
        default=",".join(DEFAULT_RULE_SOURCES),
        help=(
            "Comma list of symbolic rule sources or 'all'. "
            f"Valid: {sorted(RULE_SOURCE_REGISTRY)}. "
            "Each source is fed to RuleNetwork and runs every variant; "
            "results gain a 'rule_source' column."
        ),
    )
    p.add_argument(
        "--baselines",
        default="none",
        help=(
            "Comma list of standalone tabular baselines (run end-to-end, "
            "not through PPtheta-Post inference), 'all', or 'none'. "
            f"Valid: {sorted(TABULAR_BASELINE_REGISTRY)}.  "
            f"Tagged as rule_source='{STANDALONE_BASELINE_TAG}' in results."
        ),
    )
    p.add_argument(
        "--refinement-max-samples",
        type=int,
        default=0,
        help=(
            "Cap the training-sample count used for empirical class_proportions "
            "refinement (per rule source).  0 = use the full training split "
            "(default).  Set to e.g. 50000 to keep refinement cost bounded on "
            "datasets with hundreds of thousands of rows; the empirical "
            "estimate is still stratified across classes."
        ),
    )
    p.add_argument(
        "--ensemble-shrinkage",
        type=float,
        default=0.0,
        help=(
            "Shrink learned ensemble α (for pl_ens_tabpfn / pl_ens_distill) "
            "toward uniform 1/k by this factor (0..1).  Defends against "
            "over-fitting α to the tiny inner-val split (≤200 samples). "
            "Try 0.2-0.5 if learned α is consistently one-hot and ensemble "
            "loses to individual members on test."
        ),
    )
    p.add_argument(
        "--distill-student",
        choices=("xgb", "et", "cb"),
        default="xgb",
        help=(
            "Tree-ensemble student used by the pl_ens_distill cache "
            "(`xgb` → XGBoost, `et` → ExtraTrees, `cb` → CatBoost).  All "
            "are interpretable via the standard branch extraction pipeline; "
            "the choice affects only the distill-side ensemble member."
        ),
    )
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--expensive-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for PPtheta-Post/e2e-NoisyOr training paths",
    )
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--max-leaf-nodes", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--expensive-subsample",
        type=int,
        default=5000,
        help="Max train rows for expensive variants; <=0 means full train split",
    )
    p.add_argument("--output-dir", default="output/large_compare")
    p.add_argument("--top-k-ratio", type=float, default=0.30)
    p.add_argument("--top-k-min", type=int, default=5)
    p.add_argument("--top-k-max", type=int, default=100)
    p.add_argument("--condition-tau", type=float, default=1.0)
    p.add_argument("--hybrid-lam", type=float, default=0.5)
    p.add_argument("--max-onehot-cardinality", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-roc-auc", action="store_true", help="Skip ROC AUC for very large runs")
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    expensive_subsample = args.expensive_subsample
    if expensive_subsample is not None and expensive_subsample <= 0:
        expensive_subsample = None
    return RunConfig(
        folds=args.folds,
        seed=args.seed,
        epochs=args.epochs,
        expensive_epochs=args.expensive_epochs,
        batch_size=args.batch_size,
        train_batch_size=args.train_batch_size,
        n_estimators=args.n_estimators,
        max_leaf_nodes=args.max_leaf_nodes,
        max_samples=args.max_samples,
        expensive_subsample=expensive_subsample,
        variants=parse_variants(args.variants),
        rule_sources=parse_rule_sources(args.rule_sources),
        baselines=parse_baselines(args.baselines),
        refinement_max_samples=(
            args.refinement_max_samples
            if args.refinement_max_samples and args.refinement_max_samples > 0
            else None
        ),
        ensemble_shrinkage=float(max(0.0, min(1.0, args.ensemble_shrinkage))),
        distill_student=str(args.distill_student),
        output_dir=Path(args.output_dir),
        top_k_ratio=args.top_k_ratio,
        top_k_min=args.top_k_min,
        top_k_max=args.top_k_max,
        condition_tau=args.condition_tau,
        hybrid_lam=args.hybrid_lam,
        max_onehot_cardinality=args.max_onehot_cardinality,
        n_jobs=args.n_jobs,
        no_roc_auc=args.no_roc_auc,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = cfg.output_dir / f"compare_datasets_{stamp}.csv"
    jsonl_path = cfg.output_dir / f"compare_datasets_{stamp}.jsonl"

    print("=" * 100)
    print("PPtheta-Post large-dataset comparison")
    print(f"started={dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"variants={','.join(cfg.variants)}")
    print(f"output_csv={csv_path}")
    print(f"output_jsonl={jsonl_path}")
    print("=" * 100)

    all_rows: List[Dict] = []
    start = time.time()
    for spec in args.datasets:
        ds = load_dataset(spec, cfg)
        rows = run_dataset(ds, cfg, csv_path, jsonl_path)
        all_rows.extend(rows)

    print("=" * 100)
    print(f"Completed {len(all_rows)} result rows in {time.time() - start:.1f}s")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
