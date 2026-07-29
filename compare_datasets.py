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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    average_precision_score,
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
    compute_condition_activation,
    compute_soft_posterior,
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
    "pp_theta_post_frozen",
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

RESOURCE_ARCH_VARIANTS = (
    "pp_theta_post_shrink_theta",
    "pp_theta_post_frozen_support_prior",
    "pp_theta_post_signed_logit",
    "pp_theta_post_sparse_logit",
    "pp_theta_post_feature_reliability",
    "pp_theta_post_source_calibrated",
    "pp_theta_post_rule_utility_aux",
    "pp_theta_post_constrained_evidence",
    "pp_theta_post_evidence_logit_aux",
    "pp_theta_post_aux_v2",
    "pp_theta_post_evlogit_kd",
    "pp_theta_post_evlogit_likelihood",
    "pp_theta_post_evlogit_threshold",
    "pp_theta_post_evlogit_decomp",
    "pp_theta_post_evidence_layer_v2",
    "pp_theta_post_teacher_anchored",
    "pp_theta_post_teacher_calibrated",
    "pp_theta_post_selective_evidence",
    "pp_theta_post_rule_family",
    "pp_theta_post_contextual_support",
    "pp_theta_post_ebm_anchor",
    "pp_theta_post_clinical_objective",
    "pp_theta_post_ebm_correction_calibrated",
    "pp_theta_post_ebm_correction_mcc",
    "pp_theta_post_ebm_correction_sensitivity",
    "pp_theta_post_rule_family_calibrated",
    "pp_theta_post_rule_family_sensitivity",
    "pp_theta_post_ebm_bounded_residual_gate",
    "pp_theta_post_agreement_gated",
    "pp_theta_post_tabpfn_ebm_family_calibrated",
    "pp_theta_post_family_utility_pruned_topk",
    "pp_theta_post_operating_calibrated",
    "pp_theta_post_operating_mcc",
    "pp_theta_post_operating_sens90",
    "pp_theta_post_operating_sens92",
    "pp_theta_post_operating_sens95",
    "pp_theta_post_monotone_ebm_families",
    "pp_theta_post_bayes_llr",
    "pp_theta_post_bayes_llr_beta",
    "pp_theta_post_bayes_llr_posneg",
    "pp_theta_post_bayes_llr_posneg_mcc",
    "pp_theta_post_bayes_llr_posneg_sens92",
    "pp_theta_post_ebm_residual_calibrated",
    "pp_theta_post_ebm_residual_mcc",
    "pp_theta_post_ebm_residual_sens92",
    "pp_theta_post_ebm_residual_sens95",
    "pp_theta_post_dual_residual_calibrated",
    "pp_theta_post_dual_residual_mcc",
    "pp_theta_post_dual_residual_sens92",
    "pp_theta_post_dual_residual_sens95_cal",
)

ALL_VARIANTS = CORE_VARIANTS + EXPENSIVE_VARIANTS + RESOURCE_ARCH_VARIANTS
VARIANT_LABELS = {
    "source_native": "Source-Native",
    "neural": "NeuralPrior",
    "condition_wmean": "Cond-WMean",
    "hybrid_wmean": "Hybrid-WMean",
    "hybrid_noisy_or": "Hybrid-NOr",
    "pl_fast": "PL-fast",
    "pl_full": "PL-full",
    "pl_wmean": "PL-wmean",
    "pp_theta_post_frozen": "PPtheta-Post-Frozen",
    "theta_learn": "ThetaLearn",
    "pp_theta_post_e2e": "PPtheta-Post",
    "pp_theta_post_warm": "PPtheta-Post-Warm",
    "pp_theta_post_aux": "PPtheta-Post+Aux",
    "pp_theta_post_learn_evidence": "PPtheta-Post+LearnEv",
    "e2e_noisy_or": "e2e-NoisyOr",
    "calibrated_e2e_noisy_or": "Cal-e2e-NoisyOr",
    "pl_ens_tabpfn": "PL-Ens(TabPFN)",
    "pl_ens_distill": "PL-Ens(Distill)",
    "pp_theta_post_shrink_theta": "PPtheta-Post-ShrinkTheta",
    "pp_theta_post_frozen_support_prior": "PPtheta-Post-SupportPrior",
    "pp_theta_post_signed_logit": "PPtheta-Post-SignedLogit",
    "pp_theta_post_sparse_logit": "PPtheta-Post-SparseLogit",
    "pp_theta_post_feature_reliability": "PPtheta-Post+FeatRel",
    "pp_theta_post_source_calibrated": "PPtheta-Post+SourceCal",
    "pp_theta_post_rule_utility_aux": "PPtheta-Post+RuleUtilAux",
    "pp_theta_post_constrained_evidence": "PPtheta-Post+ConEvidence",
    "pp_theta_post_evidence_logit_aux": "PPtheta-Post+EvLogitAux",
    "pp_theta_post_aux_v2": "PPtheta-Post+AuxV2",
    "pp_theta_post_evlogit_kd": "PPtheta-Post+EvLogitKD",
    "pp_theta_post_evlogit_likelihood": "PPtheta-Post+EvLikAux",
    "pp_theta_post_evlogit_threshold": "PPtheta-Post+EvLogitThr",
    "pp_theta_post_evlogit_decomp": "PPtheta-Post+EvDecomp",
    "pp_theta_post_evidence_layer_v2": "PPtheta-Post+EvidenceLayerV2",
    "pp_theta_post_teacher_anchored": "PPtheta-Post+TeacherAnchor",
    "pp_theta_post_teacher_calibrated": "PPtheta-Post+TeacherCal",
    "pp_theta_post_selective_evidence": "PPtheta-Post+SelectiveEvidence",
    "pp_theta_post_rule_family": "PPtheta-Post+RuleFamily",
    "pp_theta_post_contextual_support": "PPtheta-Post+ContextTheta",
    "pp_theta_post_ebm_anchor": "PPtheta-Post+EBMAnchor",
    "pp_theta_post_clinical_objective": "PPtheta-Post+ClinicalObjective",
    "pp_theta_post_ebm_correction_calibrated": "EBM+PPtheta-Cal",
    "pp_theta_post_ebm_correction_mcc": "EBM+PPtheta-MCC",
    "pp_theta_post_ebm_correction_sensitivity": "EBM+PPtheta-Sens",
    "pp_theta_post_rule_family_calibrated": "PPtheta-Post+RuleFamilyCal",
    "pp_theta_post_rule_family_sensitivity": "PPtheta-Post+RuleFamilySens",
    "pp_theta_post_ebm_bounded_residual_gate": "EBM+PPtheta-BoundedGate",
    "pp_theta_post_agreement_gated": "EBM+PPtheta-AgreementGate",
    "pp_theta_post_tabpfn_ebm_family_calibrated": "TabPFN-EBM+RuleFamilyCal",
    "pp_theta_post_family_utility_pruned_topk": "PPtheta-Post+UtilityPrunedTopK",
    "pp_theta_post_operating_calibrated": "PPtheta-Post+OpCal",
    "pp_theta_post_operating_mcc": "PPtheta-Post+OpMCC",
    "pp_theta_post_operating_sens90": "PPtheta-Post+OpSens90",
    "pp_theta_post_operating_sens92": "PPtheta-Post+OpSens92",
    "pp_theta_post_operating_sens95": "PPtheta-Post+OpSens95",
    "pp_theta_post_monotone_ebm_families": "PPtheta-Post+MonotoneEBMFamilies",
    "pp_theta_post_bayes_llr": "PPtheta-Post+BayesLLR",
    "pp_theta_post_bayes_llr_beta": "PPtheta-Post+BayesLLR-Beta",
    "pp_theta_post_bayes_llr_posneg": "PPtheta-Post+BayesLLR-PosNeg",
    "pp_theta_post_bayes_llr_posneg_mcc": "PPtheta-Post+BayesLLR-PosNegMCC",
    "pp_theta_post_bayes_llr_posneg_sens92": "PPtheta-Post+BayesLLR-PosNegSens92",
    "pp_theta_post_ebm_residual_calibrated": "EBM+PPtheta-ResidualCal",
    "pp_theta_post_ebm_residual_mcc": "EBM+PPtheta-ResidualMCC",
    "pp_theta_post_ebm_residual_sens92": "EBM+PPtheta-ResidualSens92",
    "pp_theta_post_ebm_residual_sens95": "EBM+PPtheta-ResidualSens95",
    "pp_theta_post_dual_residual_calibrated": "EBM+PPtheta-DualResidualCal",
    "pp_theta_post_dual_residual_mcc": "EBM+PPtheta-DualResidualMCC",
    "pp_theta_post_dual_residual_sens92": "EBM+PPtheta-DualResidualSens92",
    "pp_theta_post_dual_residual_sens95_cal": "EBM+PPtheta-DualResidualSens95Cal",
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
    posterior_p_high: float
    posterior_p_low: float
    theta_shrinkage_strength: float
    signed_logit_temperature: float
    sparse_logit_top_k: int
    rule_budget: int
    rule_max_depth: int
    rule_min_support: float
    rule_selection: str
    hybrid_lam: float
    max_onehot_cardinality: int
    n_jobs: int
    no_roc_auc: bool
    save_predictions: bool


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
    p_high: float = 0.95,
    p_low: float = 0.05,
) -> np.ndarray:
    evidence_reliability = getattr(model, "posterior_evidence_reliability_", None)
    diff_post = DifferentiablePosterior(
        model.branches,
        p_high=p_high,
        p_low=p_low,
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
    p_high: float = 0.95,
    p_low: float = 0.05,
) -> np.ndarray:
    diff_post = DifferentiablePosterior(
        model.branches, p_high=p_high, p_low=p_low, tau=tau,
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
    p_high: float = 0.95,
    p_low: float = 0.05,
) -> np.ndarray:
    diff_post = DifferentiablePosterior(
        model.branches, p_high=p_high, p_low=p_low, tau=tau,
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



def predict_frozen_posterior_wmean_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    cfg: RunConfig,
) -> np.ndarray:
    """Fully interpretable PPtheta-Post head with frozen rule activations.

    The prior branch activation is the explicit soft satisfaction of the stored
    symbolic conditions.  No learned neural hidden activation is used; the same
    analytical posterior update is then aggregated with the theta matrix.
    """
    chunks = []
    for sl in iter_slices(len(X), cfg.batch_size):
        X_chunk = X[sl]
        prior_z = compute_condition_activation(
            model.branches, X_chunk, tau=cfg.condition_tau,
        )
        post_z = compute_soft_posterior(
            model.branches, prior_z, X_chunk,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
            tau=cfg.condition_tau,
        )
        chunks.append(aggregate_weighted_mean(post_z, theta))
    return normalize_proba(np.vstack(chunks))



def empirical_class_prior(
    y: np.ndarray,
    n_classes: int,
    smooth: float = 1.0,
) -> np.ndarray:
    counts = np.bincount(np.asarray(y).ravel().astype(int), minlength=n_classes).astype(np.float64)
    counts += float(max(smooth, 0.0))
    return counts / np.maximum(counts.sum(), 1e-12)


def estimate_branch_support(
    branches: Sequence,
    X: np.ndarray,
    batch_size: int,
    tau: float,
) -> np.ndarray:
    """Soft empirical support of each rule on X."""
    branches = list(branches)
    if not branches:
        return np.zeros(0, dtype=np.float64)
    total = np.zeros(len(branches), dtype=np.float64)
    n_seen = 0
    for sl in iter_slices(len(X), batch_size):
        z = compute_condition_activation(branches, X[sl], tau=tau)
        total += z.sum(axis=0)
        n_seen += z.shape[0]
    return np.clip(total / max(n_seen, 1), 0.0, 1.0)


def _hard_rule_mask(branch, X: np.ndarray) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for cond in branch.conditions:
        vals = X[:, cond.feature_idx]
        if cond.direction == "le":
            mask &= vals <= cond.threshold
        else:
            mask &= vals > cond.threshold
    return mask


def refine_branch_class_proportions(
    branches: Sequence,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    smooth: float = 1.0,
) -> None:
    """Refresh theta/class proportions after short-rule truncation or pruning."""
    y64 = np.asarray(y).ravel().astype(int)
    prior = empirical_class_prior(y64, n_classes, smooth=smooth)
    for branch in branches:
        mask = _hard_rule_mask(branch, X)
        if mask.any():
            counts = np.bincount(y64[mask], minlength=n_classes).astype(np.float64)
            counts += float(max(smooth, 0.0))
            branch.class_proportions = (counts / np.maximum(counts.sum(), 1e-12)).tolist()
        else:
            branch.class_proportions = prior.tolist()


def _branch_signature(branch) -> Tuple:
    cond_sig = tuple(
        (int(c.feature_idx), str(c.direction), round(float(c.threshold), 10))
        for c in branch.conditions
    )
    return (cond_sig, branch.split_feature_idx, None if branch.split_threshold is None else round(float(branch.split_threshold), 10))


def _shortened_branch(branch, max_depth: int):
    br = copy.deepcopy(branch)
    if max_depth > 0 and len(br.conditions) > max_depth:
        br.conditions = list(br.conditions[:max_depth])
        br.branch_id = f"{br.branch_id}_d{max_depth}"
    return br


def select_rule_resource_branches(
    branches_per_tree: Sequence[Sequence],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    cfg: RunConfig,
) -> Tuple[List[List], Dict[str, Any]]:
    """Build the short/stable/budgeted rule resource used by Section 12.

    The source model itself remains unchanged for the native baseline row.  Only
    the symbolic PPtheta-Post rule pool is pruned/truncated before constructing
    RuleNetworkModel.
    """
    budget = int(max(cfg.rule_budget, 0))
    max_depth = int(max(cfg.rule_max_depth, 0))
    min_support = float(max(cfg.rule_min_support, 0.0))
    if budget <= 0 and max_depth <= 0 and min_support <= 0:
        return [list(t) for t in branches_per_tree], {
            "enabled": False,
            "original": int(sum(len(t) for t in branches_per_tree)),
            "selected": int(sum(len(t) for t in branches_per_tree)),
        }

    candidates = []
    for tree_idx, branches in enumerate(branches_per_tree):
        for branch in branches:
            br = _shortened_branch(branch, max_depth)
            candidates.append((tree_idx, br))

    if not candidates:
        return [list(t) for t in branches_per_tree], {"enabled": True, "original": 0, "selected": 0}

    flat = [br for _, br in candidates]
    support = estimate_branch_support(flat, X_train, cfg.batch_size, cfg.condition_tau)

    best_by_sig: Dict[Tuple, Tuple[float, int, Any, float]] = {}
    for idx, (tree_idx, br) in enumerate(candidates):
        sup = float(support[idx])
        if min_support > 0 and sup < min_support:
            continue
        cp = np.asarray(br.class_proportions, dtype=np.float64)
        cp = cp / np.maximum(cp.sum(), 1e-12)
        purity = float(cp.max()) if cp.size else 0.0
        depth = max(len(br.conditions), 1)
        score = (purity - 1.0 / max(n_classes, 1)) * np.log1p(sup * len(X_train)) / np.sqrt(depth)
        sig = _branch_signature(br)
        prev = best_by_sig.get(sig)
        if prev is None or score > prev[0]:
            best_by_sig[sig] = (float(score), tree_idx, br, sup)

    ranked = sorted(best_by_sig.values(), key=lambda item: item[0], reverse=True)
    if budget > 0 and len(ranked) > budget:
        if cfg.rule_selection == "diverse":
            buckets: Dict[Tuple, List[Tuple[float, int, Any, float]]] = {}
            for item in ranked:
                _, _, br, _ = item
                cp = np.asarray(br.class_proportions, dtype=np.float64)
                cls = int(cp.argmax()) if cp.size else 0
                feats = tuple(sorted({int(c.feature_idx) for c in br.conditions})[:3])
                buckets.setdefault((cls, feats), []).append(item)
            selected = []
            keys = sorted(buckets, key=lambda k: buckets[k][0][0], reverse=True)
            while keys and len(selected) < budget:
                next_keys = []
                for key in keys:
                    if buckets[key] and len(selected) < budget:
                        selected.append(buckets[key].pop(0))
                    if buckets[key]:
                        next_keys.append(key)
                keys = next_keys
            ranked = selected
        else:
            ranked = ranked[:budget]

    selected_branches = [br for _, _, br, _ in ranked]
    refine_branch_class_proportions(selected_branches, X_train, y_train, n_classes)

    n_trees = max([len(branches_per_tree)] + [int(tree_idx) + 1 for _, tree_idx, _, _ in ranked])
    out = [[] for _ in range(n_trees)]
    for _, tree_idx, br, _ in ranked:
        out[int(tree_idx)].append(br)
    selected = sum(len(t) for t in out)
    if selected == 0:
        return [list(t) for t in branches_per_tree], {
            "enabled": True,
            "original": int(sum(len(t) for t in branches_per_tree)),
            "selected": int(sum(len(t) for t in branches_per_tree)),
            "fallback": "empty_selection",
        }
    return out, {
        "enabled": True,
        "original": int(sum(len(t) for t in branches_per_tree)),
        "selected": int(selected),
        "budget": budget,
        "max_depth": max_depth,
        "min_support": min_support,
        "selection": cfg.rule_selection,
    }


def shrink_theta_empirical_bayes(
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_support: np.ndarray,
    n_train: int,
    strength: float,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    prior = np.asarray(class_prior, dtype=np.float64)
    prior = prior / np.maximum(prior.sum(), 1e-12)
    strength = float(max(strength, 0.0))
    if strength <= 0:
        return theta.copy()
    eff_n = np.asarray(branch_support, dtype=np.float64).reshape(-1) * max(int(n_train), 1)
    lam = eff_n / np.maximum(eff_n + strength, 1e-12)
    shrunk = lam[:, None] * theta + (1.0 - lam[:, None]) * prior[None, :]
    return normalize_proba(shrunk)


def aggregate_signed_logit(
    z: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    temperature: float = 1.0,
    reliability: Optional[np.ndarray] = None,
    top_k: int = 0,
) -> np.ndarray:
    """Signed support/opposition aggregation in class logit space.

    A rule supports class k when theta_bk exceeds the empirical class prior and
    opposes it when theta_bk is below that prior.  Averaging by active evidence
    mass keeps correlated rule pools from dominating by sheer count.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    theta = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    z_eff = np.clip(z, 0.0, 1.0)
    if reliability is not None:
        rel = np.asarray(reliability, dtype=np.float64).reshape(1, -1)
        z_eff = z_eff * np.clip(rel, 0.0, 2.0)
    top_k = int(top_k)
    if top_k > 0 and top_k < z_eff.shape[1]:
        idx = np.argpartition(z_eff, -top_k, axis=1)[:, -top_k:]
        sparse = np.zeros_like(z_eff)
        rows = np.arange(z_eff.shape[0])[:, None]
        sparse[rows, idx] = z_eff[rows, idx]
        z_eff = sparse
    contrib = np.log(np.clip(theta, 1e-9, 1.0)) - np.log(prior)
    active_mass = np.maximum(z_eff.sum(axis=1, keepdims=True), 1e-9)
    logits = np.log(prior) + (z_eff @ contrib) / active_mass
    temp = max(float(temperature), 1e-6)
    logits = logits / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(logits)
    return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)


def predict_diff_posterior_signed_logit_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    cfg: RunConfig,
    reliability: Optional[np.ndarray] = None,
    top_k: int = 0,
) -> np.ndarray:
    diff_post = DifferentiablePosterior(
        model.branches,
        p_high=cfg.posterior_p_high,
        p_low=cfg.posterior_p_low,
        tau=cfg.condition_tau,
    )
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        chunks.append(aggregate_signed_logit(
            z,
            theta,
            class_prior,
            temperature=cfg.signed_logit_temperature,
            reliability=reliability,
            top_k=top_k,
        ))
    return normalize_proba(np.vstack(chunks))


def predict_frozen_support_prior_wmean_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    branch_prior: np.ndarray,
    cfg: RunConfig,
) -> np.ndarray:
    branch_prior = np.clip(np.asarray(branch_prior, dtype=np.float64).reshape(1, -1), 1e-6, 1.0 - 1e-6)
    chunks = []
    for sl in iter_slices(len(X), cfg.batch_size):
        X_chunk = X[sl]
        prior_z = np.repeat(branch_prior, len(X_chunk), axis=0)
        post_z = compute_soft_posterior(
            model.branches,
            prior_z,
            X_chunk,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
            tau=cfg.condition_tau,
        )
        chunks.append(aggregate_weighted_mean(post_z, theta))
    return normalize_proba(np.vstack(chunks))


def estimate_feature_group_reliability(
    branches: Sequence,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    theta: np.ndarray,
    cfg: RunConfig,
) -> np.ndarray:
    branches = list(branches)
    if not branches:
        return np.zeros(0, dtype=np.float64)
    y64 = np.asarray(y).ravel().astype(int)
    theta_safe = normalize_proba(np.asarray(theta, dtype=np.float64))
    branch_pred = theta_safe.argmax(axis=1)
    z = compute_condition_activation(branches, X, tau=cfg.condition_tau)
    branch_rel = np.ones(len(branches), dtype=np.float64)
    feat_scores: Dict[int, List[float]] = {}
    for b_idx, branch in enumerate(branches):
        mask = z[:, b_idx] >= 0.5
        if mask.any():
            precision = float(np.mean(y64[mask] == branch_pred[b_idx]))
        else:
            precision = float(theta_safe[b_idx].max())
        branch_rel[b_idx] = precision
        for cond in branch.conditions:
            feat_scores.setdefault(int(cond.feature_idx), []).append(precision)
    feat_rel = {feat: float(np.mean(vals)) for feat, vals in feat_scores.items() if vals}
    out = np.ones(len(branches), dtype=np.float64)
    for b_idx, branch in enumerate(branches):
        feats = [int(c.feature_idx) for c in branch.conditions]
        if feats:
            out[b_idx] = float(np.mean([feat_rel.get(f, branch_rel[b_idx]) for f in feats]))
        else:
            out[b_idx] = branch_rel[b_idx]
    mean = float(np.mean(out)) if len(out) else 1.0
    if mean > 0:
        out = out / mean
    return np.clip(out, 0.25, 1.75)



def estimate_source_calibrated_reliability(
    branches: Sequence,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_support: np.ndarray,
    n_train_total: int,
    cfg: RunConfig,
) -> Tuple[np.ndarray, float]:
    """Auditable branch/source reliability for PPtheta-Post+SourceCal.

    Reliability is built only from explicit rule-source statistics: empirical
    branch support, theta entropy, train-set branch precision, and feature-group
    precision.  The returned scalar source confidence mixes the posterior rule
    prediction with the empirical class prior, so weaker rule sources are
    automatically more conservative without adding an opaque predictor.
    """
    branches = list(branches)
    if not branches:
        return np.zeros(0, dtype=np.float64), 1.0

    theta_safe = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = np.asarray(class_prior, dtype=np.float64).reshape(-1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    support = np.asarray(branch_support, dtype=np.float64).reshape(-1)
    if support.size != len(branches):
        support = np.resize(support, len(branches))

    strength = max(float(cfg.theta_shrinkage_strength), 1.0)
    eff_n = np.maximum(support, 0.0) * max(int(n_train_total), 1)
    support_rel = 0.5 + 0.5 * np.sqrt(eff_n / np.maximum(eff_n + strength, 1e-12))

    k = max(int(n_classes), 2)
    entropy = -np.sum(theta_safe * np.log(np.clip(theta_safe, 1e-12, 1.0)), axis=1) / np.log(k)
    theta_rel = 0.5 + np.clip(1.0 - entropy, 0.0, 1.0)

    y64 = np.asarray(y).ravel().astype(int)
    branch_pred = theta_safe.argmax(axis=1)
    z_train = compute_condition_activation(branches, X, tau=cfg.condition_tau)
    mass = z_train.sum(axis=0)
    correct = (y64[:, None] == branch_pred[None, :]).astype(np.float64)
    fallback_precision = theta_safe[np.arange(len(branches)), branch_pred]
    precision = np.divide(
        (z_train * correct).sum(axis=0),
        np.maximum(mass, 1e-12),
        out=fallback_precision.copy(),
        where=mass > 1e-9,
    )
    prior_at_pred = prior[np.clip(branch_pred, 0, len(prior) - 1)]
    lift = np.clip((precision - prior_at_pred) / np.maximum(1.0 - prior_at_pred, 1e-6), 0.0, 1.0)
    precision_rel = 0.5 + lift

    feature_rel = estimate_feature_group_reliability(
        branches, X, y64, n_classes, theta_safe, cfg,
    )
    rel = support_rel * theta_rel * precision_rel * np.sqrt(np.clip(feature_rel, 0.25, 1.75))
    mean_rel = float(np.mean(rel)) if len(rel) else 1.0
    if mean_rel > 0:
        rel = rel / mean_rel
    rel = np.clip(rel, 0.25, 2.0)

    weights = mass + 1e-6
    source_quality = float(np.average(np.clip(precision, 0.0, 1.0), weights=weights))
    source_conf = float(np.clip(source_quality, 0.65, 0.98))
    return rel.astype(np.float64), source_conf


def predict_source_calibrated_ppost_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_reliability: np.ndarray,
    source_confidence: float,
    cfg: RunConfig,
) -> np.ndarray:
    """PPtheta-Post with explicit source/branch calibration weights."""
    diff_post = DifferentiablePosterior(
        model.branches,
        p_high=cfg.posterior_p_high,
        p_low=cfg.posterior_p_low,
        tau=cfg.condition_tau,
    )
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1e-12)
    rel = np.asarray(branch_reliability, dtype=np.float64).reshape(1, -1)
    rel = np.clip(rel, 0.0, 2.0)
    source_conf = float(np.clip(source_confidence, 0.0, 1.0))

    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        margin = compute_condition_activation(model.branches, X[sl], tau=cfg.condition_tau)
        z_eff = z * rel * (0.5 + 0.5 * margin)
        p_rule = aggregate_weighted_mean(z_eff, theta)
        p_mix = source_conf * p_rule + (1.0 - source_conf) * prior
        chunks.append(p_mix)
    return normalize_proba(np.vstack(chunks))



def _group_selected_branches(selected: Sequence, n_trees: int) -> List[List]:
    out = [[] for _ in range(max(int(n_trees), 1))]
    for br in selected:
        tree_id = int(getattr(br, "tree_id", 0))
        while tree_id >= len(out):
            out.append([])
        out[tree_id].append(copy.deepcopy(br))
    return out


def select_posterior_utility_branches(
    branches_per_tree: Sequence[Sequence],
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    cfg: RunConfig,
) -> Tuple[List[List], Dict[str, Any]]:
    """Select rules by PPtheta posterior utility instead of source purity alone."""
    flat = [br for tree in branches_per_tree for br in tree]
    original = len(flat)
    if original == 0:
        return [list(t) for t in branches_per_tree], {"enabled": True, "original": 0, "selected": 0}

    budget = int(cfg.rule_budget) if int(cfg.rule_budget) > 0 else max(64, int(round(original * 0.5)))
    budget = max(1, min(int(budget), original))
    if budget >= original:
        return [list(t) for t in branches_per_tree], {"enabled": True, "original": original, "selected": original, "budget": budget}

    y64 = np.asarray(y).ravel().astype(int)
    prior = empirical_class_prior(y64, n_classes)
    theta = build_theta_matrix(flat, n_classes)
    theta = normalize_proba(theta)
    cond_z = compute_condition_activation(flat, X, tau=cfg.condition_tau)
    post_z = compute_soft_posterior(
        flat,
        np.clip(cond_z, 1e-6, 1.0 - 1e-6),
        X,
        p_high=cfg.posterior_p_high,
        p_low=cfg.posterior_p_low,
        tau=cfg.condition_tau,
    )
    pred = theta.argmax(axis=1)
    mass = post_z.sum(axis=0)
    correct = (y64[:, None] == pred[None, :]).astype(np.float64)
    precision = np.divide(
        (post_z * correct).sum(axis=0),
        np.maximum(mass, 1e-12),
        out=theta[np.arange(original), pred].copy(),
        where=mass > 1e-9,
    )
    k = max(int(n_classes), 2)
    entropy = -np.sum(theta * np.log(np.clip(theta, 1e-12, 1.0)), axis=1) / np.log(k)
    specificity = np.clip(1.0 - entropy, 0.0, 1.0)
    prior_at_pred = prior[np.clip(pred, 0, len(prior) - 1)]
    lift = np.clip((precision - prior_at_pred) / np.maximum(1.0 - prior_at_pred, 1e-6), -1.0, 1.0)
    support = mass / max(len(X), 1)
    depth = np.array([max(len(getattr(br, "conditions", [])), 1) for br in flat], dtype=np.float64)
    score = lift * (0.25 + specificity) * np.log1p(mass) / np.sqrt(depth)
    score = score + 0.05 * np.sqrt(np.clip(support, 0.0, 1.0))

    ranked = sorted(range(original), key=lambda i: float(score[i]), reverse=True)
    if cfg.rule_selection == "diverse":
        buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        for idx in ranked:
            br = flat[idx]
            feats = tuple(sorted({int(c.feature_idx) for c in getattr(br, "conditions", [])})[:3])
            buckets.setdefault((int(pred[idx]), feats), []).append(idx)
        selected_idx: List[int] = []
        keys = sorted(buckets, key=lambda key: float(score[buckets[key][0]]), reverse=True)
        while keys and len(selected_idx) < budget:
            next_keys = []
            for key in keys:
                if buckets[key] and len(selected_idx) < budget:
                    selected_idx.append(buckets[key].pop(0))
                if buckets[key]:
                    next_keys.append(key)
            keys = next_keys
    else:
        selected_idx = ranked[:budget]

    selected = [flat[i] for i in selected_idx]
    refine_branch_class_proportions(selected, X, y64, n_classes)
    return _group_selected_branches(selected, len(branches_per_tree)), {
        "enabled": True,
        "original": original,
        "selected": len(selected),
        "budget": budget,
        "selection": "posterior_utility",
    }


def aggregate_evidence_logit(
    z: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    alpha: np.ndarray,
    class_bias: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    top_k: int = 0,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    theta = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(1, -1)
    z_eff = np.clip(z, 0.0, 1.0) * np.clip(alpha, 0.0, 10.0)
    top_k = int(top_k)
    if top_k > 0 and top_k < z_eff.shape[1]:
        idx = np.argpartition(z_eff, -top_k, axis=1)[:, -top_k:]
        sparse = np.zeros_like(z_eff)
        rows = np.arange(z_eff.shape[0])[:, None]
        sparse[rows, idx] = z_eff[rows, idx]
        z_eff = sparse
    contrib = np.log(np.clip(theta, 1e-9, 1.0)) - np.log(prior)
    active_mass = np.maximum(z_eff.sum(axis=1, keepdims=True), 1e-9)
    logits = np.log(prior) + (z_eff @ contrib) / active_mass
    if class_bias is not None:
        logits = logits + np.asarray(class_bias, dtype=np.float64).reshape(1, -1)
    temp = max(float(temperature), 1e-6)
    logits = logits / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def learn_evidence_logit_params(
    z_train: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
    l1: float = 1e-4,
    teacher_proba: Optional[np.ndarray] = None,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_br = z_train.shape[1]
    z_t = torch.tensor(z_train, dtype=torch.float32)
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior_np = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior_np = np.clip(prior_np / np.maximum(prior_np.sum(), 1e-12), 1e-9, 1.0)
    contrib_t = torch.tensor(
        np.log(np.clip(theta_np, 1e-9, 1.0)) - np.log(prior_np),
        dtype=torch.float32,
    )
    prior_log_t = torch.tensor(np.log(prior_np), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y_train).ravel(), dtype=torch.long)
    teacher_t = None
    if teacher_proba is not None and distill_weight > 0:
        teacher_np = normalize_proba(np.asarray(teacher_proba, dtype=np.float64))
        if teacher_np.shape == (len(y_t), theta_np.shape[1]):
            teacher_t = torch.tensor(teacher_np, dtype=torch.float32)
    log_a = torch.nn.Parameter(torch.zeros(n_br))
    bias = torch.nn.Parameter(torch.zeros(theta_np.shape[1]))
    temp_raw = torch.nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1.0), dtype=torch.float32))
    opt = torch.optim.Adam([log_a, bias, temp_raw], lr=lr)
    kd_w = float(np.clip(distill_weight, 0.0, 0.95)) if teacher_t is not None else 0.0
    kd_temp = max(float(distill_temperature), 1e-4)
    for _ in range(max(1, int(epochs))):
        alpha = torch.nn.functional.softplus(log_a)
        z_eff = z_t * alpha.unsqueeze(0)
        active = z_eff.sum(dim=1, keepdim=True).clamp_min(1e-9)
        logits = prior_log_t + (z_eff @ contrib_t) / active + bias.unsqueeze(0)
        temp = torch.nn.functional.softplus(temp_raw).clamp_min(1e-4)
        logp = torch.nn.functional.log_softmax(logits / temp, dim=1)
        ce = torch.nn.functional.nll_loss(logp, y_t)
        loss = ce
        if kd_w > 0.0 and teacher_t is not None:
            kd_logp = torch.nn.functional.log_softmax(logits / (temp * kd_temp), dim=1)
            kd = -(teacher_t * kd_logp).sum(dim=1).mean() * (kd_temp ** 2)
            loss = (1.0 - kd_w) * ce + kd_w * kd
        if l1 > 0:
            loss = loss + float(l1) * alpha.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (
        torch.nn.functional.softplus(log_a).detach().cpu().numpy(),
        bias.detach().cpu().numpy(),
        float(torch.nn.functional.softplus(temp_raw).detach().cpu().item()),
    )




def aggregate_evidence_layer_v2(
    z: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_alpha: np.ndarray,
    class_reliability: Optional[np.ndarray] = None,
    class_bias: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    top_k: int = 0,
) -> np.ndarray:
    """Evidence-logit aggregation with branch and class-specific reliability.

    This keeps the same PPtheta-Post audit trace (z, theta, branch rules), but
    lets the final posterior evidence layer learn how reliable each branch is
    overall and for each class.  It is still a linear aggregation of explicit
    rule evidence in log-odds space.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    theta = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    branch_alpha = np.asarray(branch_alpha, dtype=np.float64).reshape(1, -1)
    z_eff = np.clip(z, 0.0, 1.0) * np.clip(branch_alpha, 0.0, 10.0)
    top_k = int(top_k)
    if top_k > 0 and top_k < z_eff.shape[1]:
        idx = np.argpartition(z_eff, -top_k, axis=1)[:, -top_k:]
        sparse = np.zeros_like(z_eff)
        rows = np.arange(z_eff.shape[0])[:, None]
        sparse[rows, idx] = z_eff[rows, idx]
        z_eff = sparse
    contrib = np.log(np.clip(theta, 1e-9, 1.0)) - np.log(prior)
    if class_reliability is not None:
        rel = np.asarray(class_reliability, dtype=np.float64)
        if rel.shape == contrib.shape:
            contrib = contrib * np.clip(rel, 0.0, 5.0)
    active_mass = np.maximum(z_eff.sum(axis=1, keepdims=True), 1e-9)
    logits = np.log(prior) + (z_eff @ contrib) / active_mass
    if class_bias is not None:
        logits = logits + np.asarray(class_bias, dtype=np.float64).reshape(1, -1)
    temp = max(float(temperature), 1e-6)
    logits = logits / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def learn_evidence_layer_v2_params(
    z_train: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
    l1: float = 5e-5,
    balanced_weight: float = 0.35,
    brier_weight: float = 0.05,
    soft_mcc_weight: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Learn the unified Evidence Layer v2 calibration parameters.

    Objective: CE + class-balanced CE + Brier calibration + binary soft-MCC,
    plus light reliability regularization toward one.  The learned parameters
    are auditable branch reliabilities, class-specific branch reliabilities,
    a class bias, and a temperature.
    """
    z_np = np.asarray(z_train, dtype=np.float64)
    if z_np.ndim != 2:
        raise ValueError("z_train must be a 2D branch-activation matrix")
    n_br = z_np.shape[1]
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    n_classes = int(theta_np.shape[1])
    prior_np = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior_np = np.clip(prior_np / np.maximum(prior_np.sum(), 1e-12), 1e-9, 1.0)
    y_np = np.asarray(y_train).ravel().astype(int)

    z_t = torch.tensor(z_np, dtype=torch.float32)
    contrib_t = torch.tensor(
        np.log(np.clip(theta_np, 1e-9, 1.0)) - np.log(prior_np),
        dtype=torch.float32,
    )
    prior_log_t = torch.tensor(np.log(prior_np), dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    onehot_t = torch.nn.functional.one_hot(y_t, num_classes=n_classes).float()

    counts = np.bincount(np.clip(y_np, 0, n_classes - 1), minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.maximum(weights.mean(), 1e-12)
    class_weight_t = torch.tensor(weights, dtype=torch.float32)

    init = float(np.log(np.exp(1.0) - 1.0))
    log_branch_alpha = torch.nn.Parameter(torch.full((n_br,), init, dtype=torch.float32))
    log_class_rel = torch.nn.Parameter(torch.full((n_br, n_classes), init, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))
    temp_raw = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    opt = torch.optim.Adam([log_branch_alpha, log_class_rel, bias, temp_raw], lr=lr)

    for _ in range(max(1, int(epochs))):
        branch_alpha = torch.nn.functional.softplus(log_branch_alpha)
        class_rel = torch.nn.functional.softplus(log_class_rel)
        z_eff = z_t * branch_alpha.unsqueeze(0)
        active = z_eff.sum(dim=1, keepdim=True).clamp_min(1e-9)
        logits = prior_log_t + (z_eff @ (contrib_t * class_rel)) / active + bias.unsqueeze(0)
        temp = torch.nn.functional.softplus(temp_raw).clamp_min(1e-4)
        logp = torch.nn.functional.log_softmax(logits / temp, dim=1)
        p = torch.exp(logp)
        ce = torch.nn.functional.nll_loss(logp, y_t)
        ce_bal = torch.nn.functional.nll_loss(logp, y_t, weight=class_weight_t)
        brier = ((p - onehot_t) ** 2).sum(dim=1).mean()
        soft_mcc_loss = torch.tensor(0.0, dtype=torch.float32)
        if n_classes == 2:
            y1 = onehot_t[:, 1]
            p1 = p[:, 1]
            tp = (p1 * y1).sum()
            tn = ((1.0 - p1) * (1.0 - y1)).sum()
            fp = (p1 * (1.0 - y1)).sum()
            fn = ((1.0 - p1) * y1).sum()
            denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            soft_mcc = (tp * tn - fp * fn) / denom.clamp_min(1e-8)
            soft_mcc_loss = 1.0 - soft_mcc
        reg = float(l1) * 0.5 * (branch_alpha.mean() + class_rel.mean())
        reg = reg + 1e-3 * ((class_rel - 1.0) ** 2).mean()
        loss = ce + float(balanced_weight) * ce_bal + float(brier_weight) * brier
        loss = loss + float(soft_mcc_weight) * soft_mcc_loss + reg
        opt.zero_grad()
        loss.backward()
        opt.step()

    return (
        torch.nn.functional.softplus(log_branch_alpha).detach().cpu().numpy(),
        torch.nn.functional.softplus(log_class_rel).detach().cpu().numpy(),
        bias.detach().cpu().numpy(),
        float(torch.nn.functional.softplus(temp_raw).detach().cpu().item()),
    )


def predict_evidence_layer_v2_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_alpha: np.ndarray,
    class_reliability: np.ndarray,
    class_bias: np.ndarray,
    temperature: float,
    cfg: RunConfig,
    top_k: int = 0,
    use_model_reliability: bool = True,
) -> np.ndarray:
    diff_post = make_diff_posterior_for_model(
        model, cfg, use_model_reliability=use_model_reliability,
    )
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        chunks.append(aggregate_evidence_layer_v2(
            z, theta, class_prior, branch_alpha, class_reliability,
            class_bias, temperature, top_k=top_k,
        ))
    return normalize_proba(np.vstack(chunks))


def split_evidence_fit_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y).ravel().astype(int)
    n = len(y_arr)
    counts = np.bincount(np.clip(y_arr, 0, max(int(n_classes), 1) - 1), minlength=n_classes)
    if n < 100 or np.any(counts < 2):
        return X, y_arr, X, y_arr
    n_val = min(1000, max(100, int(round(0.2 * n))))
    n_val = min(n_val, max(1, n // 2))
    if n_val < n_classes:
        return X, y_arr, X, y_arr
    idx = np.arange(n)
    fit_idx, val_idx = train_test_split(
        idx,
        test_size=int(n_val),
        random_state=int(seed),
        stratify=y_arr,
    )
    return X[np.sort(fit_idx)], y_arr[np.sort(fit_idx)], X[np.sort(val_idx)], y_arr[np.sort(val_idx)]


def operating_point_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    metrics = compute_metrics(y_true, proba, n_classes, no_roc_auc=True)
    return float(
        metrics["mcc"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.05 * metrics["f1_macro"]
        - 0.02 * metrics["log_loss"]
    )


def tune_binary_threshold_operating_score(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
) -> Tuple[float, float]:
    base_score = operating_point_score(y_true, proba, n_classes)
    if int(n_classes) != 2:
        return 0.5, base_score
    y = np.asarray(y_true).ravel().astype(int)
    p1 = np.clip(normalize_proba(proba)[:, 1], 1e-9, 1.0 - 1e-9)
    grid = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 199),
        np.quantile(p1, np.linspace(0.02, 0.98, 97)),
    ]))
    best_thr, best_score = 0.5, base_score
    for thr in grid:
        shifted = apply_binary_threshold_shift(proba, float(thr))
        score = operating_point_score(y, shifted, n_classes)
        if score > best_score + 1e-12:
            best_thr, best_score = float(thr), float(score)
    return best_thr, best_score





def tune_binary_threshold_sensitivity_floor(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
    specificity_floor: float = 0.92,
) -> Tuple[float, Dict[str, float]]:
    """Select a validation threshold that maximizes sensitivity subject to specificity.

    This defines the clinical high-recall operating point used in the reviewer
    response sweep.  Ties prefer higher MCC and then lower predicted-positive
    rate, which keeps the threshold from becoming clinically unusable.
    """
    base = binary_operating_stats(y_true, proba, n_classes)
    if int(n_classes) != 2:
        return 0.5, base
    y = np.asarray(y_true).ravel().astype(int)
    p1 = np.clip(normalize_proba(proba)[:, 1], 1e-9, 1.0 - 1e-9)
    grid = np.unique(np.concatenate([
        np.linspace(0.005, 0.995, 249),
        np.quantile(p1, np.linspace(0.005, 0.995, 199)),
    ]))
    best_thr = 0.5
    best_stats = base
    best_tuple = (
        -np.inf if base.get("specificity", 0.0) < specificity_floor else base.get("sensitivity", -np.inf),
        base.get("mcc", -np.inf),
        -base.get("pred_positive_rate", np.inf),
    )
    for thr in grid:
        shifted = apply_binary_threshold_shift(proba, float(thr))
        stats = binary_operating_stats(y, shifted, n_classes)
        if stats["specificity"] < float(specificity_floor):
            continue
        cand = (stats["sensitivity"], stats["mcc"], -stats["pred_positive_rate"])
        if cand > best_tuple:
            best_thr, best_stats, best_tuple = float(thr), stats, cand
    best_stats = dict(best_stats)
    best_stats["sensitivity_floor"] = float(specificity_floor)
    best_stats["selected_threshold"] = float(best_thr)
    return float(best_thr), best_stats

def _softplus_inverse(value: float) -> float:
    value = float(max(value, 1e-8))
    if value > 20.0:
        return value
    return float(np.log(np.expm1(value)))


def aggregate_teacher_anchored_proba(
    teacher_proba: np.ndarray,
    rule_proba: np.ndarray,
    class_prior: np.ndarray,
    beta_teacher: float,
    beta_rule: float,
    class_bias: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Teacher prior plus auditable PPtheta-Post rule-evidence correction.

    beta_teacher=1, beta_rule=0 recovers the teacher probabilities.  The rule
    term is a log posterior correction relative to the empirical class prior, so
    branch evidence can move the teacher prior without double-counting the prior.
    """
    teacher = normalize_proba(np.asarray(teacher_proba, dtype=np.float64))
    rule = normalize_proba(np.asarray(rule_proba, dtype=np.float64))
    if teacher.shape != rule.shape:
        raise ValueError(f"teacher/rule proba shape mismatch: {teacher.shape} vs {rule.shape}")
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    teacher_log = np.log(np.clip(teacher, 1e-9, 1.0))
    rule_correction = np.log(np.clip(rule, 1e-9, 1.0)) - np.log(prior)
    logits = float(beta_teacher) * teacher_log + float(beta_rule) * rule_correction
    if class_bias is not None:
        logits = logits + np.asarray(class_bias, dtype=np.float64).reshape(1, -1)
    temp = max(float(temperature), 1e-6)
    logits = logits / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def learn_teacher_anchor_params(
    teacher_proba: np.ndarray,
    rule_proba: np.ndarray,
    class_prior: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 160,
    lr: float = 0.01,
    balanced_weight: float = 0.10,
    brier_weight: float = 0.15,
    soft_mcc_weight: float = 0.25,
    distill_weight: float = 0.20,
    distill_temperature: float = 2.0,
    l2: float = 1e-3,
) -> Tuple[float, float, np.ndarray, float]:
    """Learn a conservative teacher/rule product-of-experts calibration.

    The teacher keeps strong predictive performance; PPtheta-Post contributes an
    explicit log-evidence correction from the same auditable branch trace.
    """
    teacher_np = normalize_proba(np.asarray(teacher_proba, dtype=np.float64))
    rule_np = normalize_proba(np.asarray(rule_proba, dtype=np.float64))
    if teacher_np.shape != rule_np.shape:
        raise ValueError(f"teacher/rule proba shape mismatch: {teacher_np.shape} vs {rule_np.shape}")
    y_np = np.asarray(y_train).ravel().astype(int)
    n_classes = int(rule_np.shape[1])
    prior_np = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior_np = np.clip(prior_np / np.maximum(prior_np.sum(), 1e-12), 1e-9, 1.0)

    teacher_t = torch.tensor(teacher_np, dtype=torch.float32)
    teacher_log_t = torch.log(teacher_t.clamp(1e-9, 1.0))
    rule_t = torch.tensor(rule_np, dtype=torch.float32)
    prior_log_t = torch.tensor(np.log(prior_np), dtype=torch.float32)
    rule_corr_t = torch.log(rule_t.clamp(1e-9, 1.0)) - prior_log_t
    y_t = torch.tensor(y_np, dtype=torch.long)
    onehot_t = torch.nn.functional.one_hot(y_t, num_classes=n_classes).float()

    counts = np.bincount(np.clip(y_np, 0, n_classes - 1), minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.maximum(weights.mean(), 1e-12)
    class_weight_t = torch.tensor(weights, dtype=torch.float32)

    raw_beta_teacher = torch.nn.Parameter(torch.tensor(_softplus_inverse(1.0), dtype=torch.float32))
    raw_beta_rule = torch.nn.Parameter(torch.tensor(_softplus_inverse(0.25), dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))
    raw_temp = torch.nn.Parameter(torch.tensor(_softplus_inverse(1.0), dtype=torch.float32))
    opt = torch.optim.Adam([raw_beta_teacher, raw_beta_rule, bias, raw_temp], lr=lr)
    kd_temp = max(float(distill_temperature), 1e-4)

    for _ in range(max(1, int(epochs))):
        beta_teacher = torch.nn.functional.softplus(raw_beta_teacher)
        beta_rule = torch.nn.functional.softplus(raw_beta_rule)
        temp = torch.nn.functional.softplus(raw_temp).clamp_min(1e-4)
        logits = beta_teacher * teacher_log_t + beta_rule * rule_corr_t + bias.unsqueeze(0)
        logp = torch.nn.functional.log_softmax(logits / temp, dim=1)
        p = torch.exp(logp)
        ce = torch.nn.functional.nll_loss(logp, y_t)
        ce_bal = torch.nn.functional.nll_loss(logp, y_t, weight=class_weight_t)
        brier = ((p - onehot_t) ** 2).sum(dim=1).mean()
        soft_mcc_loss = torch.tensor(0.0, dtype=torch.float32)
        if n_classes == 2:
            y1 = onehot_t[:, 1]
            p1 = p[:, 1]
            tp = (p1 * y1).sum()
            tn = ((1.0 - p1) * (1.0 - y1)).sum()
            fp = (p1 * (1.0 - y1)).sum()
            fn = ((1.0 - p1) * y1).sum()
            denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            soft_mcc = (tp * tn - fp * fn) / denom.clamp_min(1e-8)
            soft_mcc_loss = 1.0 - soft_mcc
        kd_logp = torch.nn.functional.log_softmax(logits / (temp * kd_temp), dim=1)
        kd = -(teacher_t * kd_logp).sum(dim=1).mean() * (kd_temp ** 2)
        reg = float(l2) * (
            (beta_teacher - 1.0) ** 2
            + beta_rule ** 2
            + (bias ** 2).mean()
            + (temp - 1.0) ** 2
        )
        loss = ce + float(balanced_weight) * ce_bal
        loss = loss + float(brier_weight) * brier + float(soft_mcc_weight) * soft_mcc_loss
        loss = loss + float(distill_weight) * kd + reg
        opt.zero_grad()
        loss.backward()
        opt.step()

    beta_teacher = float(torch.nn.functional.softplus(raw_beta_teacher).detach().cpu().item())
    beta_rule = float(torch.nn.functional.softplus(raw_beta_rule).detach().cpu().item())
    temperature = float(torch.nn.functional.softplus(raw_temp).detach().cpu().item())
    return (
        float(np.clip(beta_teacher, 0.25, 3.0)),
        float(np.clip(beta_rule, 0.0, 2.0)),
        bias.detach().cpu().numpy(),
        float(np.clip(temperature, 0.50, 3.0)),
    )


def binary_operating_stats(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> Dict[str, float]:
    metrics = compute_metrics(y_true, proba, n_classes, no_roc_auc=True)
    out = dict(metrics)
    proba = normalize_proba(proba)
    pred = np.argmax(proba, axis=1)
    y = np.asarray(y_true).ravel().astype(int)
    out["pred_positive_rate"] = float(np.mean(pred == 1)) if int(n_classes) == 2 else float("nan")
    out["positive_prevalence"] = float(np.mean(y == 1)) if int(n_classes) == 2 else float("nan")
    if int(n_classes) == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out["sensitivity"] = float(tp / max(tp + fn, 1))
        out["specificity"] = float(tn / max(tn + fp, 1))
    else:
        out["sensitivity"] = float("nan")
        out["specificity"] = float("nan")
    return out


def conservative_teacher_anchor_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
) -> Tuple[float, str, Dict[str, float]]:
    base = binary_operating_stats(y_true, proba, n_classes)
    diagnostics = {
        "teacher_anchor_selected_mode": "calibrated",
        "teacher_anchor_threshold": 0.5,
        "teacher_anchor_val_mcc_calibrated": float(base["mcc"]),
        "teacher_anchor_val_log_loss_calibrated": float(base["log_loss"]),
        "teacher_anchor_val_specificity_calibrated": float(base["specificity"]),
        "teacher_anchor_val_pred_positive_rate_calibrated": float(base["pred_positive_rate"]),
    }
    if int(n_classes) != 2:
        diagnostics.update({
            "teacher_anchor_val_mcc_threshold": float(base["mcc"]),
            "teacher_anchor_val_log_loss_threshold": float(base["log_loss"]),
            "teacher_anchor_val_specificity_threshold": float(base["specificity"]),
            "teacher_anchor_val_pred_positive_rate_threshold": float(base["pred_positive_rate"]),
        })
        return 0.5, "calibrated", diagnostics

    def score(stats: Dict[str, float]) -> float:
        return float(
            stats["mcc"]
            + 0.10 * stats["balanced_accuracy"]
            + 0.04 * stats["f1_macro"]
            - 0.08 * stats["log_loss"]
        )

    p1 = np.clip(normalize_proba(proba)[:, 1], 1e-9, 1.0 - 1e-9)
    grid = np.unique(np.concatenate([
        np.linspace(0.02, 0.98, 193),
        np.quantile(p1, np.linspace(0.03, 0.97, 95)),
    ]))
    base_score = score(base)
    best_thr, best_stats, best_score = 0.5, base, base_score
    spec_floor = max(0.0, float(base["specificity"]) - 0.015)
    rate_cap = max(
        float(base["pred_positive_rate"]) * 1.25,
        float(base["positive_prevalence"]) + 0.03,
    )
    ll_cap = float(base["log_loss"]) + 0.005
    acc_floor = float(base["accuracy"]) - 0.002
    mcc_floor = float(base["mcc"]) - 0.001
    for thr in grid:
        shifted = apply_binary_threshold_shift(proba, float(thr))
        stats = binary_operating_stats(y_true, shifted, n_classes)
        if stats["specificity"] < spec_floor:
            continue
        if stats["pred_positive_rate"] > rate_cap:
            continue
        if stats["log_loss"] > ll_cap:
            continue
        if stats["accuracy"] < acc_floor:
            continue
        if stats["mcc"] < mcc_floor:
            continue
        cand_score = score(stats)
        if cand_score > best_score + 1e-4:
            best_thr, best_stats, best_score = float(thr), stats, cand_score

    mode = "threshold" if abs(best_thr - 0.5) > 1e-12 else "calibrated"
    diagnostics.update({
        "teacher_anchor_selected_mode": mode,
        "teacher_anchor_threshold": float(best_thr),
        "teacher_anchor_val_mcc_threshold": float(best_stats["mcc"]),
        "teacher_anchor_val_log_loss_threshold": float(best_stats["log_loss"]),
        "teacher_anchor_val_specificity_threshold": float(best_stats["specificity"]),
        "teacher_anchor_val_pred_positive_rate_threshold": float(best_stats["pred_positive_rate"]),
    })
    return float(best_thr), mode, diagnostics


def predict_teacher_anchored_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_alpha: np.ndarray,
    class_reliability: np.ndarray,
    rule_bias: np.ndarray,
    rule_temperature: float,
    teacher_proba: np.ndarray,
    beta_teacher: float,
    beta_rule: float,
    anchor_bias: np.ndarray,
    anchor_temperature: float,
    cfg: RunConfig,
    top_k: int = 0,
    use_model_reliability: bool = True,
) -> np.ndarray:
    diff_post = make_diff_posterior_for_model(
        model, cfg, use_model_reliability=use_model_reliability,
    )
    teacher = normalize_proba(np.asarray(teacher_proba, dtype=np.float64))
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        rule_proba = aggregate_evidence_layer_v2(
            z, theta, class_prior, branch_alpha, class_reliability,
            rule_bias, rule_temperature, top_k=top_k,
        )
        chunks.append(aggregate_teacher_anchored_proba(
            teacher[sl], rule_proba, class_prior, beta_teacher, beta_rule,
            anchor_bias, anchor_temperature,
        ))
    return normalize_proba(np.vstack(chunks))



def _branch_family_groups(branches: Sequence, theta: np.ndarray) -> List[np.ndarray]:
    """Group redundant branches into auditable rule families.

    Families are intentionally simple and inspectable: same predicted class and
    same set of involved feature indices.  This collapses correlated branches
    that differ only by nearby thresholds while preserving the rule evidence
    surface at a family level.
    """
    theta_safe = normalize_proba(np.asarray(theta, dtype=np.float64))
    pred = theta_safe.argmax(axis=1) if theta_safe.size else np.zeros(len(branches), dtype=int)
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for idx, br in enumerate(branches):
        feats = tuple(sorted({int(c.feature_idx) for c in getattr(br, "conditions", [])}))
        if not feats:
            feats = (-1,)
        key = (int(pred[idx]) if idx < len(pred) else 0, feats)
        buckets.setdefault(key, []).append(idx)
    groups = [np.asarray(v, dtype=np.int64) for _, v in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return groups or [np.arange(len(branches), dtype=np.int64)]


def _reduce_family_theta(theta: np.ndarray, groups: Sequence[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    theta_safe = normalize_proba(np.asarray(theta, dtype=np.float64))
    if weights is None:
        weights = np.ones(theta_safe.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    fam = []
    for g in groups:
        gg = np.asarray(g, dtype=np.int64)
        w = np.clip(weights[gg], 1e-9, None)
        val = (theta_safe[gg] * w[:, None]).sum(axis=0) / np.maximum(w.sum(), 1e-12)
        fam.append(val)
    return normalize_proba(np.vstack(fam))


def _reduce_family_z(z: np.ndarray, groups: Sequence[np.ndarray], mode: str = "max") -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    out = np.zeros((z.shape[0], len(groups)), dtype=np.float64)
    for j, g in enumerate(groups):
        gg = np.asarray(g, dtype=np.int64)
        vals = z[:, gg]
        if mode == "mean":
            out[:, j] = vals.mean(axis=1)
        else:
            out[:, j] = vals.max(axis=1)
    return np.clip(out, 0.0, 1.0)


def _fit_binary_isotonic(y_true: np.ndarray, proba: np.ndarray, n_classes: int):
    if int(n_classes) != 2:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return None
    y = np.asarray(y_true).ravel().astype(int)
    p1 = normalize_proba(proba)[:, 1]
    if len(np.unique(y)) < 2 or len(np.unique(np.round(p1, 8))) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
    iso.fit(p1, y)
    return iso


def _apply_binary_isotonic(iso, proba: np.ndarray) -> np.ndarray:
    p = normalize_proba(proba)
    if iso is None or p.shape[1] != 2:
        return p
    p1 = np.clip(iso.predict(p[:, 1]), 1e-6, 1.0 - 1e-6)
    return np.column_stack([1.0 - p1, p1])


def _binary_logit_from_proba(proba: np.ndarray) -> np.ndarray:
    p = normalize_proba(proba)
    if p.shape[1] != 2:
        return np.log(np.clip(p, 1e-9, 1.0))
    p1 = np.clip(p[:, 1], 1e-9, 1.0 - 1e-9)
    return np.log(p1 / (1.0 - p1))


def _binary_proba_from_logit(logit: np.ndarray) -> np.ndarray:
    l = np.clip(np.asarray(logit, dtype=np.float64).reshape(-1), -50.0, 50.0)
    p1 = 1.0 / (1.0 + np.exp(-l))
    return np.column_stack([1.0 - p1, p1])


def _combine_binary_residual(
    base_proba: np.ndarray,
    rule_proba: np.ndarray,
    lam: float,
    gate: Optional[np.ndarray] = None,
    delta_clip: float = 2.0,
) -> np.ndarray:
    base_logit = _binary_logit_from_proba(base_proba)
    rule_logit = _binary_logit_from_proba(rule_proba)
    if np.ndim(base_logit) != 1 or np.ndim(rule_logit) != 1:
        return normalize_proba((1.0 - lam) * normalize_proba(base_proba) + lam * normalize_proba(rule_proba))
    delta = np.clip(rule_logit - base_logit, -float(delta_clip), float(delta_clip))
    lam_vec = float(np.clip(lam, 0.0, 1.0))
    if gate is not None:
        lam_vec = lam_vec * np.asarray(gate, dtype=np.float64).reshape(-1)
    return _binary_proba_from_logit(base_logit + lam_vec * delta)


def _evidence_concentration(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    total = z.sum(axis=1)
    top = z.max(axis=1) if z.shape[1] else np.zeros(z.shape[0])
    return top / np.maximum(total, eps)


def _family_entropy(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    total = z.sum(axis=1, keepdims=True)
    p = z / np.maximum(total, eps)
    ent = -(p * np.log(np.clip(p, eps, 1.0))).sum(axis=1)
    denom = np.log(max(z.shape[1], 2))
    return ent / max(denom, eps)


def _threshold_grid_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([0.0], dtype=np.float64)
    return np.unique(np.concatenate([
        np.quantile(scores, np.linspace(0.10, 0.90, 9)),
        np.linspace(float(scores.min()), float(scores.max()), 7),
    ]))


def _pick_residual_gate(
    y_val: np.ndarray,
    base_val: np.ndarray,
    rule_val: np.ndarray,
    n_classes: int,
    confidence: Optional[np.ndarray] = None,
    mode: str = "bounded",
) -> Tuple[float, float, Dict[str, float]]:
    base_stats = binary_operating_stats(y_val, base_val, n_classes)
    best = {"score": -np.inf, "lam": 0.0, "thr": -np.inf, "stats": base_stats}
    lam_grid = np.linspace(0.0, 0.45 if mode == "bounded" else 0.90, 10)
    if confidence is None:
        thr_grid = np.array([-np.inf], dtype=np.float64)
        conf = np.ones(len(y_val), dtype=np.float64)
    else:
        conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
        thr_grid = _threshold_grid_from_scores(conf)
    for lam in lam_grid:
        for thr in thr_grid:
            gate = (conf >= float(thr)).astype(np.float64) if confidence is not None else None
            cand = _combine_binary_residual(base_val, rule_val, float(lam), gate=gate, delta_clip=1.5 if mode == "bounded" else 2.5)
            stats = binary_operating_stats(y_val, cand, n_classes)
            # Keep calibration/accuracy close to base; then prefer MCC and balanced accuracy.
            if stats["log_loss"] > base_stats["log_loss"] + (0.010 if mode == "bounded" else 0.025):
                continue
            if stats["accuracy"] < base_stats["accuracy"] - (0.004 if mode == "bounded" else 0.012):
                continue
            score = stats["mcc"] + 0.08 * stats["balanced_accuracy"] - 0.06 * stats["log_loss"]
            if score > best["score"] + 1e-8:
                best = {"score": float(score), "lam": float(lam), "thr": float(thr), "stats": stats}
    st = best["stats"]
    return float(best["lam"]), float(best["thr"]), {
        "gate_val_score": float(best["score"]),
        "gate_val_mcc": float(st.get("mcc", float("nan"))),
        "gate_val_log_loss": float(st.get("log_loss", float("nan"))),
        "gate_val_accuracy": float(st.get("accuracy", float("nan"))),
        "gate_val_balanced_accuracy": float(st.get("balanced_accuracy", float("nan"))),
    }



def _beta_shrink_family_theta(
    theta: np.ndarray,
    class_prior: np.ndarray,
    support: Optional[np.ndarray] = None,
    strength: Optional[float] = None,
) -> np.ndarray:
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = normalize_proba(np.asarray(class_prior, dtype=np.float64).reshape(1, -1))[0]
    if support is None:
        support_np = np.ones(theta_np.shape[0], dtype=np.float64)
    else:
        support_np = np.asarray(support, dtype=np.float64).reshape(-1)
        if support_np.size != theta_np.shape[0]:
            support_np = np.resize(support_np, theta_np.shape[0])
    if strength is None:
        strength = float(os.environ.get("PPPOST_BAYES_LLR_BETA_STRENGTH", "48"))
    eff = support_np / np.maximum(support_np + float(strength), 1e-8)
    shrunk = eff[:, None] * theta_np + (1.0 - eff[:, None]) * prior[None, :]
    return normalize_proba(shrunk)


def _aggregate_bayes_llr(
    z: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    *,
    top_k: int = 0,
    pos_scale: float = 1.0,
    neg_scale: float = 1.0,
    conflict_penalty: float = 0.0,
    llr_clip: Optional[float] = None,
) -> np.ndarray:
    z_np = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    if z_np.ndim == 1:
        z_np = z_np[np.newaxis, :]
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = normalize_proba(np.asarray(class_prior, dtype=np.float64).reshape(1, -1))[0]
    if theta_np.shape[0] != z_np.shape[1]:
        m = min(theta_np.shape[0], z_np.shape[1])
        theta_np = theta_np[:m]
        z_np = z_np[:, :m]
    if theta_np.shape[1] != 2:
        contrib = np.log(np.clip(theta_np, 1e-8, 1.0)) - np.log(np.clip(prior.reshape(1, -1), 1e-8, 1.0))
        if llr_clip is None:
            llr_clip = float(os.environ.get("PPPOST_BAYES_LLR_CLIP", "2.5"))
        contrib = np.clip(contrib, -float(llr_clip), float(llr_clip))
        logits = np.log(np.clip(prior, 1e-8, 1.0))[None, :] + z_np @ contrib
        return normalize_proba(np.exp(logits - logits.max(axis=1, keepdims=True)))
    p1 = np.clip(theta_np[:, 1], 1e-8, 1.0 - 1e-8)
    pi1 = float(np.clip(prior[1], 1e-8, 1.0 - 1e-8))
    llr = np.log(p1 / (1.0 - p1)) - np.log(pi1 / (1.0 - pi1))
    if llr_clip is None:
        llr_clip = float(os.environ.get("PPPOST_BAYES_LLR_CLIP", "2.5"))
    llr = np.clip(llr, -float(llr_clip), float(llr_clip))
    contrib = z_np * llr.reshape(1, -1)
    if top_k and top_k > 0 and contrib.shape[1] > top_k:
        keep = np.argpartition(np.abs(contrib), -int(top_k), axis=1)[:, -int(top_k):]
        mask = np.zeros_like(contrib, dtype=bool)
        rows = np.arange(contrib.shape[0])[:, None]
        mask[rows, keep] = True
        contrib = np.where(mask, contrib, 0.0)
    pos = np.clip(contrib, 0.0, None).sum(axis=1)
    neg = np.clip(-contrib, 0.0, None).sum(axis=1)
    conflict = np.minimum(pos, neg)
    prior_logit = np.log(pi1 / (1.0 - pi1))
    logit = prior_logit + float(pos_scale) * pos - float(neg_scale) * neg - float(conflict_penalty) * conflict
    return _binary_proba_from_logit(logit)


def _tune_bayes_llr_posneg(
    y_val: np.ndarray,
    z_val: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    n_classes: int,
    *,
    top_k: int = 0,
) -> Tuple[float, float, float, Dict[str, float]]:
    if n_classes != 2:
        return 1.0, 1.0, 0.0, {"bayes_llr_val_mcc": float("nan")}
    best: Optional[Tuple[float, float, float, Dict[str, float], float]] = None
    pos_grid = [0.50, 0.75, 1.00, 1.25, 1.50]
    neg_grid = [0.50, 0.75, 1.00, 1.25, 1.50]
    conflict_grid = [0.0, 0.05, 0.15, 0.30]
    for ps in pos_grid:
        for ns in neg_grid:
            for cp in conflict_grid:
                p = _aggregate_bayes_llr(z_val, theta, class_prior, top_k=top_k, pos_scale=ps, neg_scale=ns, conflict_penalty=cp)
                st = binary_operating_stats(y_val, p, n_classes)
                score = st["mcc"] + 0.05 * st["balanced_accuracy"] - 0.03 * st["log_loss"]
                if best is None or score > best[-1] + 1e-8:
                    best = (float(ps), float(ns), float(cp), st, float(score))
    assert best is not None
    ps, ns, cp, st, score = best
    return ps, ns, cp, {
        "bayes_llr_val_score": float(score),
        "bayes_llr_val_mcc": float(st.get("mcc", float("nan"))),
        "bayes_llr_val_log_loss": float(st.get("log_loss", float("nan"))),
        "bayes_llr_val_balanced_accuracy": float(st.get("balanced_accuracy", float("nan"))),
    }


def _family_llr_feature_matrix(z: np.ndarray, theta: np.ndarray, class_prior: np.ndarray, top_k: int = 0) -> np.ndarray:
    z_np = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    if z_np.ndim == 1:
        z_np = z_np[np.newaxis, :]
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    if theta_np.shape[1] != 2:
        return z_np
    m = min(z_np.shape[1], theta_np.shape[0])
    z_np = z_np[:, :m]
    theta_np = theta_np[:m]
    prior = normalize_proba(np.asarray(class_prior, dtype=np.float64).reshape(1, -1))[0]
    pi = float(np.clip(prior[1], 1e-8, 1.0 - 1e-8))
    p1 = np.clip(theta_np[:, 1], 1e-8, 1.0 - 1e-8)
    llr = np.log(p1 / (1.0 - p1)) - np.log(pi / (1.0 - pi))
    llr = np.clip(llr, -float(os.environ.get("PPPOST_EBM_RESIDUAL_LLR_CLIP", "2.5")), float(os.environ.get("PPPOST_EBM_RESIDUAL_LLR_CLIP", "2.5")))
    feats = z_np * llr.reshape(1, -1)
    if top_k and top_k > 0 and feats.shape[1] > top_k:
        keep = np.argpartition(np.abs(feats), -int(top_k), axis=1)[:, -int(top_k):]
        mask = np.zeros_like(feats, dtype=bool)
        rows = np.arange(feats.shape[0])[:, None]
        mask[rows, keep] = True
        feats = np.where(mask, feats, 0.0)
    return feats


def _fit_ridge_residual(X: np.ndarray, y: np.ndarray, l2: float = 2.0, sample_weight: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_np.ndim == 1:
        X_np = X_np[:, None]
    X_aug = np.column_stack([np.ones(X_np.shape[0], dtype=np.float64), X_np])
    if sample_weight is None:
        w = np.ones(X_aug.shape[0], dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != X_aug.shape[0]:
            w = np.ones(X_aug.shape[0], dtype=np.float64)
    sw = np.sqrt(np.clip(w, 1e-8, None))
    Xw = X_aug * sw[:, None]
    yw = y_np * sw
    reg = np.eye(X_aug.shape[1], dtype=np.float64) * float(l2)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw.T @ Xw + reg, Xw.T @ yw, rcond=None)[0]
    return float(beta[0]), np.asarray(beta[1:], dtype=np.float64)


def _soft_binary_target(y: np.ndarray, teacher: Optional[np.ndarray], true_weight: float = 0.45, eps: float = 1e-3) -> np.ndarray:
    y_np = np.asarray(y).ravel().astype(int)
    hard = np.where(y_np == 1, 1.0 - eps, eps).astype(np.float64)
    if teacher is None:
        return hard
    t = normalize_proba(np.asarray(teacher, dtype=np.float64))
    if t.ndim != 2 or t.shape[1] != 2 or len(t) != len(hard):
        return hard
    p_teacher = np.clip(t[:, 1], eps, 1.0 - eps)
    tw = float(np.clip(true_weight, 0.0, 1.0))
    return np.clip(tw * hard + (1.0 - tw) * p_teacher, eps, 1.0 - eps)


def _residual_target_from_proba(y: np.ndarray, teacher: Optional[np.ndarray], ebm_proba: np.ndarray, true_weight: float = 0.45) -> np.ndarray:
    p_target = _soft_binary_target(y, teacher, true_weight=true_weight)
    target_logit = _binary_logit_from_proba(np.column_stack([1.0 - p_target, p_target]))
    base_logit = _binary_logit_from_proba(ebm_proba)
    if np.ndim(target_logit) != 1 or np.ndim(base_logit) != 1:
        return np.zeros(len(p_target), dtype=np.float64)
    clip = float(os.environ.get("PPPOST_EBM_RESIDUAL_TARGET_CLIP", "3.0"))
    return np.clip(target_logit - base_logit, -clip, clip)


def _ebm_uncertainty_score(ebm_proba: np.ndarray) -> np.ndarray:
    p = normalize_proba(ebm_proba)
    if p.shape[1] != 2:
        return np.ones(p.shape[0], dtype=np.float64)
    p1 = np.clip(p[:, 1], 0.0, 1.0)
    return np.clip(1.0 - 2.0 * np.abs(p1 - 0.5), 0.0, 1.0)


def _fit_residual_family_selection(
    z_val: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    residual_target_val: np.ndarray,
    support_family: np.ndarray,
    top_k: int,
) -> np.ndarray:
    feats = _family_llr_feature_matrix(z_val, theta, class_prior, top_k=0)
    n_fam = feats.shape[1]
    if n_fam == 0:
        return np.array([], dtype=int)
    y = np.asarray(residual_target_val, dtype=np.float64).reshape(-1)
    base_mse = float(np.mean(y * y))
    support = np.asarray(support_family, dtype=np.float64).reshape(-1)
    if support.size != n_fam:
        support = np.ones(n_fam, dtype=np.float64)
    support = support / max(float(np.max(support)), 1.0)
    utilities = np.zeros(n_fam, dtype=np.float64)
    for j in range(n_fam):
        x = feats[:, j]
        denom = float(np.dot(x, x) + float(os.environ.get("PPPOST_EBM_RESIDUAL_UTILITY_L2", "1.0")))
        if denom <= 1e-12:
            continue
        beta = float(np.dot(x, y) / denom)
        pred = beta * x
        mse = float(np.mean((y - pred) ** 2))
        utilities[j] = max(0.0, base_mse - mse) * (0.5 + support[j])
    k = min(max(1, int(top_k)), n_fam)
    positive = np.where(utilities > 0)[0]
    if positive.size == 0:
        positive = np.argsort(-np.var(feats, axis=0))[:k]
    keep = positive[np.argsort(-utilities[positive])[:k]] if positive.size else np.arange(min(k, n_fam))
    return np.sort(np.asarray(keep, dtype=int))



def _fit_clinical_residual_family_selection(
    z_val: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    ebm_val: np.ndarray,
    residual_target_val: np.ndarray,
    y_val: np.ndarray,
    support_family: np.ndarray,
    top_k: int,
) -> np.ndarray:
    feats = _family_llr_feature_matrix(z_val, theta, class_prior, top_k=0)
    n_fam = feats.shape[1]
    if n_fam == 0:
        return np.array([], dtype=int)
    y = np.asarray(y_val).ravel().astype(int)
    target = np.asarray(residual_target_val, dtype=np.float64).reshape(-1)
    base_stats = binary_operating_stats(y, ebm_val, 2)
    support = np.asarray(support_family, dtype=np.float64).reshape(-1)
    if support.size != n_fam:
        support = np.ones(n_fam, dtype=np.float64)
    support = support / max(float(np.max(support)), 1.0)
    utilities = np.zeros(n_fam, dtype=np.float64)
    for j in range(n_fam):
        x = feats[:, j]
        denom = float(np.dot(x, x) + float(os.environ.get("PPPOST_DUAL_RESIDUAL_UTILITY_L2", "1.0")))
        if denom <= 1e-12:
            continue
        beta = float(np.dot(x, target) / denom)
        resid = beta * x
        cand = _combine_ebm_residual_logit(ebm_val, resid, scale=0.75, gate=np.ones(len(y), dtype=np.float64))
        st = binary_operating_stats(y, cand, 2)
        d_mcc = st["mcc"] - base_stats["mcc"]
        d_sens = st["sensitivity"] - base_stats["sensitivity"]
        brier_pen = max(0.0, st["brier_score"] - base_stats["brier_score"])
        ll_pen = max(0.0, st["log_loss"] - base_stats["log_loss"])
        utilities[j] = (d_mcc + 0.35 * d_sens - 0.75 * brier_pen - 0.20 * ll_pen) * (0.5 + support[j])
    k = min(max(1, int(top_k)), n_fam)
    positive = np.where(utilities > 0)[0]
    if positive.size == 0:
        return _fit_residual_family_selection(z_val, theta, class_prior, residual_target_val, support_family, top_k=k)
    keep = positive[np.argsort(-utilities[positive])[:k]]
    return np.sort(np.asarray(keep, dtype=int))


def _teacher_confidence_blend_target(
    y: np.ndarray,
    teacher: Optional[np.ndarray],
    default_true_weight: float,
    eps: float = 1e-3,
) -> np.ndarray:
    y_np = np.asarray(y).ravel().astype(int)
    hard = np.where(y_np == 1, 1.0 - eps, eps).astype(np.float64)
    if teacher is None:
        return hard
    t = normalize_proba(np.asarray(teacher, dtype=np.float64))
    if t.ndim != 2 or t.shape[1] != 2 or len(t) != len(hard):
        return hard
    p_teacher = np.clip(t[:, 1], eps, 1.0 - eps)
    conf = np.clip(2.0 * np.abs(p_teacher - 0.5), 0.0, 1.0)
    agree = ((p_teacher >= 0.5).astype(int) == y_np).astype(np.float64)
    alpha = conf * (0.35 + 0.65 * agree)
    alpha = np.clip(alpha, 0.0, float(os.environ.get("PPPOST_DUAL_TEACHER_MAX_WEIGHT", "0.75")))
    base_true = float(np.clip(default_true_weight, 0.0, 1.0))
    mixed = (1.0 - alpha) * (base_true * hard + (1.0 - base_true) * p_teacher) + alpha * p_teacher
    return np.clip(mixed, eps, 1.0 - eps)


def _tune_dual_residual_gate(
    y_val: np.ndarray,
    base_val: np.ndarray,
    clinical_residual_val: np.ndarray,
    clinical_conf_val: np.ndarray,
    n_classes: int,
) -> Tuple[float, float, Dict[str, float]]:
    base_stats = binary_operating_stats(y_val, base_val, n_classes)
    conf = np.asarray(clinical_conf_val, dtype=np.float64).reshape(-1)
    if conf.size != len(y_val):
        conf = np.ones(len(y_val), dtype=np.float64)
    thr_grid = _threshold_grid_from_scores(conf)
    scale_grid = np.linspace(0.0, float(os.environ.get("PPPOST_DUAL_CLINICAL_MAX_SCALE", "0.90")), 10)
    best = {"score": -np.inf, "scale": 0.0, "thr": -np.inf, "stats": base_stats}
    for scale in scale_grid:
        for thr in thr_grid:
            gate = np.where(conf >= float(thr), conf, 0.0)
            cand = _combine_ebm_residual_logit(base_val, clinical_residual_val, scale=scale, gate=gate)
            st = binary_operating_stats(y_val, cand, n_classes)
            if st["log_loss"] > base_stats["log_loss"] + float(os.environ.get("PPPOST_DUAL_LOGLOSS_TOL", "0.055")):
                continue
            if st["specificity"] < base_stats["specificity"] - float(os.environ.get("PPPOST_DUAL_SPEC_TOL", "0.060")):
                continue
            score = st["mcc"] + 0.16 * st["balanced_accuracy"] + 0.05 * st["sensitivity"] - 0.06 * st["log_loss"]
            if score > best["score"] + 1e-10:
                best = {"score": float(score), "scale": float(scale), "thr": float(thr), "stats": st}
    st = best["stats"]
    return float(best["scale"]), float(best["thr"]), {
        "dual_gate_val_score": float(best["score"]),
        "dual_gate_val_mcc": float(st.get("mcc", float("nan"))),
        "dual_gate_val_balanced_accuracy": float(st.get("balanced_accuracy", float("nan"))),
        "dual_gate_val_sensitivity": float(st.get("sensitivity", float("nan"))),
        "dual_gate_val_specificity": float(st.get("specificity", float("nan"))),
        "dual_gate_val_log_loss": float(st.get("log_loss", float("nan"))),
    }

def _tune_residual_gate(
    y_val: np.ndarray,
    ebm_val: np.ndarray,
    residual_val: np.ndarray,
    confidence_val: np.ndarray,
    n_classes: int,
) -> Tuple[float, float, Dict[str, float]]:
    base_stats = binary_operating_stats(y_val, ebm_val, n_classes)
    conf = np.asarray(confidence_val, dtype=np.float64).reshape(-1)
    if conf.size != len(y_val):
        conf = np.ones(len(y_val), dtype=np.float64)
    thr_grid = _threshold_grid_from_scores(conf)
    scale_grid = np.linspace(0.0, float(os.environ.get("PPPOST_EBM_RESIDUAL_MAX_SCALE", "0.85")), 10)
    best = {"score": -np.inf, "scale": 0.0, "thr": -np.inf, "stats": base_stats}
    for scale in scale_grid:
        for thr in thr_grid:
            gate = np.where(conf >= float(thr), conf, 0.0)
            cand = _combine_ebm_residual_logit(ebm_val, residual_val, scale=scale, gate=gate)
            st = binary_operating_stats(y_val, cand, n_classes)
            if st["log_loss"] > base_stats["log_loss"] + float(os.environ.get("PPPOST_EBM_RESIDUAL_LOGLOSS_TOL", "0.030")):
                continue
            if st["accuracy"] < base_stats["accuracy"] - float(os.environ.get("PPPOST_EBM_RESIDUAL_ACC_TOL", "0.018")):
                continue
            score = st["mcc"] + 0.08 * st["balanced_accuracy"] - 0.04 * st["log_loss"]
            if score > best["score"] + 1e-10:
                best = {"score": float(score), "scale": float(scale), "thr": float(thr), "stats": st}
    st = best["stats"]
    return float(best["scale"]), float(best["thr"]), {
        "residual_gate_val_score": float(best["score"]),
        "residual_gate_val_mcc": float(st.get("mcc", float("nan"))),
        "residual_gate_val_balanced_accuracy": float(st.get("balanced_accuracy", float("nan"))),
        "residual_gate_val_log_loss": float(st.get("log_loss", float("nan"))),
        "residual_gate_val_brier": float(st.get("brier_score", float("nan"))),
    }


def _combine_ebm_residual_logit(ebm_proba: np.ndarray, residual_logit: np.ndarray, scale: float = 1.0, gate: Optional[np.ndarray] = None) -> np.ndarray:
    base_logit = _binary_logit_from_proba(ebm_proba)
    if np.ndim(base_logit) != 1:
        return normalize_proba(ebm_proba)
    resid = np.asarray(residual_logit, dtype=np.float64).reshape(-1)
    if resid.size != base_logit.size:
        resid = np.resize(resid, base_logit.size)
    clip = float(os.environ.get("PPPOST_EBM_RESIDUAL_CLIP", "1.75"))
    resid = np.clip(resid, -clip, clip)
    if gate is None:
        gate_vec = np.ones_like(resid)
    else:
        gate_vec = np.asarray(gate, dtype=np.float64).reshape(-1)
        if gate_vec.size != resid.size:
            gate_vec = np.resize(gate_vec, resid.size)
        gate_vec = np.clip(gate_vec, 0.0, 1.0)
    return _binary_proba_from_logit(base_logit + float(scale) * gate_vec * resid)

def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X, dtype=np.float64).mean(axis=0)
    scale = np.asarray(X, dtype=np.float64).std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def learn_contextual_evidence_params(
    z_train: np.ndarray,
    X_train: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 180,
    lr: float = 0.01,
    rank: int = 4,
    context_weight: float = 0.10,
    balanced_weight: float = 0.20,
    brier_weight: float = 0.08,
    soft_mcc_weight: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Learn evidence-layer parameters plus a small low-rank context correction.

    The correction is deliberately low-capacity. It lets class support vary with
    patient context while keeping the primary prediction path in explicit rule
    evidence. The returned matrices are auditable as a separate context adapter.
    """
    z_np = np.asarray(z_train, dtype=np.float64)
    X_np = np.asarray(X_train, dtype=np.float64)
    mean, scale = _standardize_fit(X_np)
    Xs = np.clip((X_np - mean) / scale, -8.0, 8.0)
    n_br = z_np.shape[1]
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    n_classes = int(theta_np.shape[1])
    prior_np = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior_np = np.clip(prior_np / np.maximum(prior_np.sum(), 1e-12), 1e-9, 1.0)
    y_np = np.asarray(y_train).ravel().astype(int)

    z_t = torch.tensor(z_np, dtype=torch.float32)
    x_t = torch.tensor(Xs, dtype=torch.float32)
    contrib_t = torch.tensor(np.log(np.clip(theta_np, 1e-9, 1.0)) - np.log(prior_np), dtype=torch.float32)
    prior_log_t = torch.tensor(np.log(prior_np), dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    onehot_t = torch.nn.functional.one_hot(y_t, num_classes=n_classes).float()
    counts = np.bincount(np.clip(y_np, 0, n_classes - 1), minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.maximum(weights.mean(), 1e-12)
    class_weight_t = torch.tensor(weights, dtype=torch.float32)

    init = float(np.log(np.exp(1.0) - 1.0))
    log_branch_alpha = torch.nn.Parameter(torch.full((n_br,), init, dtype=torch.float32))
    log_class_rel = torch.nn.Parameter(torch.full((n_br, n_classes), init, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))
    temp_raw = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    r = max(1, min(int(rank), Xs.shape[1], 8))
    ctx_w = torch.nn.Parameter(0.01 * torch.randn(Xs.shape[1], r, dtype=torch.float32))
    ctx_v = torch.nn.Parameter(0.01 * torch.randn(r, n_classes, dtype=torch.float32))
    opt = torch.optim.Adam([log_branch_alpha, log_class_rel, bias, temp_raw, ctx_w, ctx_v], lr=lr)

    for _ in range(max(1, int(epochs))):
        branch_alpha = torch.nn.functional.softplus(log_branch_alpha)
        class_rel = torch.nn.functional.softplus(log_class_rel)
        z_eff = z_t * branch_alpha.unsqueeze(0)
        active = z_eff.sum(dim=1, keepdim=True).clamp_min(1e-9)
        rule_logits = prior_log_t + (z_eff @ (contrib_t * class_rel)) / active + bias.unsqueeze(0)
        ctx_logits = (x_t @ ctx_w) @ ctx_v
        logits = rule_logits + float(context_weight) * ctx_logits
        temp = torch.nn.functional.softplus(temp_raw).clamp_min(1e-4)
        logp = torch.nn.functional.log_softmax(logits / temp, dim=1)
        p = torch.exp(logp)
        ce = torch.nn.functional.nll_loss(logp, y_t)
        ce_bal = torch.nn.functional.nll_loss(logp, y_t, weight=class_weight_t)
        brier = ((p - onehot_t) ** 2).sum(dim=1).mean()
        soft_mcc_loss = torch.tensor(0.0, dtype=torch.float32)
        if n_classes == 2:
            y1 = onehot_t[:, 1]
            p1 = p[:, 1]
            tp = (p1 * y1).sum(); tn = ((1.0 - p1) * (1.0 - y1)).sum()
            fp = (p1 * (1.0 - y1)).sum(); fn = ((1.0 - p1) * y1).sum()
            denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            soft_mcc_loss = 1.0 - (tp * tn - fp * fn) / denom.clamp_min(1e-8)
        reg = 5e-5 * (branch_alpha.mean() + class_rel.mean())
        reg = reg + 1e-3 * ((class_rel - 1.0) ** 2).mean()
        reg = reg + 1e-4 * ((ctx_w ** 2).mean() + (ctx_v ** 2).mean())
        loss = ce + float(balanced_weight) * ce_bal + float(brier_weight) * brier + float(soft_mcc_weight) * soft_mcc_loss + reg
        opt.zero_grad(); loss.backward(); opt.step()

    return (
        torch.nn.functional.softplus(log_branch_alpha).detach().cpu().numpy(),
        torch.nn.functional.softplus(log_class_rel).detach().cpu().numpy(),
        bias.detach().cpu().numpy(),
        float(torch.nn.functional.softplus(temp_raw).detach().cpu().item()),
        mean,
        scale,
        ctx_w.detach().cpu().numpy(),
        ctx_v.detach().cpu().numpy(),
    )


def aggregate_contextual_evidence(
    z: np.ndarray,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_alpha: np.ndarray,
    class_reliability: np.ndarray,
    class_bias: np.ndarray,
    temperature: float,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    ctx_w: np.ndarray,
    ctx_v: np.ndarray,
    context_weight: float = 0.10,
    top_k: int = 0,
) -> np.ndarray:
    rule = aggregate_evidence_layer_v2(
        z, theta, class_prior, branch_alpha, class_reliability,
        class_bias, temperature=1.0, top_k=top_k,
    )
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    logits = np.log(np.clip(rule, 1e-9, 1.0))
    Xs = np.clip((np.asarray(X, dtype=np.float64) - x_mean) / x_scale, -8.0, 8.0)
    logits = logits + float(context_weight) * ((Xs @ ctx_w) @ ctx_v)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def predict_contextual_evidence_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    branch_alpha: np.ndarray,
    class_reliability: np.ndarray,
    class_bias: np.ndarray,
    temperature: float,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    ctx_w: np.ndarray,
    ctx_v: np.ndarray,
    cfg: RunConfig,
    top_k: int = 0,
    use_model_reliability: bool = True,
) -> np.ndarray:
    diff_post = make_diff_posterior_for_model(model, cfg, use_model_reliability=use_model_reliability)
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X[sl]).float()).detach().cpu().numpy()
        chunks.append(aggregate_contextual_evidence(
            z, X[sl], theta, class_prior, branch_alpha, class_reliability,
            class_bias, temperature, x_mean, x_scale, ctx_w, ctx_v,
            context_weight=0.10, top_k=top_k,
        ))
    return normalize_proba(np.vstack(chunks))


def fit_ebm_anchor_proba(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, n_classes: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    baseline = make_tabular_baseline(
        "ebm",
        max_bins=int(os.environ.get("PPPOST_EBM_ANCHOR_MAX_BINS", "128")),
        max_interaction_bins=int(os.environ.get("PPPOST_EBM_ANCHOR_MAX_INTERACTION_BINS", "16")),
        interactions=int(os.environ.get("PPPOST_EBM_ANCHOR_INTERACTIONS", "4")),
        outer_bags=int(os.environ.get("PPPOST_EBM_ANCHOR_OUTER_BAGS", "2")),
    )
    baseline.fit(X_fit, y_fit, n_classes=n_classes, seed=seed)
    return (
        _ensure_2d_proba(baseline.predict_proba(X_fit), n_classes),
        _ensure_2d_proba(baseline.predict_proba(X_val), n_classes),
        _ensure_2d_proba(baseline.predict_proba(X_test), n_classes),
        float(baseline.fit_seconds),
    )

def learn_evidence_decomp_params(
    z_train: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
    l1: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_br = z_train.shape[1]
    z_t = torch.tensor(z_train, dtype=torch.float32)
    theta_np = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior_np = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior_np = np.clip(prior_np / np.maximum(prior_np.sum(), 1e-12), 1e-9, 1.0)
    contrib_np = np.log(np.clip(theta_np, 1e-9, 1.0)) - np.log(prior_np)
    pos_t = torch.tensor(np.maximum(contrib_np, 0.0), dtype=torch.float32)
    neg_t = torch.tensor(np.maximum(-contrib_np, 0.0), dtype=torch.float32)
    prior_log_t = torch.tensor(np.log(prior_np), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y_train).ravel(), dtype=torch.long)
    log_ap = torch.nn.Parameter(torch.zeros(n_br))
    log_an = torch.nn.Parameter(torch.zeros(n_br))
    bias = torch.nn.Parameter(torch.zeros(theta_np.shape[1]))
    temp_raw = torch.nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1.0), dtype=torch.float32))
    opt = torch.optim.Adam([log_ap, log_an, bias, temp_raw], lr=lr)
    for _ in range(max(1, int(epochs))):
        alpha_pos = torch.nn.functional.softplus(log_ap)
        alpha_neg = torch.nn.functional.softplus(log_an)
        z_pos = z_t * alpha_pos.unsqueeze(0)
        z_neg = z_t * alpha_neg.unsqueeze(0)
        active = (0.5 * (z_pos + z_neg)).sum(dim=1, keepdim=True).clamp_min(1e-9)
        logits = prior_log_t + ((z_pos @ pos_t) - (z_neg @ neg_t)) / active + bias.unsqueeze(0)
        temp = torch.nn.functional.softplus(temp_raw).clamp_min(1e-4)
        logp = torch.nn.functional.log_softmax(logits / temp, dim=1)
        loss = torch.nn.functional.nll_loss(logp, y_t)
        if l1 > 0:
            loss = loss + float(l1) * (alpha_pos.mean() + alpha_neg.mean()) * 0.5
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (
        torch.nn.functional.softplus(log_ap).detach().cpu().numpy(),
        torch.nn.functional.softplus(log_an).detach().cpu().numpy(),
        bias.detach().cpu().numpy(),
        float(torch.nn.functional.softplus(temp_raw).detach().cpu().item()),
    )


def _copy_evidence_reliability(diff_post: DifferentiablePosterior, reliability: Optional[np.ndarray]) -> None:
    if reliability is None or diff_post.evidence_reliability_logit is None:
        return
    r = np.asarray(reliability, dtype=np.float32).reshape(-1)
    if r.size != diff_post.n_branches:
        return
    r_scaled = np.clip(r / diff_post.reliability_max, 1e-4, 1.0 - 1e-4)
    with torch.no_grad():
        diff_post.evidence_reliability_logit.copy_(
            torch.log(torch.from_numpy(r_scaled) / (1.0 - torch.from_numpy(r_scaled)))
        )


def make_diff_posterior_for_model(
    model: RuleNetworkModel,
    cfg: RunConfig,
    use_model_reliability: bool = False,
) -> DifferentiablePosterior:
    reliability = getattr(model, "posterior_evidence_reliability_", None) if use_model_reliability else None
    diff_post = DifferentiablePosterior(
        model.branches,
        p_high=cfg.posterior_p_high,
        p_low=cfg.posterior_p_low,
        tau=cfg.condition_tau,
        learn_reliability=reliability is not None,
    )
    _copy_evidence_reliability(diff_post, reliability)
    return diff_post


def posterior_z_matrix(
    model: RuleNetworkModel,
    X: np.ndarray,
    cfg: RunConfig,
    use_model_reliability: bool = False,
) -> np.ndarray:
    diff_post = make_diff_posterior_for_model(model, cfg, use_model_reliability)
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            chunks.append(diff_post(bp_t, x_t).detach().cpu().numpy())
    return np.vstack(chunks)


def aggregate_evidence_decomp_logit(
    z: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    alpha_pos: np.ndarray,
    alpha_neg: np.ndarray,
    class_bias: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    top_k: int = 0,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    theta = normalize_proba(np.asarray(theta, dtype=np.float64))
    prior = np.asarray(class_prior, dtype=np.float64).reshape(1, -1)
    prior = np.clip(prior / np.maximum(prior.sum(), 1e-12), 1e-9, 1.0)
    alpha_pos = np.asarray(alpha_pos, dtype=np.float64).reshape(1, -1)
    alpha_neg = np.asarray(alpha_neg, dtype=np.float64).reshape(1, -1)
    z_pos = np.clip(z, 0.0, 1.0) * np.clip(alpha_pos, 0.0, 10.0)
    z_neg = np.clip(z, 0.0, 1.0) * np.clip(alpha_neg, 0.0, 10.0)
    top_k = int(top_k)
    if top_k > 0 and top_k < z.shape[1]:
        score = 0.5 * (z_pos + z_neg)
        idx = np.argpartition(score, -top_k, axis=1)[:, -top_k:]
        sparse_pos = np.zeros_like(z_pos)
        sparse_neg = np.zeros_like(z_neg)
        rows = np.arange(z.shape[0])[:, None]
        sparse_pos[rows, idx] = z_pos[rows, idx]
        sparse_neg[rows, idx] = z_neg[rows, idx]
        z_pos, z_neg = sparse_pos, sparse_neg
    contrib = np.log(np.clip(theta, 1e-9, 1.0)) - np.log(prior)
    pos = np.maximum(contrib, 0.0)
    neg = np.maximum(-contrib, 0.0)
    active_mass = np.maximum((0.5 * (z_pos + z_neg)).sum(axis=1, keepdims=True), 1e-9)
    logits = np.log(prior) + ((z_pos @ pos) - (z_neg @ neg)) / active_mass
    if class_bias is not None:
        logits = logits + np.asarray(class_bias, dtype=np.float64).reshape(1, -1)
    temp = max(float(temperature), 1e-6)
    logits = logits / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def predict_evidence_decomp_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    alpha_pos: np.ndarray,
    alpha_neg: np.ndarray,
    class_bias: np.ndarray,
    temperature: float,
    cfg: RunConfig,
    top_k: int = 0,
    use_model_reliability: bool = False,
) -> np.ndarray:
    diff_post = make_diff_posterior_for_model(model, cfg, use_model_reliability)
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        chunks.append(aggregate_evidence_decomp_logit(
            z, theta, class_prior, alpha_pos, alpha_neg, class_bias, temperature, top_k=top_k,
        ))
    return normalize_proba(np.vstack(chunks))


def tune_binary_threshold_mcc(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    proba = normalize_proba(proba)
    if proba.shape[1] != 2:
        return 0.5, float("nan")
    y = np.asarray(y_true).ravel().astype(int)
    p1 = np.clip(proba[:, 1], 1e-9, 1.0 - 1e-9)
    grid = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 199),
        np.quantile(p1, np.linspace(0.02, 0.98, 97)),
    ]))
    best_thr, best_mcc, best_f1 = 0.5, -np.inf, -np.inf
    for thr in grid:
        pred = (p1 >= float(thr)).astype(int)
        mcc = float(matthews_corrcoef(y, pred))
        f1m = float(f1_score(y, pred, average="macro", zero_division=0))
        if (mcc > best_mcc + 1e-12) or (abs(mcc - best_mcc) <= 1e-12 and f1m > best_f1):
            best_thr, best_mcc, best_f1 = float(thr), mcc, f1m
    return best_thr, best_mcc


def apply_binary_threshold_shift(proba: np.ndarray, threshold: float) -> np.ndarray:
    proba = normalize_proba(proba)
    if proba.shape[1] != 2:
        return proba
    p1 = np.clip(proba[:, 1], 1e-9, 1.0 - 1e-9)
    thr = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    logit = np.log(p1 / (1.0 - p1)) - np.log(thr / (1.0 - thr))
    p1_adj = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    return np.column_stack([1.0 - p1_adj, p1_adj])


def teacher_proba_for_source(
    src,
    fitted,
    X: np.ndarray,
    n_classes: int,
    cfg: RunConfig,
    y_fallback: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    teacher = None
    if getattr(fitted, "extra", None):
        teacher = fitted.extra.get("tabpfn_teacher_model")
    try:
        if teacher is not None:
            return predict_sklearn_chunks(teacher, X, cfg.batch_size, n_classes)
        return normalize_proba(src.predict_proba_native(fitted.native_model, X, n_classes))
    except Exception as exc:
        print(f"    [warn] teacher proba failed ({exc}); using hard labels for KD")
        if y_fallback is None:
            return None
        y = np.asarray(y_fallback).ravel().astype(int)
        out = np.full((len(y), n_classes), 1e-3 / max(n_classes - 1, 1), dtype=np.float64)
        out[np.arange(len(y)), np.clip(y, 0, n_classes - 1)] = 1.0 - 1e-3
        return normalize_proba(out)

def predict_evidence_logit_aux_chunks(
    model: RuleNetworkModel,
    X: np.ndarray,
    theta: np.ndarray,
    class_prior: np.ndarray,
    alpha: np.ndarray,
    class_bias: np.ndarray,
    temperature: float,
    cfg: RunConfig,
    top_k: int = 0,
    use_model_reliability: bool = False,
) -> np.ndarray:
    diff_post = make_diff_posterior_for_model(
        model, cfg, use_model_reliability=use_model_reliability,
    )
    chunks = []
    for sl, bp in branch_probs_chunks(model, X, cfg.batch_size):
        with torch.no_grad():
            bp_t = torch.from_numpy(bp).float()
            x_t = torch.from_numpy(X[sl]).float()
            z = diff_post(bp_t, x_t).detach().cpu().numpy()
        chunks.append(aggregate_evidence_logit(
            z, theta, class_prior, alpha, class_bias, temperature, top_k=top_k,
        ))
    return normalize_proba(np.vstack(chunks))

def train_ppost_aux_model(
    branches_per_tree: Sequence[Sequence],
    n_features: int,
    n_classes: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: RunConfig,
    aux_branch_weight: float = 0.05,
    aux_soft_and: str = "geomean",
    learn_evidence: bool = False,
    evidence_reg_weight: float = 1e-3,
) -> Tuple[RuleNetworkModel, np.ndarray]:
    model_aux = RuleNetworkModel(task="classification")
    model_aux.build_model_from_branches(
        branches_per_tree,
        in_features=n_features,
        out_features=n_classes,
    )
    theta_init = build_theta_matrix(model_aux.branches, n_classes)
    model_aux, theta_aux = model_aux.fit_problog_posterior_e2e(
        X_train,
        y_train,
        X_test,
        y_test,
        theta_init,
        epochs=cfg.expensive_epochs,
        batch_size=cfg.train_batch_size,
        tau=cfg.condition_tau,
        aux_branch_weight=aux_branch_weight,
        aux_soft_and=aux_soft_and,
        learn_evidence=learn_evidence,
        evidence_reg_weight=evidence_reg_weight,
        p_high=cfg.posterior_p_high,
        p_low=cfg.posterior_p_low,
    )
    return model_aux, theta_aux

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
        arr = np.load(path, allow_pickle=True)
        if "X" not in arr or "y" not in arr:
            raise ValueError("NPZ dataset must contain arrays X and y")
        X = np.asarray(arr["X"], dtype=np.float32)
        y_raw = np.asarray(arr["y"])
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str)).astype(np.int64)
        if "feature_names" in arr.files:
            feature_names = [str(v) for v in arr["feature_names"].tolist()]
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        if "class_names" in arr.files:
            class_names = [str(c) for c in arr["class_names"].tolist()]
        else:
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



def _one_hot_labels(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    out = np.zeros((len(y), n_classes), dtype=float)
    valid = (y >= 0) & (y < n_classes)
    out[np.arange(len(y))[valid], y[valid]] = 1.0
    return out


def _expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    if len(y_true) == 0:
        return float("nan")
    proba = normalize_proba(proba)
    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def _binary_rates(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).astype(float)
    tn, fp, fn, tp = cm.ravel()
    def div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")
    return {
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
    }


def _binary_net_benefit(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float) -> float:
    threshold = float(threshold)
    if threshold <= 0.0 or threshold >= 1.0 or len(y_true) == 0:
        return float("nan")
    pred_pos = proba_pos >= threshold
    y_pos = np.asarray(y_true) == 1
    tp = float(np.logical_and(pred_pos, y_pos).sum())
    fp = float(np.logical_and(pred_pos, ~y_pos).sum())
    n = float(len(y_true))
    return float(tp / n - fp / n * (threshold / (1.0 - threshold)))


def _average_precision_ovr(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        scores = []
        weights = []
        for cls in range(n_classes):
            target = (y_true == cls).astype(int)
            support = int(target.sum())
            if support == 0 or support == len(target):
                continue
            scores.append(float(average_precision_score(target, proba[:, cls])))
            weights.append(support)
        if not scores:
            return float("nan")
        return float(np.average(scores, weights=np.asarray(weights, dtype=float)))
    except Exception:
        return float("nan")


def _brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    proba = normalize_proba(proba)
    if n_classes == 2:
        return float(np.mean((proba[:, 1] - (np.asarray(y_true) == 1).astype(float)) ** 2))
    onehot = _one_hot_labels(y_true, n_classes)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))

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
        "auprc_ovr": _average_precision_ovr(y_true, proba, n_classes),
        "brier_score": _brier_score(y_true, proba, n_classes),
        "ece_10": _expected_calibration_error(y_true, proba, n_bins=10),
        "ece_20": _expected_calibration_error(y_true, proba, n_bins=20),
    }
    if n_classes == 2:
        out.update(_binary_rates(y_true, pred))
        out["net_benefit_0_10"] = _binary_net_benefit(y_true, proba[:, 1], 0.10)
        out["net_benefit_0_20"] = _binary_net_benefit(y_true, proba[:, 1], 0.20)
    else:
        out.update({
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "net_benefit_0_10": float("nan"),
            "net_benefit_0_20": float("nan"),
        })
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
    csv_row = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }

    write_header = not csv_path.exists()
    fieldnames = list(csv_row.keys())
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = list(reader.fieldnames or [])
            missing = [key for key in fieldnames if key not in old_fieldnames]
            if old_fieldnames and missing:
                existing_rows = list(reader)
                fieldnames = old_fieldnames + missing
                with csv_path.open("w", newline="", encoding="utf-8") as wf:
                    writer = csv.DictWriter(wf, fieldnames=fieldnames)
                    writer.writeheader()
                    for old_row in existing_rows:
                        writer.writerow(old_row)
            elif old_fieldnames:
                fieldnames = old_fieldnames

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    metrics = compute_metrics(y_true, proba, n_classes, no_roc_auc=cfg.no_roc_auc)
    proba = normalize_proba(proba)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, pred, labels=range(n_classes)).astype(int).tolist()
    prediction_artifact = None
    if cfg.save_predictions:
        pred_dir = csv_path.parent / "prediction_artifacts"
        pred_dir.mkdir(parents=True, exist_ok=True)
        safe_dataset = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(dataset))
        safe_source = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(rule_source))
        safe_variant = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(variant))
        artifact_path = pred_dir / f"{safe_dataset}__fold{fold}__{safe_source}__{safe_variant}__{len(rows):06d}.npz"
        np.savez_compressed(
            artifact_path,
            y_true=np.asarray(y_true, dtype=int),
            proba=np.asarray(proba, dtype=float),
            pred=np.asarray(pred, dtype=int),
            dataset=np.asarray(str(dataset)),
            fold=np.asarray(int(fold)),
            rule_source=np.asarray(str(rule_source)),
            variant=np.asarray(str(variant)),
        )
        prediction_artifact = str(artifact_path)
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
        "condition_tau": float(cfg.condition_tau),
        "posterior_p_high": float(cfg.posterior_p_high),
        "posterior_p_low": float(cfg.posterior_p_low),
        "theta_shrinkage_strength": float(cfg.theta_shrinkage_strength),
        "signed_logit_temperature": float(cfg.signed_logit_temperature),
        "sparse_logit_top_k": int(cfg.sparse_logit_top_k),
        "rule_budget": int(cfg.rule_budget),
        "rule_max_depth": int(cfg.rule_max_depth),
        "rule_min_support": float(cfg.rule_min_support),
        "rule_selection": str(cfg.rule_selection),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        **metrics,
        "confusion_matrix": cm,
    }
    if prediction_artifact is not None:
        row["prediction_artifact"] = prediction_artifact
    if extra_fields:
        row.update(extra_fields)
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

    branches_per_tree, resource_meta = select_rule_resource_branches(
        fitted.branches_per_tree, X_train, y_train, n_classes, cfg,
    )
    if resource_meta.get("enabled"):
        print(
            "    rule_resource "
            f"selected={resource_meta.get('selected')} "
            f"from={resource_meta.get('original')} "
            f"budget={resource_meta.get('budget', 0)} "
            f"max_depth={resource_meta.get('max_depth', 0)} "
            f"min_support={resource_meta.get('min_support', 0.0):.4f} "
            f"selection={resource_meta.get('selection', cfg.rule_selection)}"
        )

    t0 = time.time()
    model = RuleNetworkModel(task="classification")
    model.build_model_from_branches(
        branches_per_tree,
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
    class_prior = empirical_class_prior(y_train, n_classes)
    branch_support_train: Optional[np.ndarray] = None

    def get_branch_support_train() -> np.ndarray:
        nonlocal branch_support_train
        if branch_support_train is None:
            branch_support_train = estimate_branch_support(
                model.branches, X_train, cfg.batch_size, cfg.condition_tau,
            )
        return branch_support_train

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

    if "pp_theta_post_frozen" in cfg.variants:
        t0 = time.time()
        proba = predict_frozen_posterior_wmean_chunks(model, X_test, theta, cfg)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_frozen",
            y_test, proba, n_classes, fitted.fit_seconds, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_shrink_theta" in cfg.variants:
        t0 = time.time()
        theta_shrunk = shrink_theta_empirical_bayes(
            theta, class_prior, get_branch_support_train(), len(X_train),
            cfg.theta_shrinkage_strength,
        )
        proba = predict_diff_posterior_wmean_chunks(
            model, X_test, theta_shrunk, cfg.batch_size,
            tau=cfg.condition_tau,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_shrink_theta",
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_frozen_support_prior" in cfg.variants:
        t0 = time.time()
        proba = predict_frozen_support_prior_wmean_chunks(
            model, X_test, theta, get_branch_support_train(), cfg,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_frozen_support_prior",
            y_test, proba, n_classes, fitted.fit_seconds, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_signed_logit" in cfg.variants:
        t0 = time.time()
        proba = predict_diff_posterior_signed_logit_chunks(
            model, X_test, theta, class_prior, cfg, top_k=0,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_signed_logit",
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_sparse_logit" in cfg.variants:
        t0 = time.time()
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        proba = predict_diff_posterior_signed_logit_chunks(
            model, X_test, theta, class_prior, cfg, top_k=sparse_k,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_sparse_logit",
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_feature_reliability" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        reliability = estimate_feature_group_reliability(
            model.branches, X_rel, y_rel, n_classes, theta, cfg,
        )
        proba = predict_diff_posterior_signed_logit_chunks(
            model, X_test, theta, class_prior, cfg, reliability=reliability, top_k=0,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_feature_reliability",
            y_test, proba, n_classes, base_fit, time.time() - t0,
            n_branches, top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_source_calibrated" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_sc = RuleNetworkModel(task="classification")
        model_sc.build_model_from_branches(
            branches_per_tree,
            in_features=n_features,
            out_features=n_classes,
        )
        theta_init = build_theta_matrix(model_sc.branches, n_classes)
        model_sc, theta_sc = model_sc.fit_problog_posterior_e2e(
            X_rel, y_rel, X_test, y_test, theta_init,
            epochs=cfg.expensive_epochs,
            batch_size=cfg.train_batch_size,
            aux_branch_weight=0.05,
        )
        branch_support_sc = estimate_branch_support(
            model_sc.branches, X_train, cfg.batch_size, cfg.condition_tau,
        )
        branch_reliability, source_confidence = estimate_source_calibrated_reliability(
            model_sc.branches, X_rel, y_rel, n_classes, theta_sc, class_prior,
            branch_support_sc, len(X_train), cfg,
        )
        theta_cal = shrink_theta_empirical_bayes(
            theta_sc, class_prior, branch_support_sc, len(X_train), cfg.theta_shrinkage_strength,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_source_calibrated_ppost_chunks(
            model_sc, X_test, theta_cal, class_prior, branch_reliability,
            source_confidence, cfg,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_source_calibrated",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_sc.branches), top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_rule_utility_aux" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        utility_branches, utility_meta = select_posterior_utility_branches(
            branches_per_tree, X_rel, y_rel, n_classes, cfg,
        )
        if utility_meta.get("enabled"):
            print(
                "    rule_utility "
                f"selected={utility_meta.get('selected')} "
                f"from={utility_meta.get('original')} "
                f"budget={utility_meta.get('budget', 0)}"
            )
        model_ru, theta_ru = train_ppost_aux_model(
            utility_branches, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_ru, X_test, theta_ru, cfg.batch_size,
            tau=cfg.condition_tau,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_rule_utility_aux",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_ru.branches), min(cfg.top_k_max, max(cfg.top_k_min, round(len(model_ru.branches) * cfg.top_k_ratio))),
            cfg, rule_source=source_name,
        )

    if "pp_theta_post_constrained_evidence" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_ce, theta_ce = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=5e-3,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_ce, X_test, theta_ce, cfg.batch_size,
            tau=cfg.condition_tau,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_constrained_evidence",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_ce.branches), top_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_evidence_logit_aux" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_el, theta_el = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
        )
        diff_post_el = DifferentiablePosterior(
            model_el.branches,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
            tau=cfg.condition_tau,
        )
        z_rel_chunks = []
        for sl, bp in branch_probs_chunks(model_el, X_rel, cfg.batch_size):
            with torch.no_grad():
                bp_t = torch.from_numpy(bp).float()
                x_t = torch.from_numpy(X_rel[sl]).float()
                z_rel_chunks.append(diff_post_el(bp_t, x_t).detach().cpu().numpy())
        z_rel = np.vstack(z_rel_chunks)
        alpha_el, bias_el, temp_el = learn_evidence_logit_params(
            z_rel, theta_el, class_prior, y_rel,
            epochs=max(20, min(int(cfg.expensive_epochs), 200)),
            lr=0.01,
            l1=1e-4,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        proba = predict_evidence_logit_aux_chunks(
            model_el, X_test, theta_el, class_prior, alpha_el, bias_el, temp_el,
            cfg, top_k=sparse_k,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_evidence_logit_aux",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_el.branches), sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_evidence_layer_v2" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(
            X_rel, y_rel, n_classes, seed,
        )
        t0 = time.time()
        model_v2, theta_v2 = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        branch_support_v2 = estimate_branch_support(
            model_v2.branches, X_train, cfg.batch_size, cfg.condition_tau,
        )
        theta_v2 = shrink_theta_empirical_bayes(
            theta_v2, class_prior, branch_support_v2, len(X_train), cfg.theta_shrinkage_strength,
        )
        z_fit = posterior_z_matrix(model_v2, X_fit, cfg, use_model_reliability=True)
        alpha_v2, class_rel_v2, bias_v2, temp_v2 = learn_evidence_layer_v2_params(
            z_fit, theta_v2, class_prior, y_fit,
            epochs=max(30, min(int(cfg.expensive_epochs), 240)),
            lr=0.01,
            l1=5e-5,
            balanced_weight=0.35,
            brier_weight=0.05,
            soft_mcc_weight=0.10,
        )
        model_reliability = getattr(model_v2, "posterior_evidence_reliability_", None)
        if model_reliability is not None and len(model_reliability) == len(alpha_v2):
            alpha_v2 = alpha_v2 * np.clip(np.asarray(model_reliability, dtype=np.float64), 0.25, 2.0)
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        z_val = posterior_z_matrix(model_v2, X_val, cfg, use_model_reliability=True)
        proba_val = aggregate_evidence_layer_v2(
            z_val, theta_v2, class_prior, alpha_v2, class_rel_v2,
            bias_v2, temp_v2, top_k=sparse_k,
        )
        base_score = operating_point_score(y_val, proba_val, n_classes)
        threshold, threshold_score = tune_binary_threshold_operating_score(
            y_val, proba_val, n_classes,
        )
        selected_mode = "threshold" if threshold_score > base_score + 1e-4 else "calibrated"
        selected_threshold = threshold if selected_mode == "threshold" else 0.5
        print(
            "    evidence_layer_v2 "
            f"mode={selected_mode} threshold={selected_threshold:.4f} "
            f"val_score={max(base_score, threshold_score):.4f} "
            f"base_score={base_score:.4f} threshold_score={threshold_score:.4f}"
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_evidence_layer_v2_chunks(
            model_v2, X_test, theta_v2, class_prior, alpha_v2, class_rel_v2,
            bias_v2, temp_v2, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        if selected_mode == "threshold":
            proba = apply_binary_threshold_shift(proba, selected_threshold)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_evidence_layer_v2",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_v2.branches), sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_teacher_anchored" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(
            X_rel, y_rel, n_classes, seed,
        )
        t0 = time.time()
        model_ta, theta_ta = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=1e-3,
        )
        branch_support_ta = estimate_branch_support(
            model_ta.branches, X_train, cfg.batch_size, cfg.condition_tau,
        )
        theta_ta = shrink_theta_empirical_bayes(
            theta_ta, class_prior, branch_support_ta, len(X_train), cfg.theta_shrinkage_strength,
        )
        z_fit = posterior_z_matrix(model_ta, X_fit, cfg, use_model_reliability=True)
        alpha_ta, class_rel_ta, rule_bias_ta, rule_temp_ta = learn_evidence_layer_v2_params(
            z_fit, theta_ta, class_prior, y_fit,
            epochs=max(30, min(int(cfg.expensive_epochs), 220)),
            lr=0.01,
            l1=1e-4,
            balanced_weight=0.12,
            brier_weight=0.15,
            soft_mcc_weight=0.30,
        )
        alpha_ta = np.clip(alpha_ta, 0.0, 2.5)
        class_rel_ta = np.clip(class_rel_ta, 0.0, 2.0)
        model_reliability = getattr(model_ta, "posterior_evidence_reliability_", None)
        if model_reliability is not None and len(model_reliability) == len(alpha_ta):
            alpha_ta = alpha_ta * np.clip(np.asarray(model_reliability, dtype=np.float64), 0.50, 1.50)
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        rule_fit = aggregate_evidence_layer_v2(
            z_fit, theta_ta, class_prior, alpha_ta, class_rel_ta,
            rule_bias_ta, rule_temp_ta, top_k=sparse_k,
        )
        z_val = posterior_z_matrix(model_ta, X_val, cfg, use_model_reliability=True)
        rule_val = aggregate_evidence_layer_v2(
            z_val, theta_ta, class_prior, alpha_ta, class_rel_ta,
            rule_bias_ta, rule_temp_ta, top_k=sparse_k,
        )
        teacher_fit = teacher_proba_for_source(src, fitted, X_fit, n_classes, cfg)
        teacher_val = teacher_proba_for_source(src, fitted, X_val, n_classes, cfg)
        if teacher_fit is None or np.asarray(teacher_fit).shape != rule_fit.shape:
            teacher_fit = rule_fit
        if teacher_val is None or np.asarray(teacher_val).shape != rule_val.shape:
            teacher_val = rule_val
        beta_teacher, beta_rule, anchor_bias, anchor_temp = learn_teacher_anchor_params(
            teacher_fit, rule_fit, class_prior, y_fit,
            epochs=max(30, min(int(cfg.expensive_epochs), 220)),
            lr=0.01,
            balanced_weight=0.10,
            brier_weight=0.15,
            soft_mcc_weight=0.25,
            distill_weight=0.20,
            distill_temperature=2.0,
            l2=1e-3,
        )
        proba_val = aggregate_teacher_anchored_proba(
            teacher_val, rule_val, class_prior, beta_teacher, beta_rule,
            anchor_bias, anchor_temp,
        )
        selected_threshold, selected_mode, anchor_diag = conservative_teacher_anchor_threshold(
            y_val, proba_val, n_classes,
        )
        print(
            "    teacher_anchor "
            f"mode={selected_mode} threshold={selected_threshold:.4f} "
            f"beta_teacher={beta_teacher:.3f} beta_rule={beta_rule:.3f} "
            f"val_mcc={anchor_diag.get('teacher_anchor_val_mcc_threshold', float('nan')):.4f} "
            f"val_log_loss={anchor_diag.get('teacher_anchor_val_log_loss_threshold', float('nan')):.4f}"
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        teacher_test = teacher_proba_for_source(src, fitted, X_test, n_classes, cfg)
        if teacher_test is None:
            teacher_test = predict_evidence_layer_v2_chunks(
                model_ta, X_test, theta_ta, class_prior, alpha_ta, class_rel_ta,
                rule_bias_ta, rule_temp_ta, cfg, top_k=sparse_k, use_model_reliability=True,
            )
        proba = predict_teacher_anchored_chunks(
            model_ta, X_test, theta_ta, class_prior, alpha_ta, class_rel_ta,
            rule_bias_ta, rule_temp_ta, teacher_test, beta_teacher, beta_rule,
            anchor_bias, anchor_temp, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        if selected_mode == "threshold":
            proba = apply_binary_threshold_shift(proba, selected_threshold)
        extra_fields = {
            **anchor_diag,
            "teacher_anchor_beta_teacher": float(beta_teacher),
            "teacher_anchor_beta_rule": float(beta_rule),
            "teacher_anchor_temperature": float(anchor_temp),
            "teacher_anchor_rule_temperature": float(rule_temp_ta),
        }
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_teacher_anchored",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_ta.branches), sparse_k, cfg, rule_source=source_name,
            extra_fields=extra_fields,
        )


    if "pp_theta_post_selective_evidence" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_sel, theta_sel = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        branch_support_sel = estimate_branch_support(model_sel.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_sel = shrink_theta_empirical_bayes(theta_sel, class_prior, branch_support_sel, len(X_train), cfg.theta_shrinkage_strength)
        z_fit = posterior_z_matrix(model_sel, X_fit, cfg, use_model_reliability=True)
        alpha_sel, class_rel_sel, bias_sel, temp_sel = learn_evidence_layer_v2_params(
            z_fit, theta_sel, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 260)),
            lr=0.01, l1=8e-5, balanced_weight=0.45, brier_weight=0.08, soft_mcc_weight=0.25,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else min(64, max(16, top_k))
        z_val = posterior_z_matrix(model_sel, X_val, cfg, use_model_reliability=True)
        proba_val = aggregate_evidence_layer_v2(z_val, theta_sel, class_prior, alpha_sel, class_rel_sel, bias_sel, temp_sel, top_k=sparse_k)
        threshold, threshold_score = tune_binary_threshold_operating_score(y_val, proba_val, n_classes)
        base_score = operating_point_score(y_val, proba_val, n_classes)
        selected_mode = "threshold" if threshold_score > base_score + 1e-4 else "calibrated"
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_evidence_layer_v2_chunks(
            model_sel, X_test, theta_sel, class_prior, alpha_sel, class_rel_sel,
            bias_sel, temp_sel, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        if selected_mode == "threshold":
            proba = apply_binary_threshold_shift(proba, threshold)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_selective_evidence",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_sel.branches), sparse_k, cfg, rule_source=source_name,
            extra_fields={"selective_mode": selected_mode, "selective_threshold": float(threshold), "selective_val_score": float(max(base_score, threshold_score))},
        )

    if "pp_theta_post_rule_family" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_fam, theta_fam_raw = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_fam = estimate_branch_support(model_fam.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_fam_raw = shrink_theta_empirical_bayes(theta_fam_raw, class_prior, support_fam, len(X_train), cfg.theta_shrinkage_strength)
        groups = _branch_family_groups(model_fam.branches, theta_fam_raw)
        theta_fam = _reduce_family_theta(theta_fam_raw, groups, weights=support_fam)
        z_fit = _reduce_family_z(posterior_z_matrix(model_fam, X_fit, cfg, use_model_reliability=True), groups)
        alpha_fam, class_rel_fam, bias_fam, temp_fam = learn_evidence_layer_v2_params(
            z_fit, theta_fam, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 240)),
            lr=0.01, l1=5e-5, balanced_weight=0.35, brier_weight=0.08, soft_mcc_weight=0.18,
        )
        z_val = _reduce_family_z(posterior_z_matrix(model_fam, X_val, cfg, use_model_reliability=True), groups)
        proba_val = aggregate_evidence_layer_v2(z_val, theta_fam, class_prior, alpha_fam, class_rel_fam, bias_fam, temp_fam, top_k=0)
        iso = _fit_binary_isotonic(y_val, proba_val, n_classes)
        fit_secs = time.time() - t0
        t0 = time.time()
        diff_post = make_diff_posterior_for_model(model_fam, cfg, use_model_reliability=True)
        chunks = []
        for sl, bp in branch_probs_chunks(model_fam, X_test, cfg.batch_size):
            with torch.no_grad():
                z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X_test[sl]).float()).detach().cpu().numpy()
            z_family = _reduce_family_z(z, groups)
            chunks.append(aggregate_evidence_layer_v2(z_family, theta_fam, class_prior, alpha_fam, class_rel_fam, bias_fam, temp_fam, top_k=0))
        proba = _apply_binary_isotonic(iso, normalize_proba(np.vstack(chunks)))
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_rule_family",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(groups), min(len(groups), top_k), cfg, rule_source=source_name,
            extra_fields={"rule_family_count": int(len(groups)), "rule_family_original_branches": int(len(model_fam.branches))},
        )

    if "pp_theta_post_contextual_support" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_ctx, theta_ctx = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_ctx = estimate_branch_support(model_ctx.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_ctx = shrink_theta_empirical_bayes(theta_ctx, class_prior, support_ctx, len(X_train), cfg.theta_shrinkage_strength)
        z_fit = posterior_z_matrix(model_ctx, X_fit, cfg, use_model_reliability=True)
        alpha_ctx, class_rel_ctx, bias_ctx, temp_ctx, x_mean_ctx, x_scale_ctx, ctx_w, ctx_v = learn_contextual_evidence_params(
            z_fit, X_fit, theta_ctx, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 220)),
            lr=0.01, rank=4, context_weight=0.10, balanced_weight=0.20, brier_weight=0.08, soft_mcc_weight=0.12,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        z_val = posterior_z_matrix(model_ctx, X_val, cfg, use_model_reliability=True)
        proba_val = aggregate_contextual_evidence(z_val, X_val, theta_ctx, class_prior, alpha_ctx, class_rel_ctx, bias_ctx, temp_ctx, x_mean_ctx, x_scale_ctx, ctx_w, ctx_v, top_k=sparse_k)
        iso = _fit_binary_isotonic(y_val, proba_val, n_classes)
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_contextual_evidence_chunks(
            model_ctx, X_test, theta_ctx, class_prior, alpha_ctx, class_rel_ctx,
            bias_ctx, temp_ctx, x_mean_ctx, x_scale_ctx, ctx_w, ctx_v, cfg,
            top_k=sparse_k, use_model_reliability=True,
        )
        proba = _apply_binary_isotonic(iso, proba)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_contextual_support",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_ctx.branches), sparse_k, cfg, rule_source=source_name,
            extra_fields={"context_rank": int(ctx_w.shape[1]), "context_weight": 0.10},
        )

    if "pp_theta_post_teacher_calibrated" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_tc, theta_tc = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=1e-3,
        )
        support_tc = estimate_branch_support(model_tc.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_tc = shrink_theta_empirical_bayes(theta_tc, class_prior, support_tc, len(X_train), cfg.theta_shrinkage_strength)
        z_fit = posterior_z_matrix(model_tc, X_fit, cfg, use_model_reliability=True)
        alpha_tc, class_rel_tc, rule_bias_tc, rule_temp_tc = learn_evidence_layer_v2_params(
            z_fit, theta_tc, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 220)), lr=0.01,
            l1=1e-4, balanced_weight=0.08, brier_weight=0.35, soft_mcc_weight=0.12,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        rule_fit = aggregate_evidence_layer_v2(z_fit, theta_tc, class_prior, alpha_tc, class_rel_tc, rule_bias_tc, rule_temp_tc, top_k=sparse_k)
        z_val = posterior_z_matrix(model_tc, X_val, cfg, use_model_reliability=True)
        rule_val = aggregate_evidence_layer_v2(z_val, theta_tc, class_prior, alpha_tc, class_rel_tc, rule_bias_tc, rule_temp_tc, top_k=sparse_k)
        teacher_fit = teacher_proba_for_source(src, fitted, X_fit, n_classes, cfg)
        teacher_val = teacher_proba_for_source(src, fitted, X_val, n_classes, cfg)
        if teacher_fit is None or np.asarray(teacher_fit).shape != rule_fit.shape:
            teacher_fit = rule_fit
        if teacher_val is None or np.asarray(teacher_val).shape != rule_val.shape:
            teacher_val = rule_val
        beta_teacher, beta_rule, anchor_bias, anchor_temp = learn_teacher_anchor_params(
            teacher_fit, rule_fit, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 240)), lr=0.01,
            balanced_weight=0.05, brier_weight=0.45, soft_mcc_weight=0.10,
            distill_weight=0.10, distill_temperature=2.0, l2=2e-3,
        )
        proba_val = aggregate_teacher_anchored_proba(teacher_val, rule_val, class_prior, beta_teacher, beta_rule, anchor_bias, anchor_temp)
        iso = _fit_binary_isotonic(y_val, proba_val, n_classes)
        fit_secs = time.time() - t0
        t0 = time.time()
        teacher_test = teacher_proba_for_source(src, fitted, X_test, n_classes, cfg)
        if teacher_test is None:
            teacher_test = rule_val[: len(X_test)] if len(rule_val) == len(X_test) else None
        if teacher_test is None:
            teacher_test = predict_evidence_layer_v2_chunks(model_tc, X_test, theta_tc, class_prior, alpha_tc, class_rel_tc, rule_bias_tc, rule_temp_tc, cfg, top_k=sparse_k, use_model_reliability=True)
        proba = predict_teacher_anchored_chunks(
            model_tc, X_test, theta_tc, class_prior, alpha_tc, class_rel_tc, rule_bias_tc, rule_temp_tc,
            teacher_test, beta_teacher, beta_rule, anchor_bias, anchor_temp, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        proba = _apply_binary_isotonic(iso, proba)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_teacher_calibrated",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_tc.branches), sparse_k, cfg, rule_source=source_name,
            extra_fields={"teacher_cal_beta_teacher": float(beta_teacher), "teacher_cal_beta_rule": float(beta_rule), "teacher_cal_temperature": float(anchor_temp), "teacher_cal_isotonic": int(iso is not None)},
        )


    if any(v in cfg.variants for v in (
        "pp_theta_post_ebm_correction_calibrated",
        "pp_theta_post_ebm_correction_mcc",
        "pp_theta_post_ebm_correction_sensitivity",
    )):
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_ec, theta_ec = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=1e-3,
        )
        support_ec = estimate_branch_support(model_ec.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_ec = shrink_theta_empirical_bayes(theta_ec, class_prior, support_ec, len(X_train), cfg.theta_shrinkage_strength)
        z_fit = posterior_z_matrix(model_ec, X_fit, cfg, use_model_reliability=True)
        alpha_ec, class_rel_ec, rule_bias_ec, rule_temp_ec = learn_evidence_layer_v2_params(
            z_fit, theta_ec, class_prior, y_fit,
            epochs=max(50, min(int(cfg.expensive_epochs), 260)), lr=0.01,
            l1=8e-5, balanced_weight=0.18, brier_weight=0.28, soft_mcc_weight=0.22,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else min(64, max(16, top_k))
        rule_fit = aggregate_evidence_layer_v2(z_fit, theta_ec, class_prior, alpha_ec, class_rel_ec, rule_bias_ec, rule_temp_ec, top_k=sparse_k)
        z_val = posterior_z_matrix(model_ec, X_val, cfg, use_model_reliability=True)
        rule_val = aggregate_evidence_layer_v2(z_val, theta_ec, class_prior, alpha_ec, class_rel_ec, rule_bias_ec, rule_temp_ec, top_k=sparse_k)
        ebm_fit, ebm_val, ebm_test, _ = fit_ebm_anchor_proba(X_fit, y_fit, X_val, X_test, n_classes, seed)
        beta_ebm, beta_rule, corr_bias, corr_temp = learn_teacher_anchor_params(
            ebm_fit, rule_fit, class_prior, y_fit,
            epochs=max(50, min(int(cfg.expensive_epochs), 260)), lr=0.01,
            balanced_weight=0.10, brier_weight=0.35, soft_mcc_weight=0.22,
            distill_weight=0.05, distill_temperature=2.0, l2=2e-3,
        )
        val_raw = aggregate_teacher_anchored_proba(ebm_val, rule_val, class_prior, beta_ebm, beta_rule, corr_bias, corr_temp)
        iso = _fit_binary_isotonic(y_val, val_raw, n_classes)
        val_cal = _apply_binary_isotonic(iso, val_raw)
        mcc_thr, mcc_val = tune_binary_threshold_mcc(y_val, val_cal)
        sens_thr, sens_stats = tune_binary_threshold_sensitivity_floor(y_val, val_cal, n_classes, specificity_floor=float(os.environ.get("PPPOST_SENS_SPEC_FLOOR", "0.92")))
        fit_secs = time.time() - t0
        t0 = time.time()
        proba_raw = predict_teacher_anchored_chunks(
            model_ec, X_test, theta_ec, class_prior, alpha_ec, class_rel_ec, rule_bias_ec, rule_temp_ec,
            ebm_test, beta_ebm, beta_rule, corr_bias, corr_temp, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        proba_cal = _apply_binary_isotonic(iso, proba_raw)
        pred_secs = time.time() - t0
        common_extra = {
            "ebm_correction_beta_ebm": float(beta_ebm),
            "ebm_correction_beta_rule": float(beta_rule),
            "ebm_correction_temperature": float(corr_temp),
            "ebm_correction_isotonic": int(iso is not None),
        }
        if "pp_theta_post_ebm_correction_calibrated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_correction_calibrated",
                y_test, proba_cal, n_classes, fit_secs, pred_secs,
                len(model_ec.branches), sparse_k, cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "calibrated-risk", "operating_threshold": 0.5},
            )
        if "pp_theta_post_ebm_correction_mcc" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_correction_mcc",
                y_test, apply_binary_threshold_shift(proba_cal, mcc_thr), n_classes, fit_secs, pred_secs,
                len(model_ec.branches), sparse_k, cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "mcc", "operating_threshold": float(mcc_thr), "operating_val_mcc": float(mcc_val)},
            )
        if "pp_theta_post_ebm_correction_sensitivity" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_correction_sensitivity",
                y_test, apply_binary_threshold_shift(proba_cal, sens_thr), n_classes, fit_secs, pred_secs,
                len(model_ec.branches), sparse_k, cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "sensitivity", "operating_threshold": float(sens_thr), **{f"operating_val_{k}": v for k, v in sens_stats.items()}},
            )

    if any(v in cfg.variants for v in ("pp_theta_post_rule_family_calibrated", "pp_theta_post_rule_family_sensitivity")):
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_rfs, theta_rfs_raw = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.03, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_rfs = estimate_branch_support(model_rfs.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_rfs_raw = shrink_theta_empirical_bayes(theta_rfs_raw, class_prior, support_rfs, len(X_train), cfg.theta_shrinkage_strength)
        groups = _branch_family_groups(model_rfs.branches, theta_rfs_raw)
        theta_rfs = _reduce_family_theta(theta_rfs_raw, groups, weights=support_rfs)
        z_fit = _reduce_family_z(posterior_z_matrix(model_rfs, X_fit, cfg, use_model_reliability=True), groups)
        alpha_rfs, class_rel_rfs, bias_rfs, temp_rfs = learn_evidence_layer_v2_params(
            z_fit, theta_rfs, class_prior, y_fit,
            epochs=max(50, min(int(cfg.expensive_epochs), 260)), lr=0.01,
            l1=4e-5, balanced_weight=0.45, brier_weight=0.10, soft_mcc_weight=0.30,
        )
        rule_family_top_k = max(1, int(os.environ.get("PPPOST_RULE_FAMILY_TOPK", "32")))
        z_val = _reduce_family_z(posterior_z_matrix(model_rfs, X_val, cfg, use_model_reliability=True), groups)
        val_raw = aggregate_evidence_layer_v2(z_val, theta_rfs, class_prior, alpha_rfs, class_rel_rfs, bias_rfs, temp_rfs, top_k=min(rule_family_top_k, len(groups)))
        iso = _fit_binary_isotonic(y_val, val_raw, n_classes)
        val_cal = _apply_binary_isotonic(iso, val_raw)
        sens_thr, sens_stats = tune_binary_threshold_sensitivity_floor(y_val, val_cal, n_classes, specificity_floor=float(os.environ.get("PPPOST_SENS_SPEC_FLOOR", "0.92")))
        fit_secs = time.time() - t0
        t0 = time.time()
        diff_post = make_diff_posterior_for_model(model_rfs, cfg, use_model_reliability=True)
        chunks = []
        for sl, bp in branch_probs_chunks(model_rfs, X_test, cfg.batch_size):
            with torch.no_grad():
                z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X_test[sl]).float()).detach().cpu().numpy()
            z_family = _reduce_family_z(z, groups)
            chunks.append(aggregate_evidence_layer_v2(z_family, theta_rfs, class_prior, alpha_rfs, class_rel_rfs, bias_rfs, temp_rfs, top_k=min(rule_family_top_k, len(groups))))
        proba_cal = _apply_binary_isotonic(iso, normalize_proba(np.vstack(chunks)))
        pred_secs = time.time() - t0
        extra = {"rule_family_count": int(len(groups)), "rule_family_original_branches": int(len(model_rfs.branches)), "rule_family_isotonic": int(iso is not None)}
        if "pp_theta_post_rule_family_calibrated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_rule_family_calibrated",
                y_test, proba_cal, n_classes, fit_secs, pred_secs,
                len(groups), min(rule_family_top_k, len(groups)), cfg, rule_source=source_name,
                extra_fields={**extra, "operating_mode": "calibrated-risk", "operating_threshold": 0.5},
            )
        if "pp_theta_post_rule_family_sensitivity" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_rule_family_sensitivity",
                y_test, apply_binary_threshold_shift(proba_cal, sens_thr), n_classes, fit_secs, pred_secs,
                len(groups), min(rule_family_top_k, len(groups)), cfg, rule_source=source_name,
                extra_fields={**extra, "operating_mode": "sensitivity", "operating_threshold": float(sens_thr), **{f"operating_val_{k}": v for k, v in sens_stats.items()}},
            )


    new_family_variants = {
        "pp_theta_post_ebm_bounded_residual_gate",
        "pp_theta_post_agreement_gated",
        "pp_theta_post_tabpfn_ebm_family_calibrated",
        "pp_theta_post_family_utility_pruned_topk",
        "pp_theta_post_operating_calibrated",
        "pp_theta_post_operating_mcc",
        "pp_theta_post_operating_sens90",
        "pp_theta_post_operating_sens92",
        "pp_theta_post_operating_sens95",
        "pp_theta_post_monotone_ebm_families",
    }
    if any(v in cfg.variants for v in new_family_variants):
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_nf, theta_nf_raw = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.03, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_nf = estimate_branch_support(model_nf.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_nf_raw = shrink_theta_empirical_bayes(theta_nf_raw, class_prior, support_nf, len(X_train), cfg.theta_shrinkage_strength)
        groups_all = _branch_family_groups(model_nf.branches, theta_nf_raw)
        theta_all = _reduce_family_theta(theta_nf_raw, groups_all, weights=support_nf)
        z_fit_all = _reduce_family_z(posterior_z_matrix(model_nf, X_fit, cfg, use_model_reliability=True), groups_all)
        z_val_all = _reduce_family_z(posterior_z_matrix(model_nf, X_val, cfg, use_model_reliability=True), groups_all)

        def _learn_family(z_fit_in: np.ndarray, theta_in: np.ndarray):
            return learn_evidence_layer_v2_params(
                z_fit_in, theta_in, class_prior, y_fit,
                epochs=max(50, min(int(cfg.expensive_epochs), 260)), lr=0.01,
                l1=4e-5, balanced_weight=0.42, brier_weight=0.12, soft_mcc_weight=0.28,
            )

        alpha_all, class_rel_all, bias_all, temp_all = _learn_family(z_fit_all, theta_all)
        top_all = min(max(1, int(os.environ.get("PPPOST_RULE_FAMILY_TOPK", "32"))), max(1, len(groups_all)))
        val_raw = aggregate_evidence_layer_v2(z_val_all, theta_all, class_prior, alpha_all, class_rel_all, bias_all, temp_all, top_k=top_all)
        iso_all = _fit_binary_isotonic(y_val, val_raw, n_classes)
        val_cal = _apply_binary_isotonic(iso_all, val_raw)
        mcc_thr, mcc_val = tune_binary_threshold_mcc(y_val, val_cal)
        sens_thresholds: Dict[float, Tuple[float, Dict[str, float]]] = {}
        for floor in (0.90, 0.92, 0.95):
            sens_thresholds[floor] = tune_binary_threshold_sensitivity_floor(y_val, val_cal, n_classes, specificity_floor=floor)
        needs_ebm_gate = any(v in cfg.variants for v in ("pp_theta_post_ebm_bounded_residual_gate", "pp_theta_post_agreement_gated"))
        ebm_val = ebm_test = None
        bounded_lam = bounded_thr = agree_lam = agree_thr = 0.0
        bounded_diag: Dict[str, float] = {}
        agree_diag: Dict[str, float] = {}
        if needs_ebm_gate:
            ebm_fit, ebm_val, ebm_test, _ = fit_ebm_anchor_proba(X_fit, y_fit, X_val, X_test, n_classes, seed)
            del ebm_fit
            bounded_conf = _evidence_concentration(z_val_all)
            bounded_lam, bounded_thr, bounded_diag = _pick_residual_gate(y_val, ebm_val, val_cal, n_classes, bounded_conf, mode="bounded")
            agree_conf = _evidence_concentration(z_val_all) * (1.0 - _family_entropy(z_val_all))
            agree_lam, agree_thr, agree_diag = _pick_residual_gate(y_val, ebm_val, val_cal, n_classes, agree_conf, mode="agreement")

        # Validation-only utility pruning: keep compact family evidence that changes
        # class separation, instead of averaging many correlated branches.
        keep_idx = np.arange(len(groups_all), dtype=int)
        utility_top_k = min(int(os.environ.get("PPPOST_FAMILY_UTILITY_TOPK", "32")), len(groups_all))
        if utility_top_k > 0 and len(groups_all) > utility_top_k and n_classes == 2:
            pos = z_fit_all[np.asarray(y_fit) == 1]
            neg = z_fit_all[np.asarray(y_fit) == 0]
            if len(pos) and len(neg):
                separation = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
            else:
                separation = np.std(z_fit_all, axis=0)
            support_family = np.array([float(np.sum(support_nf[g])) for g in groups_all], dtype=np.float64)
            support_family = support_family / max(float(np.max(support_family)), 1.0)
            theta_signal = np.abs(theta_all[:, 1] - class_prior[1]) if theta_all.shape[1] == 2 else np.max(np.abs(theta_all - class_prior.reshape(1, -1)), axis=1)
            utility = separation * (0.5 + support_family) * (0.5 + theta_signal)
            keep_idx = np.argsort(-utility)[:utility_top_k]
            keep_idx = np.sort(keep_idx)
        z_fit_pruned = z_fit_all[:, keep_idx]
        z_val_pruned = z_val_all[:, keep_idx]
        theta_pruned = theta_all[keep_idx]
        alpha_pr, class_rel_pr, bias_pr, temp_pr = _learn_family(z_fit_pruned, theta_pruned)
        top_pr = min(utility_top_k if utility_top_k > 0 else len(keep_idx), len(keep_idx))
        val_pr = aggregate_evidence_layer_v2(z_val_pruned, theta_pruned, class_prior, alpha_pr, class_rel_pr, bias_pr, temp_pr, top_k=top_pr)
        iso_pr = _fit_binary_isotonic(y_val, val_pr, n_classes)

        theta_mono = theta_all.copy()
        mono_boosted = 0
        alpha_mo = class_rel_mo = bias_mo = temp_mo = iso_mo = None
        if "pp_theta_post_monotone_ebm_families" in cfg.variants:
            if n_classes == 2 and len(groups_all):
                X_fit_np = np.asarray(X_fit, dtype=np.float64)
                y_center = np.asarray(y_fit, dtype=np.float64) - float(np.mean(y_fit))
                x_center = X_fit_np - np.nanmean(X_fit_np, axis=0, keepdims=True)
                denom = np.maximum(np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center)), 1e-8)
                corr = np.nan_to_num((x_center.T @ y_center) / denom, nan=0.0, posinf=0.0, neginf=0.0)
                for gi, g in enumerate(groups_all):
                    feats = sorted({
                        int(cond.feature_idx)
                        for b in g
                        for cond in getattr(model_nf.branches[int(b)], "conditions", [])
                        if 0 <= int(cond.feature_idx) < len(corr)
                    })
                    if not feats:
                        continue
                    clinical_dir = float(np.mean(corr[feats]))
                    evidence_dir = float(theta_mono[gi, 1] - class_prior[1])
                    if abs(clinical_dir) < 1e-5 or abs(evidence_dir) < 1e-5:
                        continue
                    if clinical_dir * evidence_dir > 0:
                        theta_mono[gi, 1] = class_prior[1] + 1.20 * (theta_mono[gi, 1] - class_prior[1])
                        mono_boosted += 1
                    else:
                        theta_mono[gi, 1] = class_prior[1] + 0.75 * (theta_mono[gi, 1] - class_prior[1])
                    theta_mono[gi, 0] = 1.0 - theta_mono[gi, 1]
                theta_mono = normalize_proba(theta_mono)
            alpha_mo, class_rel_mo, bias_mo, temp_mo = _learn_family(z_fit_all, theta_mono)
            val_mo = aggregate_evidence_layer_v2(z_val_all, theta_mono, class_prior, alpha_mo, class_rel_mo, bias_mo, temp_mo, top_k=top_all)
            iso_mo = _fit_binary_isotonic(y_val, val_mo, n_classes)
        fit_secs = time.time() - t0

        t0 = time.time()
        diff_post = make_diff_posterior_for_model(model_nf, cfg, use_model_reliability=True)
        chunks_all = []
        chunks_pruned = []
        chunks_mono = []
        z_test_conf = []
        z_test_agree = []
        for sl, bp in branch_probs_chunks(model_nf, X_test, cfg.batch_size):
            with torch.no_grad():
                z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X_test[sl]).float()).detach().cpu().numpy()
            z_family = _reduce_family_z(z, groups_all)
            chunks_all.append(aggregate_evidence_layer_v2(z_family, theta_all, class_prior, alpha_all, class_rel_all, bias_all, temp_all, top_k=top_all))
            chunks_pruned.append(aggregate_evidence_layer_v2(z_family[:, keep_idx], theta_pruned, class_prior, alpha_pr, class_rel_pr, bias_pr, temp_pr, top_k=top_pr))
            if "pp_theta_post_monotone_ebm_families" in cfg.variants:
                chunks_mono.append(aggregate_evidence_layer_v2(z_family, theta_mono, class_prior, alpha_mo, class_rel_mo, bias_mo, temp_mo, top_k=top_all))
            z_test_conf.append(_evidence_concentration(z_family))
            z_test_agree.append(_evidence_concentration(z_family) * (1.0 - _family_entropy(z_family)))
        test_cal = _apply_binary_isotonic(iso_all, normalize_proba(np.vstack(chunks_all)))
        test_pruned = _apply_binary_isotonic(iso_pr, normalize_proba(np.vstack(chunks_pruned)))
        test_mono = _apply_binary_isotonic(iso_mo, normalize_proba(np.vstack(chunks_mono))) if chunks_mono else test_cal
        test_conf = np.concatenate(z_test_conf) if z_test_conf else np.zeros(len(X_test), dtype=np.float64)
        test_agree = np.concatenate(z_test_agree) if z_test_agree else np.zeros(len(X_test), dtype=np.float64)
        bounded = _combine_binary_residual(ebm_test, test_cal, bounded_lam, gate=(test_conf >= bounded_thr), delta_clip=1.5) if ebm_test is not None else test_cal
        agreed = _combine_binary_residual(ebm_test, test_cal, agree_lam, gate=(test_agree >= agree_thr), delta_clip=2.5) if ebm_test is not None else test_cal
        pred_secs = time.time() - t0

        common_extra = {
            "rule_family_count": int(len(groups_all)),
            "rule_family_original_branches": int(len(model_nf.branches)),
            "rule_family_top_k": int(top_all),
            "rule_family_isotonic": int(iso_all is not None),
        }
        if "pp_theta_post_ebm_bounded_residual_gate" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_bounded_residual_gate",
                y_test, bounded, n_classes, fit_secs, pred_secs,
                len(groups_all), top_all, cfg, rule_source=source_name,
                extra_fields={**common_extra, "residual_gate_lambda": float(bounded_lam), "residual_gate_threshold": float(bounded_thr), **bounded_diag},
            )
        if "pp_theta_post_agreement_gated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_agreement_gated",
                y_test, agreed, n_classes, fit_secs, pred_secs,
                len(groups_all), top_all, cfg, rule_source=source_name,
                extra_fields={**common_extra, "agreement_gate_lambda": float(agree_lam), "agreement_gate_threshold": float(agree_thr), **agree_diag},
            )
        if "pp_theta_post_tabpfn_ebm_family_calibrated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_tabpfn_ebm_family_calibrated",
                y_test, test_cal, n_classes, fit_secs, pred_secs,
                len(groups_all), top_all, cfg, rule_source=source_name,
                extra_fields={**common_extra, "distill_student": int(source_name == "tabpfn_distill_ebm_terms")},
            )
        if "pp_theta_post_family_utility_pruned_topk" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_family_utility_pruned_topk",
                y_test, test_pruned, n_classes, fit_secs, pred_secs,
                len(keep_idx), top_pr, cfg, rule_source=source_name,
                extra_fields={**common_extra, "utility_pruned_families": int(len(keep_idx)), "utility_pruned_original_families": int(len(groups_all)), "utility_pruned_isotonic": int(iso_pr is not None)},
            )
        if "pp_theta_post_operating_calibrated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_operating_calibrated",
                y_test, test_cal, n_classes, fit_secs, pred_secs,
                len(groups_all), top_all, cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "calibrated-risk", "operating_threshold": 0.5},
            )
        if "pp_theta_post_operating_mcc" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_operating_mcc",
                y_test, apply_binary_threshold_shift(test_cal, mcc_thr), n_classes, fit_secs, pred_secs,
                len(groups_all), top_all, cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "mcc", "operating_threshold": float(mcc_thr), "operating_val_mcc": float(mcc_val)},
            )
        for floor, variant_id in ((0.90, "pp_theta_post_operating_sens90"), (0.92, "pp_theta_post_operating_sens92"), (0.95, "pp_theta_post_operating_sens95")):
            if variant_id in cfg.variants:
                thr, stats = sens_thresholds[floor]
                evaluate_and_stream(
                    rows, csv_path, jsonl_path, ds.name, fold, variant_id,
                    y_test, apply_binary_threshold_shift(test_cal, thr), n_classes, fit_secs, pred_secs,
                    len(groups_all), top_all, cfg, rule_source=source_name,
                    extra_fields={**common_extra, "operating_mode": f"sensitivity@spec{floor:.2f}", "operating_threshold": float(thr), **{f"operating_val_{k}": v for k, v in stats.items()}},
                )
        if "pp_theta_post_monotone_ebm_families" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_monotone_ebm_families",
                y_test, test_mono, n_classes, fit_secs, pred_secs,
                len(groups_all), top_all, cfg, rule_source=source_name,
                extra_fields={**common_extra, "monotone_prior_boosted_families": int(mono_boosted), "monotone_prior_isotonic": int(iso_mo is not None)},
            )




    dual_residual_variants = {
        "pp_theta_post_dual_residual_calibrated",
        "pp_theta_post_dual_residual_mcc",
        "pp_theta_post_dual_residual_sens92",
        "pp_theta_post_dual_residual_sens95_cal",
    }
    if any(v in cfg.variants for v in dual_residual_variants):
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_dr, theta_dr_raw = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.03, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_dr = estimate_branch_support(model_dr.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_dr_raw = shrink_theta_empirical_bayes(theta_dr_raw, class_prior, support_dr, len(X_train), cfg.theta_shrinkage_strength)
        groups_dr = _branch_family_groups(model_dr.branches, theta_dr_raw)
        theta_dr = _reduce_family_theta(theta_dr_raw, groups_dr, weights=support_dr)
        support_family_dr = np.array([float(np.sum(support_dr[g])) for g in groups_dr], dtype=np.float64)
        z_fit_dr = _reduce_family_z(posterior_z_matrix(model_dr, X_fit, cfg, use_model_reliability=True), groups_dr)
        z_val_dr = _reduce_family_z(posterior_z_matrix(model_dr, X_val, cfg, use_model_reliability=True), groups_dr)
        ebm_fit, ebm_val, ebm_test, ebm_fit_secs = fit_ebm_anchor_proba(X_fit, y_fit, X_val, X_test, n_classes, seed)

        teacher_model = None
        if getattr(fitted, "extra", None):
            teacher_model = fitted.extra.get("tabpfn_teacher_model")
        teacher_fit = teacher_val = None
        if teacher_model is not None:
            try:
                teacher_fit = predict_sklearn_chunks(teacher_model, X_fit, cfg.batch_size, n_classes)
                teacher_val = predict_sklearn_chunks(teacher_model, X_val, cfg.batch_size, n_classes)
            except Exception as exc:
                print(f"    [warn] dual residual teacher failed ({exc}); using true-label residual targets")
                teacher_fit = teacher_val = None
        use_teacher_conf = bool(int(os.environ.get("PPPOST_DUAL_TEACHER_CONF", "0")))
        risk_true_weight = float(os.environ.get("PPPOST_DUAL_RISK_TRUE_WEIGHT", "0.60"))
        clinical_true_weight = float(os.environ.get("PPPOST_DUAL_CLINICAL_TRUE_WEIGHT", "0.85"))
        if use_teacher_conf:
            p_risk_fit = _teacher_confidence_blend_target(y_fit, teacher_fit, risk_true_weight)
            p_risk_val = _teacher_confidence_blend_target(y_val, teacher_val, risk_true_weight)
            p_clin_fit = _teacher_confidence_blend_target(y_fit, teacher_fit, clinical_true_weight)
            p_clin_val = _teacher_confidence_blend_target(y_val, teacher_val, clinical_true_weight)
        else:
            p_risk_fit = _soft_binary_target(y_fit, teacher_fit, true_weight=risk_true_weight)
            p_risk_val = _soft_binary_target(y_val, teacher_val, true_weight=risk_true_weight)
            p_clin_fit = _soft_binary_target(y_fit, teacher_fit, true_weight=clinical_true_weight)
            p_clin_val = _soft_binary_target(y_val, teacher_val, true_weight=clinical_true_weight)
        residual_risk_fit = np.clip(
            _binary_logit_from_proba(np.column_stack([1.0 - p_risk_fit, p_risk_fit])) - _binary_logit_from_proba(ebm_fit),
            -float(os.environ.get("PPPOST_DUAL_RISK_TARGET_CLIP", "2.0")),
            float(os.environ.get("PPPOST_DUAL_RISK_TARGET_CLIP", "2.0")),
        )
        residual_risk_val = np.clip(
            _binary_logit_from_proba(np.column_stack([1.0 - p_risk_val, p_risk_val])) - _binary_logit_from_proba(ebm_val),
            -float(os.environ.get("PPPOST_DUAL_RISK_TARGET_CLIP", "2.0")),
            float(os.environ.get("PPPOST_DUAL_RISK_TARGET_CLIP", "2.0")),
        )
        residual_clin_fit = np.clip(
            _binary_logit_from_proba(np.column_stack([1.0 - p_clin_fit, p_clin_fit])) - _binary_logit_from_proba(ebm_fit),
            -float(os.environ.get("PPPOST_DUAL_CLINICAL_TARGET_CLIP", "3.0")),
            float(os.environ.get("PPPOST_DUAL_CLINICAL_TARGET_CLIP", "3.0")),
        )
        residual_clin_val = np.clip(
            _binary_logit_from_proba(np.column_stack([1.0 - p_clin_val, p_clin_val])) - _binary_logit_from_proba(ebm_val),
            -float(os.environ.get("PPPOST_DUAL_CLINICAL_TARGET_CLIP", "3.0")),
            float(os.environ.get("PPPOST_DUAL_CLINICAL_TARGET_CLIP", "3.0")),
        )

        top_dr = min(int(os.environ.get("PPPOST_DUAL_RESIDUAL_TOPK", "32")), len(groups_dr))
        keep_risk = _fit_residual_family_selection(z_val_dr, theta_dr, class_prior, residual_risk_val, support_family_dr, top_k=max(1, top_dr))
        if bool(int(os.environ.get("PPPOST_DUAL_CLINICAL_UTILITY", "1"))):
            keep_clin = _fit_clinical_residual_family_selection(z_val_dr, theta_dr, class_prior, ebm_val, residual_clin_val, y_val, support_family_dr, top_k=max(1, top_dr))
        else:
            keep_clin = _fit_residual_family_selection(z_val_dr, theta_dr, class_prior, residual_clin_val, support_family_dr, top_k=max(1, top_dr))
        if keep_risk.size == 0:
            keep_risk = np.arange(min(max(1, top_dr), len(groups_dr)), dtype=int)
        if keep_clin.size == 0:
            keep_clin = keep_risk.copy()
        feats_risk_fit = _family_llr_feature_matrix(z_fit_dr[:, keep_risk], theta_dr[keep_risk], class_prior, top_k=0)
        feats_risk_val = _family_llr_feature_matrix(z_val_dr[:, keep_risk], theta_dr[keep_risk], class_prior, top_k=0)
        feats_clin_fit = _family_llr_feature_matrix(z_fit_dr[:, keep_clin], theta_dr[keep_clin], class_prior, top_k=0)
        feats_clin_val = _family_llr_feature_matrix(z_val_dr[:, keep_clin], theta_dr[keep_clin], class_prior, top_k=0)
        ebm_unc_fit = _ebm_uncertainty_score(ebm_fit)
        ebm_unc_val = _ebm_uncertainty_score(ebm_val)
        y_fit_np = np.asarray(y_fit).ravel().astype(int)
        p_ebm_fit = normalize_proba(ebm_fit)[:, 1]
        ambiguous_fit = ((p_ebm_fit >= 0.15) & (p_ebm_fit <= 0.65)).astype(np.float64)
        risk_sw = 0.5 + ebm_unc_fit
        clinical_sw = 1.0 + float(os.environ.get("PPPOST_DUAL_POS_WEIGHT", "2.5")) * (y_fit_np == 1).astype(np.float64) + float(os.environ.get("PPPOST_DUAL_UNCERT_WEIGHT", "1.0")) * ebm_unc_fit + 0.75 * ambiguous_fit
        risk_bias, risk_coef = _fit_ridge_residual(feats_risk_fit, residual_risk_fit, l2=float(os.environ.get("PPPOST_DUAL_RISK_L2", "4.0")), sample_weight=risk_sw)
        clin_bias, clin_coef = _fit_ridge_residual(feats_clin_fit, residual_clin_fit, l2=float(os.environ.get("PPPOST_DUAL_CLINICAL_L2", "1.5")), sample_weight=clinical_sw)
        risk_val_pred = risk_bias + feats_risk_val @ risk_coef
        clin_val_pred = clin_bias + feats_clin_val @ clin_coef
        risk_conf_val = np.sqrt(np.clip(ebm_unc_val * _evidence_concentration(z_val_dr[:, keep_risk]), 0.0, 1.0))
        clin_conf_val = np.sqrt(np.clip(ebm_unc_val * _evidence_concentration(z_val_dr[:, keep_clin]), 0.0, 1.0))
        risk_scale, risk_thr, risk_diag = _tune_residual_gate(y_val, ebm_val, risk_val_pred, risk_conf_val, n_classes)
        val_risk = _combine_ebm_residual_logit(ebm_val, risk_val_pred, scale=risk_scale, gate=np.where(risk_conf_val >= risk_thr, risk_conf_val, 0.0))
        clin_scale, clin_thr, clin_diag = _tune_dual_residual_gate(y_val, val_risk, clin_val_pred, clin_conf_val, n_classes)
        val_dual_raw = _combine_ebm_residual_logit(val_risk, clin_val_pred, scale=clin_scale, gate=np.where(clin_conf_val >= clin_thr, clin_conf_val, 0.0))
        iso_dual = _fit_binary_isotonic(y_val, val_dual_raw, n_classes)
        val_dual_cal = _apply_binary_isotonic(iso_dual, val_dual_raw)
        mcc_thr_dual, mcc_val_dual = tune_binary_threshold_mcc(y_val, val_dual_cal)
        sens92_thr_dual, sens92_stats_dual = tune_binary_threshold_sensitivity_floor(y_val, val_dual_cal, n_classes, specificity_floor=0.92)
        sens95_thr_dual, sens95_stats_dual = tune_binary_threshold_sensitivity_floor(y_val, val_dual_cal, n_classes, specificity_floor=0.95)
        val_s95_shift = apply_binary_threshold_shift(val_dual_cal, sens95_thr_dual)
        iso_s95 = _fit_binary_isotonic(y_val, val_s95_shift, n_classes)
        fit_secs = time.time() - t0

        t0 = time.time()
        diff_post = make_diff_posterior_for_model(model_dr, cfg, use_model_reliability=True)
        risk_chunks = []
        clin_chunks = []
        risk_conf_chunks = []
        clin_conf_chunks = []
        for sl, bp in branch_probs_chunks(model_dr, X_test, cfg.batch_size):
            with torch.no_grad():
                z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X_test[sl]).float()).detach().cpu().numpy()
            z_family = _reduce_family_z(z, groups_dr)
            fr = _family_llr_feature_matrix(z_family[:, keep_risk], theta_dr[keep_risk], class_prior, top_k=0)
            fc = _family_llr_feature_matrix(z_family[:, keep_clin], theta_dr[keep_clin], class_prior, top_k=0)
            risk_chunks.append(risk_bias + fr @ risk_coef)
            clin_chunks.append(clin_bias + fc @ clin_coef)
            risk_conf_chunks.append(_evidence_concentration(z_family[:, keep_risk]))
            clin_conf_chunks.append(_evidence_concentration(z_family[:, keep_clin]))
        risk_test_pred = np.concatenate(risk_chunks) if risk_chunks else np.zeros(len(X_test), dtype=np.float64)
        clin_test_pred = np.concatenate(clin_chunks) if clin_chunks else np.zeros(len(X_test), dtype=np.float64)
        ebm_unc_test = _ebm_uncertainty_score(ebm_test)
        risk_conf_test = np.sqrt(np.clip(ebm_unc_test * (np.concatenate(risk_conf_chunks) if risk_conf_chunks else 0.0), 0.0, 1.0))
        clin_conf_test = np.sqrt(np.clip(ebm_unc_test * (np.concatenate(clin_conf_chunks) if clin_conf_chunks else 0.0), 0.0, 1.0))
        test_risk = _combine_ebm_residual_logit(ebm_test, risk_test_pred, scale=risk_scale, gate=np.where(risk_conf_test >= risk_thr, risk_conf_test, 0.0))
        test_dual_raw = _combine_ebm_residual_logit(test_risk, clin_test_pred, scale=clin_scale, gate=np.where(clin_conf_test >= clin_thr, clin_conf_test, 0.0))
        test_dual_cal = _apply_binary_isotonic(iso_dual, test_dual_raw)
        test_s95_cal = _apply_binary_isotonic(iso_s95, apply_binary_threshold_shift(test_dual_cal, sens95_thr_dual))
        pred_secs = time.time() - t0
        common_extra = {
            "dual_family_count": int(len(groups_dr)),
            "dual_risk_families": int(len(keep_risk)),
            "dual_clinical_families": int(len(keep_clin)),
            "dual_original_branches": int(len(model_dr.branches)),
            "dual_has_tabpfn_teacher": int(teacher_model is not None),
            "dual_teacher_confidence": int(use_teacher_conf),
            "dual_risk_true_weight": float(risk_true_weight),
            "dual_clinical_true_weight": float(clinical_true_weight),
            "dual_clinical_utility": int(bool(int(os.environ.get("PPPOST_DUAL_CLINICAL_UTILITY", "1")))),
            "dual_risk_scale": float(risk_scale),
            "dual_risk_threshold": float(risk_thr),
            "dual_clinical_scale": float(clin_scale),
            "dual_clinical_threshold": float(clin_thr),
            "dual_risk_coef_l1": float(np.sum(np.abs(risk_coef))),
            "dual_clinical_coef_l1": float(np.sum(np.abs(clin_coef))),
            "dual_isotonic": int(iso_dual is not None),
            "dual_sens95_cal_isotonic": int(iso_s95 is not None),
            "ebm_anchor_fit_seconds": float(ebm_fit_secs),
            **{f"risk_{k}": v for k, v in risk_diag.items()},
            **{f"clinical_{k}": v for k, v in clin_diag.items()},
        }
        if "pp_theta_post_dual_residual_calibrated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_dual_residual_calibrated",
                y_test, test_dual_cal, n_classes, fit_secs, pred_secs,
                len(keep_risk) + len(keep_clin), len(keep_risk) + len(keep_clin), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "calibrated-risk", "operating_threshold": 0.5},
            )
        if "pp_theta_post_dual_residual_mcc" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_dual_residual_mcc",
                y_test, apply_binary_threshold_shift(test_dual_cal, mcc_thr_dual), n_classes, fit_secs, pred_secs,
                len(keep_risk) + len(keep_clin), len(keep_risk) + len(keep_clin), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "mcc", "operating_threshold": float(mcc_thr_dual), "operating_val_mcc": float(mcc_val_dual)},
            )
        if "pp_theta_post_dual_residual_sens92" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_dual_residual_sens92",
                y_test, apply_binary_threshold_shift(test_dual_cal, sens92_thr_dual), n_classes, fit_secs, pred_secs,
                len(keep_risk) + len(keep_clin), len(keep_risk) + len(keep_clin), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "sensitivity@spec0.92", "operating_threshold": float(sens92_thr_dual), **{f"operating_val_{k}": v for k, v in sens92_stats_dual.items()}},
            )
        if "pp_theta_post_dual_residual_sens95_cal" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_dual_residual_sens95_cal",
                y_test, test_s95_cal, n_classes, fit_secs, pred_secs,
                len(keep_risk) + len(keep_clin), len(keep_risk) + len(keep_clin), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "sensitivity@spec0.95-calibrated", "operating_threshold": float(sens95_thr_dual), **{f"operating_val_{k}": v for k, v in sens95_stats_dual.items()}},
            )

    ebm_residual_variants = {
        "pp_theta_post_ebm_residual_calibrated",
        "pp_theta_post_ebm_residual_mcc",
        "pp_theta_post_ebm_residual_sens92",
        "pp_theta_post_ebm_residual_sens95",
    }
    if any(v in cfg.variants for v in ebm_residual_variants):
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_er, theta_er_raw = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.03, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_er = estimate_branch_support(model_er.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_er_raw = shrink_theta_empirical_bayes(theta_er_raw, class_prior, support_er, len(X_train), cfg.theta_shrinkage_strength)
        groups_er = _branch_family_groups(model_er.branches, theta_er_raw)
        theta_er = _reduce_family_theta(theta_er_raw, groups_er, weights=support_er)
        support_family_er = np.array([float(np.sum(support_er[g])) for g in groups_er], dtype=np.float64)
        z_fit_er = _reduce_family_z(posterior_z_matrix(model_er, X_fit, cfg, use_model_reliability=True), groups_er)
        z_val_er = _reduce_family_z(posterior_z_matrix(model_er, X_val, cfg, use_model_reliability=True), groups_er)
        ebm_fit, ebm_val, ebm_test, ebm_fit_secs = fit_ebm_anchor_proba(X_fit, y_fit, X_val, X_test, n_classes, seed)

        teacher_model = None
        if getattr(fitted, "extra", None):
            teacher_model = fitted.extra.get("tabpfn_teacher_model")
        teacher_fit = teacher_val = None
        if teacher_model is not None:
            try:
                teacher_fit = predict_sklearn_chunks(teacher_model, X_fit, cfg.batch_size, n_classes)
                teacher_val = predict_sklearn_chunks(teacher_model, X_val, cfg.batch_size, n_classes)
            except Exception as exc:
                print(f"    [warn] tabpfn residual teacher failed ({exc}); using true-label residual target")
                teacher_fit = teacher_val = None
        true_weight = float(os.environ.get("PPPOST_EBM_RESIDUAL_TRUE_WEIGHT", "0.45"))
        residual_fit = _residual_target_from_proba(y_fit, teacher_fit, ebm_fit, true_weight=true_weight)
        residual_val_target = _residual_target_from_proba(y_val, teacher_val, ebm_val, true_weight=true_weight)
        utility_top = min(int(os.environ.get("PPPOST_EBM_RESIDUAL_TOPK", "32")), len(groups_er))
        keep_er = _fit_residual_family_selection(z_val_er, theta_er, class_prior, residual_val_target, support_family_er, top_k=max(1, utility_top))
        if keep_er.size == 0:
            keep_er = np.arange(min(max(1, utility_top), len(groups_er)), dtype=int)
        feats_fit = _family_llr_feature_matrix(z_fit_er[:, keep_er], theta_er[keep_er], class_prior, top_k=0)
        feats_val = _family_llr_feature_matrix(z_val_er[:, keep_er], theta_er[keep_er], class_prior, top_k=0)
        sw = 0.5 + _ebm_uncertainty_score(ebm_fit)
        bias_er, coef_er = _fit_ridge_residual(
            feats_fit, residual_fit,
            l2=float(os.environ.get("PPPOST_EBM_RESIDUAL_RIDGE_L2", "2.0")),
            sample_weight=sw,
        )
        residual_val_pred = bias_er + feats_val @ coef_er
        conf_val = np.sqrt(np.clip(_ebm_uncertainty_score(ebm_val) * _evidence_concentration(z_val_er[:, keep_er]), 0.0, 1.0))
        gate_scale, gate_thr, gate_diag = _tune_residual_gate(y_val, ebm_val, residual_val_pred, conf_val, n_classes)
        val_raw = _combine_ebm_residual_logit(ebm_val, residual_val_pred, scale=gate_scale, gate=np.where(conf_val >= gate_thr, conf_val, 0.0))
        iso_er = _fit_binary_isotonic(y_val, val_raw, n_classes)
        val_cal = _apply_binary_isotonic(iso_er, val_raw)
        mcc_thr_er, mcc_val_er = tune_binary_threshold_mcc(y_val, val_cal)
        sens92_thr_er, sens92_stats_er = tune_binary_threshold_sensitivity_floor(y_val, val_cal, n_classes, specificity_floor=0.92)
        sens95_thr_er, sens95_stats_er = tune_binary_threshold_sensitivity_floor(y_val, val_cal, n_classes, specificity_floor=0.95)
        fit_secs = time.time() - t0

        t0 = time.time()
        diff_post = make_diff_posterior_for_model(model_er, cfg, use_model_reliability=True)
        chunks_resid = []
        chunks_conf = []
        for sl, bp in branch_probs_chunks(model_er, X_test, cfg.batch_size):
            with torch.no_grad():
                z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X_test[sl]).float()).detach().cpu().numpy()
            z_family = _reduce_family_z(z, groups_er)
            feats = _family_llr_feature_matrix(z_family[:, keep_er], theta_er[keep_er], class_prior, top_k=0)
            chunks_resid.append(bias_er + feats @ coef_er)
            chunks_conf.append(_evidence_concentration(z_family[:, keep_er]))
        residual_test_pred = np.concatenate(chunks_resid) if chunks_resid else np.zeros(len(X_test), dtype=np.float64)
        evidence_conf_test = np.concatenate(chunks_conf) if chunks_conf else np.zeros(len(X_test), dtype=np.float64)
        conf_test = np.sqrt(np.clip(_ebm_uncertainty_score(ebm_test) * evidence_conf_test, 0.0, 1.0))
        test_raw = _combine_ebm_residual_logit(ebm_test, residual_test_pred, scale=gate_scale, gate=np.where(conf_test >= gate_thr, conf_test, 0.0))
        test_cal = _apply_binary_isotonic(iso_er, test_raw)
        pred_secs = time.time() - t0
        common_extra = {
            "ebm_residual_family_count": int(len(groups_er)),
            "ebm_residual_selected_families": int(len(keep_er)),
            "ebm_residual_original_branches": int(len(model_er.branches)),
            "ebm_residual_true_weight": float(true_weight),
            "ebm_residual_has_tabpfn_teacher": int(teacher_model is not None),
            "ebm_residual_gate_scale": float(gate_scale),
            "ebm_residual_gate_threshold": float(gate_thr),
            "ebm_residual_bias": float(bias_er),
            "ebm_residual_coef_l1": float(np.sum(np.abs(coef_er))),
            "ebm_residual_isotonic": int(iso_er is not None),
            "ebm_anchor_fit_seconds": float(ebm_fit_secs),
            **gate_diag,
        }
        if "pp_theta_post_ebm_residual_calibrated" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_residual_calibrated",
                y_test, test_cal, n_classes, fit_secs, pred_secs,
                len(keep_er), len(keep_er), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "calibrated-risk", "operating_threshold": 0.5},
            )
        if "pp_theta_post_ebm_residual_mcc" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_residual_mcc",
                y_test, apply_binary_threshold_shift(test_cal, mcc_thr_er), n_classes, fit_secs, pred_secs,
                len(keep_er), len(keep_er), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "mcc", "operating_threshold": float(mcc_thr_er), "operating_val_mcc": float(mcc_val_er)},
            )
        if "pp_theta_post_ebm_residual_sens92" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_residual_sens92",
                y_test, apply_binary_threshold_shift(test_cal, sens92_thr_er), n_classes, fit_secs, pred_secs,
                len(keep_er), len(keep_er), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "sensitivity@spec0.92", "operating_threshold": float(sens92_thr_er), **{f"operating_val_{k}": v for k, v in sens92_stats_er.items()}},
            )
        if "pp_theta_post_ebm_residual_sens95" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_residual_sens95",
                y_test, apply_binary_threshold_shift(test_cal, sens95_thr_er), n_classes, fit_secs, pred_secs,
                len(keep_er), len(keep_er), cfg, rule_source=source_name,
                extra_fields={**common_extra, "operating_mode": "sensitivity@spec0.95", "operating_threshold": float(sens95_thr_er), **{f"operating_val_{k}": v for k, v in sens95_stats_er.items()}},
            )

    bayes_llr_variants = {
        "pp_theta_post_bayes_llr",
        "pp_theta_post_bayes_llr_beta",
        "pp_theta_post_bayes_llr_posneg",
        "pp_theta_post_bayes_llr_posneg_mcc",
        "pp_theta_post_bayes_llr_posneg_sens92",
    }
    if any(v in cfg.variants for v in bayes_llr_variants):
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_bl, theta_bl_raw = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.03, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_bl = estimate_branch_support(model_bl.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_bl_raw = shrink_theta_empirical_bayes(theta_bl_raw, class_prior, support_bl, len(X_train), cfg.theta_shrinkage_strength)
        groups_bl = _branch_family_groups(model_bl.branches, theta_bl_raw)
        support_family_bl = np.array([float(np.sum(support_bl[g])) for g in groups_bl], dtype=np.float64)
        theta_bl = _reduce_family_theta(theta_bl_raw, groups_bl, weights=support_bl)
        theta_bl_beta = _beta_shrink_family_theta(theta_bl, class_prior, support_family_bl)
        top_bl = int(os.environ.get("PPPOST_BAYES_LLR_TOPK", "32"))
        top_bl = min(top_bl, len(groups_bl)) if top_bl > 0 else 0
        z_val_bl = _reduce_family_z(posterior_z_matrix(model_bl, X_val, cfg, use_model_reliability=True), groups_bl)
        val_llr = _aggregate_bayes_llr(z_val_bl, theta_bl, class_prior, top_k=top_bl)
        iso_llr = _fit_binary_isotonic(y_val, val_llr, n_classes)
        val_llr_cal = _apply_binary_isotonic(iso_llr, val_llr)
        val_beta = _aggregate_bayes_llr(z_val_bl, theta_bl_beta, class_prior, top_k=top_bl)
        iso_beta = _fit_binary_isotonic(y_val, val_beta, n_classes)
        val_beta_cal = _apply_binary_isotonic(iso_beta, val_beta)
        pos_scale, neg_scale, conflict_penalty, posneg_diag = _tune_bayes_llr_posneg(
            y_val, z_val_bl, theta_bl_beta, class_prior, n_classes, top_k=top_bl,
        )
        val_posneg = _aggregate_bayes_llr(
            z_val_bl, theta_bl_beta, class_prior, top_k=top_bl,
            pos_scale=pos_scale, neg_scale=neg_scale, conflict_penalty=conflict_penalty,
        )
        iso_posneg = _fit_binary_isotonic(y_val, val_posneg, n_classes)
        val_posneg_cal = _apply_binary_isotonic(iso_posneg, val_posneg)
        mcc_thr_bl, mcc_val_bl = tune_binary_threshold_mcc(y_val, val_posneg_cal)
        sens_thr_bl, sens_stats_bl = tune_binary_threshold_sensitivity_floor(y_val, val_posneg_cal, n_classes, specificity_floor=0.92)
        fit_secs = time.time() - t0

        t0 = time.time()
        diff_post = make_diff_posterior_for_model(model_bl, cfg, use_model_reliability=True)
        chunks_llr = []
        chunks_beta = []
        chunks_posneg = []
        for sl, bp in branch_probs_chunks(model_bl, X_test, cfg.batch_size):
            with torch.no_grad():
                z = diff_post(torch.from_numpy(bp).float(), torch.from_numpy(X_test[sl]).float()).detach().cpu().numpy()
            z_family = _reduce_family_z(z, groups_bl)
            chunks_llr.append(_aggregate_bayes_llr(z_family, theta_bl, class_prior, top_k=top_bl))
            chunks_beta.append(_aggregate_bayes_llr(z_family, theta_bl_beta, class_prior, top_k=top_bl))
            chunks_posneg.append(_aggregate_bayes_llr(
                z_family, theta_bl_beta, class_prior, top_k=top_bl,
                pos_scale=pos_scale, neg_scale=neg_scale, conflict_penalty=conflict_penalty,
            ))
        test_llr = _apply_binary_isotonic(iso_llr, normalize_proba(np.vstack(chunks_llr)))
        test_beta = _apply_binary_isotonic(iso_beta, normalize_proba(np.vstack(chunks_beta)))
        test_posneg = _apply_binary_isotonic(iso_posneg, normalize_proba(np.vstack(chunks_posneg)))
        pred_secs = time.time() - t0

        common_extra = {
            "bayes_llr_family_count": int(len(groups_bl)),
            "bayes_llr_original_branches": int(len(model_bl.branches)),
            "bayes_llr_top_k": int(top_bl),
            "bayes_llr_beta_strength": float(os.environ.get("PPPOST_BAYES_LLR_BETA_STRENGTH", "48")),
        }
        if "pp_theta_post_bayes_llr" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_bayes_llr",
                y_test, test_llr, n_classes, fit_secs, pred_secs,
                len(groups_bl), top_bl, cfg, rule_source=source_name,
                extra_fields={**common_extra, "bayes_llr_beta": 0, "bayes_llr_isotonic": int(iso_llr is not None)},
            )
        if "pp_theta_post_bayes_llr_beta" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_bayes_llr_beta",
                y_test, test_beta, n_classes, fit_secs, pred_secs,
                len(groups_bl), top_bl, cfg, rule_source=source_name,
                extra_fields={**common_extra, "bayes_llr_beta": 1, "bayes_llr_isotonic": int(iso_beta is not None)},
            )
        posneg_extra = {
            **common_extra,
            **posneg_diag,
            "bayes_llr_beta": 1,
            "bayes_llr_pos_scale": float(pos_scale),
            "bayes_llr_neg_scale": float(neg_scale),
            "bayes_llr_conflict_penalty": float(conflict_penalty),
            "bayes_llr_isotonic": int(iso_posneg is not None),
        }
        if "pp_theta_post_bayes_llr_posneg" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_bayes_llr_posneg",
                y_test, test_posneg, n_classes, fit_secs, pred_secs,
                len(groups_bl), top_bl, cfg, rule_source=source_name,
                extra_fields={**posneg_extra, "operating_mode": "calibrated-risk", "operating_threshold": 0.5},
            )
        if "pp_theta_post_bayes_llr_posneg_mcc" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_bayes_llr_posneg_mcc",
                y_test, apply_binary_threshold_shift(test_posneg, mcc_thr_bl), n_classes, fit_secs, pred_secs,
                len(groups_bl), top_bl, cfg, rule_source=source_name,
                extra_fields={**posneg_extra, "operating_mode": "mcc", "operating_threshold": float(mcc_thr_bl), "operating_val_mcc": float(mcc_val_bl)},
            )
        if "pp_theta_post_bayes_llr_posneg_sens92" in cfg.variants:
            evaluate_and_stream(
                rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_bayes_llr_posneg_sens92",
                y_test, apply_binary_threshold_shift(test_posneg, sens_thr_bl), n_classes, fit_secs, pred_secs,
                len(groups_bl), top_bl, cfg, rule_source=source_name,
                extra_fields={**posneg_extra, "operating_mode": "sensitivity@spec0.92", "operating_threshold": float(sens_thr_bl), **{f"operating_val_{k}": v for k, v in sens_stats_bl.items()}},
            )

    if "pp_theta_post_ebm_anchor" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_ea, theta_ea = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=1e-3,
        )
        support_ea = estimate_branch_support(model_ea.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_ea = shrink_theta_empirical_bayes(theta_ea, class_prior, support_ea, len(X_train), cfg.theta_shrinkage_strength)
        z_fit = posterior_z_matrix(model_ea, X_fit, cfg, use_model_reliability=True)
        alpha_ea, class_rel_ea, rule_bias_ea, rule_temp_ea = learn_evidence_layer_v2_params(
            z_fit, theta_ea, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 220)), lr=0.01,
            l1=1e-4, balanced_weight=0.15, brier_weight=0.20, soft_mcc_weight=0.18,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        rule_fit = aggregate_evidence_layer_v2(z_fit, theta_ea, class_prior, alpha_ea, class_rel_ea, rule_bias_ea, rule_temp_ea, top_k=sparse_k)
        z_val = posterior_z_matrix(model_ea, X_val, cfg, use_model_reliability=True)
        rule_val = aggregate_evidence_layer_v2(z_val, theta_ea, class_prior, alpha_ea, class_rel_ea, rule_bias_ea, rule_temp_ea, top_k=sparse_k)
        ebm_fit, ebm_val, ebm_test, ebm_fit_secs = fit_ebm_anchor_proba(X_fit, y_fit, X_val, X_test, n_classes, seed)
        beta_teacher, beta_rule, anchor_bias, anchor_temp = learn_teacher_anchor_params(
            ebm_fit, rule_fit, class_prior, y_fit,
            epochs=max(40, min(int(cfg.expensive_epochs), 220)), lr=0.01,
            balanced_weight=0.08, brier_weight=0.30, soft_mcc_weight=0.15,
            distill_weight=0.05, distill_temperature=2.0, l2=2e-3,
        )
        proba_val = aggregate_teacher_anchored_proba(ebm_val, rule_val, class_prior, beta_teacher, beta_rule, anchor_bias, anchor_temp)
        iso = _fit_binary_isotonic(y_val, proba_val, n_classes)
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_teacher_anchored_chunks(
            model_ea, X_test, theta_ea, class_prior, alpha_ea, class_rel_ea, rule_bias_ea, rule_temp_ea,
            ebm_test, beta_teacher, beta_rule, anchor_bias, anchor_temp, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        proba = _apply_binary_isotonic(iso, proba)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_ebm_anchor",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_ea.branches), sparse_k, cfg, rule_source=source_name,
            extra_fields={"ebm_anchor_beta_ebm": float(beta_teacher), "ebm_anchor_beta_rule": float(beta_rule), "ebm_anchor_isotonic": int(iso is not None)},
        )

    if "pp_theta_post_clinical_objective" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        X_fit, y_fit, X_val, y_val = split_evidence_fit_validation(X_rel, y_rel, n_classes, seed)
        t0 = time.time()
        model_cl, theta_cl = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_fit, y_fit, X_val, y_val, cfg,
            aux_branch_weight=0.08, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        support_cl = estimate_branch_support(model_cl.branches, X_train, cfg.batch_size, cfg.condition_tau)
        theta_cl = shrink_theta_empirical_bayes(theta_cl, class_prior, support_cl, len(X_train), cfg.theta_shrinkage_strength)
        z_fit = posterior_z_matrix(model_cl, X_fit, cfg, use_model_reliability=True)
        alpha_cl, class_rel_cl, bias_cl, temp_cl = learn_evidence_layer_v2_params(
            z_fit, theta_cl, class_prior, y_fit,
            epochs=max(50, min(int(cfg.expensive_epochs), 280)), lr=0.01,
            l1=8e-5, balanced_weight=0.65, brier_weight=0.05, soft_mcc_weight=0.45,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else min(96, max(32, top_k))
        z_val = posterior_z_matrix(model_cl, X_val, cfg, use_model_reliability=True)
        proba_val = aggregate_evidence_layer_v2(z_val, theta_cl, class_prior, alpha_cl, class_rel_cl, bias_cl, temp_cl, top_k=sparse_k)
        threshold, threshold_score = tune_binary_threshold_operating_score(y_val, proba_val, n_classes)
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_evidence_layer_v2_chunks(
            model_cl, X_test, theta_cl, class_prior, alpha_cl, class_rel_cl,
            bias_cl, temp_cl, cfg, top_k=sparse_k, use_model_reliability=True,
        )
        proba = apply_binary_threshold_shift(proba, threshold)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_clinical_objective",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_cl.branches), sparse_k, cfg, rule_source=source_name,
            extra_fields={"clinical_threshold": float(threshold), "clinical_val_score": float(threshold_score)},
        )

    if "pp_theta_post_evlogit_kd" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_kd, theta_kd = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
        )
        z_rel = posterior_z_matrix(model_kd, X_rel, cfg, use_model_reliability=False)
        teacher_rel = teacher_proba_for_source(
            src, fitted, X_rel, n_classes, cfg, y_fallback=y_rel,
        )
        alpha_kd, bias_kd, temp_kd = learn_evidence_logit_params(
            z_rel, theta_kd, class_prior, y_rel,
            epochs=max(20, min(int(cfg.expensive_epochs), 200)),
            lr=0.01,
            l1=1e-4,
            teacher_proba=teacher_rel,
            distill_weight=0.50,
            distill_temperature=2.0,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        proba = predict_evidence_logit_aux_chunks(
            model_kd, X_test, theta_kd, class_prior, alpha_kd, bias_kd, temp_kd,
            cfg, top_k=sparse_k,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_evlogit_kd",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_kd.branches), sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_evlogit_likelihood" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_lik, theta_lik = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
            learn_evidence=True, evidence_reg_weight=2e-3,
        )
        z_rel = posterior_z_matrix(model_lik, X_rel, cfg, use_model_reliability=True)
        alpha_lik, bias_lik, temp_lik = learn_evidence_logit_params(
            z_rel, theta_lik, class_prior, y_rel,
            epochs=max(20, min(int(cfg.expensive_epochs), 200)),
            lr=0.01,
            l1=1e-4,
        )
        reliability = getattr(model_lik, "posterior_evidence_reliability_", None)
        if reliability is not None and len(reliability) == len(alpha_lik):
            alpha_lik = alpha_lik * np.clip(np.asarray(reliability, dtype=np.float64), 0.25, 2.0)
        fit_secs = time.time() - t0
        t0 = time.time()
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        proba = predict_evidence_logit_aux_chunks(
            model_lik, X_test, theta_lik, class_prior, alpha_lik, bias_lik, temp_lik,
            cfg, top_k=sparse_k, use_model_reliability=True,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_evlogit_likelihood",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_lik.branches), sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_evlogit_threshold" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_thr, theta_thr = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
        )
        z_rel = posterior_z_matrix(model_thr, X_rel, cfg, use_model_reliability=False)
        alpha_thr, bias_thr, temp_thr = learn_evidence_logit_params(
            z_rel, theta_thr, class_prior, y_rel,
            epochs=max(20, min(int(cfg.expensive_epochs), 200)),
            lr=0.01,
            l1=1e-4,
        )
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        proba_rel = aggregate_evidence_logit(
            z_rel, theta_thr, class_prior, alpha_thr, bias_thr, temp_thr, top_k=sparse_k,
        )
        threshold, thr_mcc = tune_binary_threshold_mcc(y_rel, proba_rel)
        print(f"    evlogit_threshold threshold={threshold:.4f} val_mcc={thr_mcc:.4f}")
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_evidence_logit_aux_chunks(
            model_thr, X_test, theta_thr, class_prior, alpha_thr, bias_thr, temp_thr,
            cfg, top_k=sparse_k,
        )
        proba = apply_binary_threshold_shift(proba, threshold)
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_evlogit_threshold",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_thr.branches), sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_evlogit_decomp" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_dec, theta_dec = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.05, aux_soft_and="geomean",
        )
        z_rel = posterior_z_matrix(model_dec, X_rel, cfg, use_model_reliability=False)
        alpha_pos, alpha_neg, bias_dec, temp_dec = learn_evidence_decomp_params(
            z_rel, theta_dec, class_prior, y_rel,
            epochs=max(20, min(int(cfg.expensive_epochs), 200)),
            lr=0.01,
            l1=1e-4,
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        sparse_k = cfg.sparse_logit_top_k if cfg.sparse_logit_top_k > 0 else top_k
        proba = predict_evidence_decomp_chunks(
            model_dec, X_test, theta_dec, class_prior, alpha_pos, alpha_neg,
            bias_dec, temp_dec, cfg, top_k=sparse_k,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_evlogit_decomp",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_dec.branches), sparse_k, cfg, rule_source=source_name,
        )

    if "pp_theta_post_aux_v2" in cfg.variants:
        X_rel, y_rel = select_expensive_training_subset(X_train, y_train, cfg, seed)
        t0 = time.time()
        model_a2, theta_a2 = train_ppost_aux_model(
            branches_per_tree, n_features, n_classes, X_rel, y_rel, X_test, y_test, cfg,
            aux_branch_weight=0.15, aux_soft_and="mean",
        )
        fit_secs = time.time() - t0
        t0 = time.time()
        proba = predict_diff_posterior_wmean_chunks(
            model_a2, X_test, theta_a2, cfg.batch_size,
            tau=cfg.condition_tau,
            p_high=cfg.posterior_p_high,
            p_low=cfg.posterior_p_low,
        )
        evaluate_and_stream(
            rows, csv_path, jsonl_path, ds.name, fold, "pp_theta_post_aux_v2",
            y_test, proba, n_classes, fit_secs, time.time() - t0,
            len(model_a2.branches), top_k, cfg, rule_source=source_name,
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
        theta_l, alpha_l = learn_theta_alpha(
            post_sub, theta, y_sub, epochs=cfg.expensive_epochs
        )
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
            branches_per_tree,
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
            branches_per_tree,
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
            branches_per_tree,
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
            branches_per_tree,
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
            branches_per_tree,
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
    p.add_argument(
        "--append-results-to",
        default=None,
        help="Append streamed rows to this existing/new CSV instead of creating a timestamped file",
    )
    p.add_argument(
        "--append-jsonl-to",
        default=None,
        help="Optional JSONL path paired with --append-results-to; defaults to CSV path with .jsonl",
    )
    p.add_argument("--top-k-ratio", type=float, default=0.30)
    p.add_argument("--top-k-min", type=int, default=5)
    p.add_argument("--top-k-max", type=int, default=100)
    p.add_argument("--condition-tau", type=float, default=1.0)
    p.add_argument(
        "--posterior-p-high",
        type=float,
        default=0.95,
        help="Likelihood P(evidence atom matches | rule active) for configurable posterior variants",
    )
    p.add_argument(
        "--posterior-p-low",
        type=float,
        default=0.05,
        help="Likelihood P(evidence atom matches | rule inactive) for configurable posterior variants",
    )
    p.add_argument(
        "--theta-shrinkage-strength",
        type=float,
        default=32.0,
        help="Empirical-Bayes pseudo-count strength for pp_theta_post_shrink_theta",
    )
    p.add_argument(
        "--signed-logit-temperature",
        type=float,
        default=1.0,
        help="Temperature for signed log-odds PPtheta aggregation",
    )
    p.add_argument(
        "--sparse-logit-top-k",
        type=int,
        default=0,
        help="Top-k posterior branches for sparse signed-logit aggregation; 0 uses top-k ratio/min/max",
    )
    p.add_argument(
        "--rule-budget",
        type=int,
        default=0,
        help="Max symbolic rules kept for PPtheta heads after purity/support ranking; 0 keeps all",
    )
    p.add_argument(
        "--rule-max-depth",
        type=int,
        default=0,
        help="Truncate rule conditions to this depth before selection; 0 keeps full rules",
    )
    p.add_argument(
        "--rule-min-support",
        type=float,
        default=0.0,
        help="Drop rules with soft empirical support below this fraction before budget selection",
    )
    p.add_argument(
        "--rule-selection",
        choices=("purity", "diverse"),
        default="purity",
        help="Rule budget selection policy: purity ranking or class/feature-diverse round robin",
    )
    p.add_argument("--hybrid-lam", type=float, default=0.5)
    p.add_argument("--max-onehot-cardinality", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-roc-auc", action="store_true", help="Skip ROC AUC for very large runs")
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write per-fold y_true/proba prediction_artifacts next to the CSV for bootstrap, calibration, and audit analyses",
    )
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
        posterior_p_high=float(np.clip(args.posterior_p_high, 1e-6, 1.0 - 1e-6)),
        posterior_p_low=float(np.clip(args.posterior_p_low, 1e-6, 1.0 - 1e-6)),
        theta_shrinkage_strength=float(max(args.theta_shrinkage_strength, 0.0)),
        signed_logit_temperature=float(max(args.signed_logit_temperature, 1e-6)),
        sparse_logit_top_k=int(max(args.sparse_logit_top_k, 0)),
        rule_budget=int(max(args.rule_budget, 0)),
        rule_max_depth=int(max(args.rule_max_depth, 0)),
        rule_min_support=float(max(args.rule_min_support, 0.0)),
        rule_selection=str(args.rule_selection),
        hybrid_lam=args.hybrid_lam,
        max_onehot_cardinality=args.max_onehot_cardinality,
        n_jobs=args.n_jobs,
        no_roc_auc=args.no_roc_auc,
        save_predictions=bool(args.save_predictions),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.append_results_to:
        csv_path = Path(args.append_results_to)
        jsonl_path = (
            Path(args.append_jsonl_to)
            if args.append_jsonl_to
            else csv_path.with_suffix(".jsonl")
        )
    else:
        csv_path = cfg.output_dir / f"compare_datasets_{stamp}.csv"
        jsonl_path = cfg.output_dir / f"compare_datasets_{stamp}.jsonl"

    print("=" * 100)
    print("PPtheta-Post large-dataset comparison")
    print(f"started={dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"variants={','.join(cfg.variants)}")
    print(
        "posterior="
        f"tau={cfg.condition_tau} p_high={cfg.posterior_p_high} p_low={cfg.posterior_p_low} "
        f"theta_shrink={cfg.theta_shrinkage_strength} signed_temp={cfg.signed_logit_temperature}"
    )
    print(
        "rule_resource="
        f"budget={cfg.rule_budget} max_depth={cfg.rule_max_depth} "
        f"min_support={cfg.rule_min_support} selection={cfg.rule_selection}"
    )
    print(f"output_csv={csv_path}")
    print(f"output_jsonl={jsonl_path}")
    print(f"append_mode={bool(args.append_results_to)}")
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
