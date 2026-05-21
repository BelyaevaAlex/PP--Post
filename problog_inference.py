"""ProbLog inference engine for RuleNetwork.

Replaces the neural softmax(W2 @ h) with full probabilistic inference:
    1. Neural net computes P(z(b,X)) for each branch b
    2. Classification head: theta_bk :: supports(k, b, X) :- z(b, X).
    3. Aggregation:         class(X, K) :- supports(K, _, X).
    4. Queries P(class(X,k)) for each class k; argmax → predicted class

Three inference modes:
    - "fast"  : noisy-or over prior P(z)  [vectorized NumPy, no evidence]
    - "full"  : analytical posterior P(z|evidence) + noisy-or  [vectorized, exact]
    - "full_problog" : ProbLog engine per sample  [slow, for verification only]

The "full" mode is **mathematically identical** to what ProbLog computes
for our latent-variable model with scoped manifestation rules, because:
    - Each branch has its own set of condition atoms (scoped)
    - Conditions are conditionally independent given z(b,X)
    - The posterior P(z(b,X) | evidence_b) factorises per branch

Therefore the analytical computation:
    P(z|ev) = P(z) · P(ev|z=1) / [P(z)·P(ev|z=1) + (1-P(z))·P(ev|z=0)]
with depth-adjusted per-condition probabilities (branch b has m_b conditions):
    p_h_b = p_high^(1/m_b),   p_l_b = p_low^(1/m_b)
    P(ev|z=1) = p_h_b^n_match · (1 - p_h_b)^n_miss
    P(ev|z=0) = p_l_b^n_match · (1 - p_l_b)^n_miss

gives the exact posterior (identical to ProbLog), which is then plugged into
noisy-or for classification.  The same adjusted probabilities are used in
``export_full_problog_program`` so all three modes produce identical results.

Usage:
    from problog_inference import ProbLogClassifier
    clf = ProbLogClassifier(branches, n_classes, mode="full")
    proba = clf.predict_proba(branch_probs, X_test)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

from branch_schema import Branch, Condition
from problog_export import (
    export_full_problog_program,
    _class_proportions_to_theta,
    classification_head_rules,
    class_aggregation_rules,
    query_rules_for_sample,
)


# ─────────────────────────────────────────────────────────────
# Condition evaluation helpers
# ─────────────────────────────────────────────────────────────

def _evaluate_conditions_batch(
    branches: List[Branch],
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate branch conditions for all samples (vectorized).

    For each branch b and sample x, count how many of b's conditions
    are satisfied (n_match) and how many are not (n_miss).

    Parameters
    ----------
    branches : list of Branch (length n_branches)
    X : np.ndarray of shape [n_samples, n_features]

    Returns
    -------
    n_match : np.ndarray of shape [n_samples, n_branches]
        Number of satisfied conditions per branch per sample.
    n_total : np.ndarray of shape [n_branches]
        Total number of conditions per branch.
    """
    n_samples = X.shape[0]
    n_branches = len(branches)
    n_match = np.zeros((n_samples, n_branches), dtype=np.float64)
    n_total = np.zeros(n_branches, dtype=np.float64)

    for br_idx, branch in enumerate(branches):
        conditions = branch.conditions
        n_conds = len(conditions)
        n_total[br_idx] = n_conds

        if n_conds == 0:
            # Branch with no conditions (root): all conditions trivially match
            n_match[:, br_idx] = 0.0
            continue

        # Vectorized evaluation of all conditions for this branch
        satisfied = np.zeros((n_samples, n_conds), dtype=bool)
        for c_idx, cond in enumerate(conditions):
            feat_vals = X[:, cond.feature_idx]
            if cond.direction == "le":
                satisfied[:, c_idx] = feat_vals <= cond.threshold
            else:  # "gt"
                satisfied[:, c_idx] = feat_vals > cond.threshold

        n_match[:, br_idx] = satisfied.sum(axis=1)

    return n_match, n_total


def _evaluate_conditions_soft_batch(
    branches: List[Branch],
    X: np.ndarray,
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate branch conditions with *soft* sigmoid matching.

    Instead of binary match (0 or 1), each condition yields a soft score
    in (0, 1) via sigmoid:
        match = σ((threshold − x_j) / τ)   for "le" conditions
        match = σ((x_j − threshold) / τ)   for "gt" conditions

    A value right at the threshold → 0.5; far inside → ~1.0; far outside → ~0.0.
    `τ` (tau) controls sharpness: smaller τ = sharper (closer to binary).

    Parameters
    ----------
    branches : list of Branch
    X : [n_samples, n_features]
    tau : float — temperature for sigmoid.  τ=1.0 is a good default for
          normalised features.  For un-normalised features, set τ to a
          fraction of the typical feature standard deviation.

    Returns
    -------
    soft_match : [n_samples, n_branches] — soft count of matching conditions
    n_total : [n_branches] — total number of conditions per branch
    """
    n_samples = X.shape[0]
    n_branches = len(branches)
    soft_match = np.zeros((n_samples, n_branches), dtype=np.float64)
    n_total = np.zeros(n_branches, dtype=np.float64)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    for br_idx, branch in enumerate(branches):
        conditions = branch.conditions
        n_conds = len(conditions)
        n_total[br_idx] = n_conds

        if n_conds == 0:
            continue

        cond_scores = np.zeros((n_samples, n_conds), dtype=np.float64)
        for c_idx, cond in enumerate(conditions):
            feat_vals = X[:, cond.feature_idx]
            if cond.direction == "le":
                cond_scores[:, c_idx] = _sigmoid((cond.threshold - feat_vals) / tau)
            else:  # "gt"
                cond_scores[:, c_idx] = _sigmoid((feat_vals - cond.threshold) / tau)

        soft_match[:, br_idx] = cond_scores.sum(axis=1)

    return soft_match, n_total


def compute_tempered_posterior(
    branches: List[Branch],
    branch_probs: np.ndarray,
    X: np.ndarray,
    p_high: float = 0.95,
    p_low: float = 0.05,
    beta: float = 0.5,
) -> np.ndarray:
    """Compute posterior P(z | evidence) with *tempered* evidence (power posterior).

    Standard Bayes:
        log P(z|ev) ∝ log P(z) + log L(ev|z)

    Tempered:
        log P(z|ev) ∝ log P(z) + β · log L(ev|z)

    β = 1.0  → standard Bayesian posterior
    β = 0.0  → ignore evidence entirely (posterior = prior)
    β ∈ (0,1) → softer evidence — posterior changes but doesn't collapse

    This is a well-studied technique: "power posteriors", "tempered likelihoods",
    "Safe Bayes" (Grünwald 2012).

    Parameters
    ----------
    branches : list of Branch
    branch_probs : [n_samples, n_branches] — prior P(z)
    X : [n_samples, n_features]
    p_high, p_low : float — per-condition probabilities
    beta : float ∈ [0, 1] — tempering coefficient

    Returns
    -------
    posterior_z : [n_samples, n_branches] — tempered posterior
    """
    probs = np.asarray(branch_probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    n_match, n_total = _evaluate_conditions_batch(branches, X)
    n_miss = n_total[np.newaxis, :] - n_match

    n_total_safe = np.where(n_total > 0, n_total, 1.0)
    p_high_adj = p_high ** (1.0 / n_total_safe)
    p_low_adj  = p_low  ** (1.0 / n_total_safe)

    log_lik_true  = (n_match * np.log(p_high_adj)[np.newaxis, :]
                     + n_miss * np.log(1.0 - p_high_adj)[np.newaxis, :])
    log_lik_false = (n_match * np.log(p_low_adj)[np.newaxis, :]
                     + n_miss * np.log(1.0 - p_low_adj)[np.newaxis, :])

    pz = np.clip(probs, 1e-15, 1.0 - 1e-15)

    # Tempered: multiply log-likelihood by β
    log_jt = np.log(pz)       + beta * log_lik_true
    log_jf = np.log(1.0 - pz) + beta * log_lik_false

    max_log = np.maximum(log_jt, log_jf)
    log_ev = max_log + np.log(np.exp(log_jt - max_log) + np.exp(log_jf - max_log))

    return np.exp(log_jt - log_ev)


def compute_soft_posterior(
    branches: List[Branch],
    branch_probs: np.ndarray,
    X: np.ndarray,
    p_high: float = 0.95,
    p_low: float = 0.05,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute posterior P(z | soft evidence) using sigmoid-based matching.

    Instead of binary condition evaluation (match=0 or match=1), uses:
        soft_match = σ((threshold − x_j) / τ)

    The log-likelihood becomes:
        log L(ev|z=1) = Σ_i [s_i · log p_h + (1-s_i) · log(1-p_h)]
    where s_i ∈ (0,1) is the soft match score.

    This means a value *close* to the threshold contributes weakly (≈ neutral),
    while a value far inside or outside contributes strongly.

    Parameters
    ----------
    branches : list of Branch
    branch_probs : [n_samples, n_branches]
    X : [n_samples, n_features]
    p_high, p_low : float
    tau : float — sigmoid temperature

    Returns
    -------
    posterior_z : [n_samples, n_branches]
    """
    probs = np.asarray(branch_probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    soft_match, n_total = _evaluate_conditions_soft_batch(branches, X, tau=tau)
    soft_miss = n_total[np.newaxis, :] - soft_match

    n_total_safe = np.where(n_total > 0, n_total, 1.0)
    p_high_adj = p_high ** (1.0 / n_total_safe)
    p_low_adj  = p_low  ** (1.0 / n_total_safe)

    log_lik_true  = (soft_match * np.log(p_high_adj)[np.newaxis, :]
                     + soft_miss * np.log(1.0 - p_high_adj)[np.newaxis, :])
    log_lik_false = (soft_match * np.log(p_low_adj)[np.newaxis, :]
                     + soft_miss * np.log(1.0 - p_low_adj)[np.newaxis, :])

    pz = np.clip(probs, 1e-15, 1.0 - 1e-15)
    log_jt = np.log(pz)       + log_lik_true
    log_jf = np.log(1.0 - pz) + log_lik_false

    max_log = np.maximum(log_jt, log_jf)
    log_ev = max_log + np.log(np.exp(log_jt - max_log) + np.exp(log_jf - max_log))

    return np.exp(log_jt - log_ev)


def mix_prior_posterior(
    prior_z: np.ndarray,
    posterior_z: np.ndarray,
    lam: float = 0.5,
) -> np.ndarray:
    """Linear interpolation between prior and posterior.

    z_eff = λ · prior + (1 - λ) · posterior

    λ = 1.0 → pure prior (no evidence)
    λ = 0.0 → pure posterior
    λ = 0.5 → balanced mix

    This ensures that even when posterior collapses, the prior contribution
    keeps branches partially "alive".

    Parameters
    ----------
    prior_z : [n_samples, n_branches]
    posterior_z : [n_samples, n_branches]
    lam : float ∈ [0, 1]

    Returns
    -------
    [n_samples, n_branches]
    """
    return lam * prior_z + (1.0 - lam) * posterior_z


def compute_adaptive_tempered_posterior(
    branches: List[Branch],
    branch_probs: np.ndarray,
    X: np.ndarray,
    p_high: float = 0.95,
    p_low: float = 0.05,
    beta_base: float = 0.5,
    depth_ref: float = None,
) -> np.ndarray:
    """Compute posterior with *adaptive* tempering — β varies per branch.

    Motivation: branches with **few** conditions have reliable evidence
    (β can be higher = trust evidence more), while branches with **many**
    conditions risk multiplicative collapse (β should be lower = trust less).

    Formula:
        β_b = beta_base × (m_ref / m_b)^0.5

    where m_b = number of conditions in branch b,
          m_ref = median m across branches (or user-specified).

    This means:
        - Branch with m_b = m_ref → β = beta_base  (baseline)
        - Branch with m_b = 4*m_ref → β = beta_base/2  (halved — deep branch)
        - Branch with m_b = m_ref/4 → β = 2*beta_base  (doubled — shallow branch)

    The β values are clamped to [0.05, 0.95] to prevent extremes.

    Parameters
    ----------
    branches : list of Branch
    branch_probs : [n_samples, n_branches]
    X : [n_samples, n_features]
    p_high, p_low : float
    beta_base : float — baseline tempering coefficient
    depth_ref : float or None — reference depth (default: median of branch depths)

    Returns
    -------
    posterior_z : [n_samples, n_branches]
    """
    probs = np.asarray(branch_probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    n_match, n_total = _evaluate_conditions_batch(branches, X)
    n_miss = n_total[np.newaxis, :] - n_match

    # Depth-adjusted p_high/p_low (same as standard analytical)
    n_total_safe = np.where(n_total > 0, n_total, 1.0)
    p_high_adj = p_high ** (1.0 / n_total_safe)
    p_low_adj  = p_low  ** (1.0 / n_total_safe)

    log_lik_true  = (n_match * np.log(p_high_adj)[np.newaxis, :]
                     + n_miss * np.log(1.0 - p_high_adj)[np.newaxis, :])
    log_lik_false = (n_match * np.log(p_low_adj)[np.newaxis, :]
                     + n_miss * np.log(1.0 - p_low_adj)[np.newaxis, :])

    # Adaptive β per branch: β_b = beta_base * sqrt(m_ref / m_b)
    if depth_ref is None:
        depths = n_total[n_total > 0]
        depth_ref = float(np.median(depths)) if len(depths) > 0 else 1.0

    beta_per_branch = np.where(
        n_total > 0,
        beta_base * np.sqrt(depth_ref / n_total_safe),
        0.0,
    )
    beta_per_branch = np.clip(beta_per_branch, 0.05, 0.95)  # [n_branches]

    pz = np.clip(probs, 1e-15, 1.0 - 1e-15)

    # Tempered with per-branch β
    log_jt = np.log(pz)       + beta_per_branch[np.newaxis, :] * log_lik_true
    log_jf = np.log(1.0 - pz) + beta_per_branch[np.newaxis, :] * log_lik_false

    max_log = np.maximum(log_jt, log_jf)
    log_ev = max_log + np.log(np.exp(log_jt - max_log) + np.exp(log_jf - max_log))

    return np.exp(log_jt - log_ev)


def compute_match_informed_adaptive_posterior(
    branches: List[Branch],
    branch_probs: np.ndarray,
    X: np.ndarray,
    p_high: float = 0.95,
    p_low: float = 0.05,
    beta_base: float = 0.5,
    depth_ref: float = None,
    match_boost: float = 0.5,
) -> np.ndarray:
    """Adaptive posterior where β depends on BOTH branch depth AND match quality.

    Key insight: even for a deep branch, if ALL conditions are satisfied,
    the evidence is highly reliable → β should be higher.  Conversely,
    a shallow branch where most conditions fail provides unreliable evidence.

    Formula:
        β_b,x = β_depth_b  ×  (ε + match_boost × match_ratio_b,x)

    where β_depth_b = beta_base × √(m_ref / m_b)  [depth part, same as adaptive]
          match_ratio_b,x = n_match / n_total     [sample-dependent part]
          ε = 1 − match_boost  (ensures β_min = β_depth × ε > 0)

    The result is a **per-sample, per-branch** tempering coefficient,
    which is the most fine-grained form of evidence weighting.

    Parameters
    ----------
    branches : list of Branch
    branch_probs : [n_samples, n_branches]
    X : [n_samples, n_features]
    p_high, p_low : float
    beta_base : float — baseline tempering
    depth_ref : float or None — reference depth (default: median)
    match_boost : float ∈ [0, 1] — how much match_ratio influences β.
        0 = pure depth-adaptive (no match influence);
        1 = full match influence (β ∈ [0, β_depth] depending on match).

    Returns
    -------
    posterior_z : [n_samples, n_branches]
    """
    probs = np.asarray(branch_probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    n_match, n_total = _evaluate_conditions_batch(branches, X)
    n_miss = n_total[np.newaxis, :] - n_match

    # Depth-adjusted p_high/p_low
    n_total_safe = np.where(n_total > 0, n_total, 1.0)
    p_high_adj = p_high ** (1.0 / n_total_safe)
    p_low_adj  = p_low  ** (1.0 / n_total_safe)

    log_lik_true  = (n_match * np.log(p_high_adj)[np.newaxis, :]
                     + n_miss * np.log(1.0 - p_high_adj)[np.newaxis, :])
    log_lik_false = (n_match * np.log(p_low_adj)[np.newaxis, :]
                     + n_miss * np.log(1.0 - p_low_adj)[np.newaxis, :])

    # Depth-adaptive β component
    if depth_ref is None:
        depths = n_total[n_total > 0]
        depth_ref = float(np.median(depths)) if len(depths) > 0 else 1.0

    beta_depth = np.where(
        n_total > 0,
        beta_base * np.sqrt(depth_ref / n_total_safe),
        0.0,
    )  # [n_branches]

    # Match-informed component: per-sample, per-branch
    match_ratio = n_match / n_total_safe[np.newaxis, :]  # [n_samples, n_branches]
    eps = 1.0 - match_boost
    beta_matrix = beta_depth[np.newaxis, :] * (eps + match_boost * match_ratio)
    beta_matrix = np.clip(beta_matrix, 0.05, 0.95)  # [n_samples, n_branches]

    pz = np.clip(probs, 1e-15, 1.0 - 1e-15)

    # Tempered with per-sample, per-branch β
    log_jt = np.log(pz)       + beta_matrix * log_lik_true
    log_jf = np.log(1.0 - pz) + beta_matrix * log_lik_false

    max_log = np.maximum(log_jt, log_jf)
    log_ev = max_log + np.log(np.exp(log_jt - max_log) + np.exp(log_jf - max_log))

    return np.exp(log_jt - log_ev)


def find_optimal_beta(
    branches: List[Branch],
    branch_probs: np.ndarray,
    X: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    p_high: float = 0.95,
    p_low: float = 0.05,
    min_theta: float = 1e-6,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find optimal β (tempering coefficient) via grid search on given data.

    For each β, computes tempered posterior → weighted mean → accuracy.
    Returns (best_beta, best_accuracy).

    Parameters
    ----------
    branches : list of Branch
    branch_probs : [n_samples, n_branches] — prior P(z)
    X : [n_samples, n_features]
    y_true : [n_samples]
    n_classes : int
    p_high, p_low : float
    min_theta : float
    grid : optional array of β values to try

    Returns
    -------
    (best_beta, best_accuracy)
    """
    if grid is None:
        grid = np.arange(0.05, 1.01, 0.05)

    theta = build_theta_matrix(branches, n_classes, min_theta=min_theta)
    best_beta, best_acc = 0.5, -1.0

    for beta in grid:
        z = compute_tempered_posterior(branches, branch_probs, X,
                                       p_high=p_high, p_low=p_low, beta=beta)
        proba = aggregate_weighted_mean(z, theta)
        pred = np.argmax(proba, axis=1)
        acc = float(np.mean(pred == y_true))
        if acc > best_acc:
            best_acc = acc
            best_beta = float(beta)

    return best_beta, best_acc


def _compute_analytical_posterior(
    branches: List[Branch],
    branch_probs: np.ndarray,
    X: np.ndarray,
    n_classes: int,
    p_high: float = 0.95,
    p_low: float = 0.05,
    min_theta: float = 1e-6,
) -> np.ndarray:
    """Compute class probabilities via analytical posterior + noisy-or.

    **Mathematically identical** to ProbLog inference for the latent-variable
    model with scoped manifestation rules and depth-normalised per-condition
    probabilities: for a branch with *m* conditions, each condition uses
    ``p_high^(1/m)`` and ``p_low^(1/m)`` instead of raw ``p_high``/``p_low``.

    This prevents posterior collapse for deep branches (many conditions)
    while ensuring that the analytical computation matches the ProbLog
    program exactly (see ``export_full_problog_program``).

    Steps:
        1. Evaluate branch conditions → n_match, n_miss per (sample, branch)
        2. Compute depth-adjusted per-condition probs:
           p_high_b = p_high^(1/m_b), p_low_b = p_low^(1/m_b)
        3. Bayes update with adjusted likelihoods → posterior P(z | evidence)
        4. Build theta matrix from class_proportions
        5. Noisy-or with posterior z → class probabilities

    Parameters
    ----------
    branches : list of Branch
    branch_probs : np.ndarray [n_samples, n_branches]
    X : np.ndarray [n_samples, n_features]
    n_classes : int
    p_high, p_low : float
    min_theta : float

    Returns
    -------
    np.ndarray [n_samples, n_classes]
    """
    probs = np.asarray(branch_probs, dtype=np.float64)
    single = probs.ndim == 1
    if single:
        probs = probs[np.newaxis, :]

    n_samples, n_branches = probs.shape

    # ── Step 1: Evaluate conditions ────────────────────────
    n_match, n_total = _evaluate_conditions_batch(branches, X)
    n_miss = n_total[np.newaxis, :] - n_match   # [n_samples, n_branches]

    # ── Step 2: Depth-adjusted per-condition probabilities ─
    # For branch b with m_b conditions: p_h_b = p_high^(1/m_b)
    n_total_safe = np.where(n_total > 0, n_total, 1.0)  # [n_branches]
    p_high_adj = p_high ** (1.0 / n_total_safe)          # [n_branches]
    p_low_adj  = p_low  ** (1.0 / n_total_safe)          # [n_branches]

    log_p_high_adj    = np.log(p_high_adj)                # [n_branches]
    log_1m_p_high_adj = np.log(1.0 - p_high_adj)         # [n_branches]
    log_p_low_adj     = np.log(p_low_adj)                 # [n_branches]
    log_1m_p_low_adj  = np.log(1.0 - p_low_adj)          # [n_branches]

    # Log-likelihoods with adjusted per-condition probabilities
    # This is EXACTLY what ProbLog computes with the adjusted program
    log_lik_z_true  = (n_match * log_p_high_adj[np.newaxis, :]
                       + n_miss * log_1m_p_high_adj[np.newaxis, :])
    log_lik_z_false = (n_match * log_p_low_adj[np.newaxis, :]
                       + n_miss * log_1m_p_low_adj[np.newaxis, :])

    # Clamp prior
    pz = np.clip(probs, 1e-15, 1.0 - 1e-15)

    # log P(z=true, ev)  and  log P(z=false, ev)
    log_joint_true  = np.log(pz)       + log_lik_z_true
    log_joint_false = np.log(1.0 - pz) + log_lik_z_false

    # log-sum-exp
    max_log = np.maximum(log_joint_true, log_joint_false)
    log_evidence = max_log + np.log(
        np.exp(log_joint_true  - max_log) +
        np.exp(log_joint_false - max_log)
    )

    posterior_z = np.exp(log_joint_true - log_evidence)
    # [n_samples, n_branches]

    # ── Step 3: Theta matrix ───────────────────────────────
    theta = np.zeros((n_branches, n_classes), dtype=np.float64)
    for br_idx, branch in enumerate(branches):
        t = _class_proportions_to_theta(branch)
        if not t:
            continue
        for k in range(min(n_classes, len(t))):
            theta[br_idx, k] = max(min(t[k], 1.0 - min_theta), min_theta)

    # ── Step 4: Noisy-or with posterior z ──────────────────
    p_support = posterior_z[:, :, np.newaxis] * theta[np.newaxis, :, :]

    log_complement = np.log1p(-np.clip(p_support, 0, 1 - 1e-15))
    log_noisy_or = np.sum(log_complement, axis=1)
    class_probs = 1.0 - np.exp(log_noisy_or)

    # Normalise
    row_sums = class_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    class_probs = class_probs / row_sums

    if single:
        return class_probs[0]
    return class_probs


# ─────────────────────────────────────────────────────────────
# Fast mode: noisy-or without evidence
# ─────────────────────────────────────────────────────────────

def _compute_noisy_or_fast(
    branches: List[Branch],
    branch_probs: np.ndarray,
    n_classes: int,
    min_theta: float = 1e-6,
) -> np.ndarray:
    """Compute class probabilities via noisy-or in NumPy (vectorized).

    This is the analytical equivalent of ProbLog inference for the program:
        pZ :: z(b, x_id).
        theta_bk :: supports(k, b, X) :- z(b, X).
        class(X, K) :- supports(K, _, X).

    ProbLog semantics: P(class(x,k)) = 1 - Π_b (1 - θ_bk · P(z(b,x)))

    Parameters
    ----------
    branches : list of Branch (len = n_branches)
    branch_probs : np.ndarray of shape [n_samples, n_branches] or [n_branches]
    n_classes : int
    min_theta : float

    Returns
    -------
    np.ndarray of shape [n_samples, n_classes] or [n_classes]
    """
    probs = np.asarray(branch_probs, dtype=np.float64)
    single = probs.ndim == 1
    if single:
        probs = probs[np.newaxis, :]

    n_samples, n_branches = probs.shape

    # Build theta matrix [n_branches, n_classes]
    theta = np.zeros((n_branches, n_classes), dtype=np.float64)
    for br_idx, branch in enumerate(branches):
        t = _class_proportions_to_theta(branch)
        if not t:
            continue
        for k in range(min(n_classes, len(t))):
            theta[br_idx, k] = max(min(t[k], 1.0 - min_theta), min_theta)

    # P(supports(k,b,x)) = theta_bk * P(z(b,x))
    p_support = probs[:, :, np.newaxis] * theta[np.newaxis, :, :]

    # Noisy-or: P(class(x,k)) = 1 - Π_b (1 - p_support(k,b,x))
    log_complement = np.log1p(-np.clip(p_support, 0, 1 - 1e-15))
    log_noisy_or = np.sum(log_complement, axis=1)
    class_probs = 1.0 - np.exp(log_noisy_or)

    # Normalize to probability distribution
    row_sums = class_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    class_probs = class_probs / row_sums

    if single:
        return class_probs[0]
    return class_probs


# ─────────────────────────────────────────────────────────────
# ProbLog engine (slow, for verification only)
# ─────────────────────────────────────────────────────────────

def _build_fast_program(
    branches: List[Branch],
    branch_probs_single: np.ndarray,
    x_id: int,
    n_classes: int,
    min_theta: float = 1e-6,
) -> str:
    """Build a lightweight ProbLog program (no evidence/manifestation).

    Note: for batch inference, use _compute_noisy_or_fast() instead.
    This function is kept for debugging / single-sample ProbLog verification.
    """
    lines: List[str] = []

    probs = np.asarray(branch_probs_single)
    for br_idx, branch in enumerate(branches):
        pz = float(probs[br_idx])
        pz = max(min(pz, 1.0 - 1e-8), 1e-8)
        lines.append(f"{pz:.8f}::z({branch.branch_id},{x_id}).")
    lines.append('')

    lines.extend(classification_head_rules(branches, n_classes, min_theta))
    lines.append('')

    lines.extend(class_aggregation_rules(n_classes))
    lines.append('')

    lines.extend(query_rules_for_sample(x_id, n_classes))

    return '\n'.join(lines)


def _run_problog_inference(program_text: str, n_classes: int, x_id) -> np.ndarray:
    """Run ProbLog inference on a program and return class probabilities.

    Returns
    -------
    np.ndarray of shape [n_classes] with P(class(x_id, k)) for each k.
    """
    from problog.program import PrologString
    from problog import get_evaluatable

    try:
        model = PrologString(program_text)
        result = get_evaluatable().create_from(model).evaluate()
    except Exception as e:
        print(f"  [ProbLog] Inference failed for sample {x_id}: {e}")
        return np.ones(n_classes) / n_classes

    probs = np.zeros(n_classes)
    for key, val in result.items():
        key_str = str(key)
        for k in range(n_classes):
            if f"class({x_id},{k})" in key_str:
                probs[k] = float(val)
                break

    if probs.sum() <= 0:
        return np.ones(n_classes) / n_classes
    return probs


# ─────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────

class ProbLogClassifier:
    """ProbLog-based classifier for RuleNetwork.

    Replaces the frozen W2 head with probabilistic rules:
        theta_bk :: supports(k, b, X) :- z(b, X).
        class(X, K) :- supports(K, _, X).

    The theta values come from branch class_proportions (same as W2).

    Three modes
    -----------
    "fast"
        Noisy-or with prior P(z(b,x)) from the neural network.
        No evidence, no manifestation.  Vectorized.  < 1 ms.

    "full"
        Analytical Bayesian posterior: updates P(z) with observed condition
        evidence, then noisy-or.  Vectorized.  < 10 ms.
        **Mathematically identical to ProbLog inference** for our model
        (scoped conditions, conditional independence given z).

    "full_problog"
        Calls ProbLog engine per sample.  Very slow (~20 s/sample).
        Kept only for verification that the analytical mode is correct.

    Parameters
    ----------
    branches : list of Branch
    n_classes : int
    mode : str — "fast", "full", or "full_problog"
    p_high : float — P(condition | z=true) for manifestation  (full modes)
    p_low : float  — P(condition | z=false) for manifestation (full modes)
    min_theta : float
    """

    VALID_MODES = ("fast", "full", "full_problog")

    def __init__(
        self,
        branches: List[Branch],
        n_classes: int,
        mode: str = "fast",
        p_high: float = 0.95,
        p_low: float = 0.05,
        min_theta: float = 1e-6,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.branches = branches
        self.n_classes = n_classes
        self.mode = mode
        self.p_high = p_high
        self.p_low = p_low
        self.min_theta = min_theta

    # ── Public API ─────────────────────────────────────────

    def predict_proba(
        self,
        branch_probs: np.ndarray,
        X: np.ndarray = None,
        verbose: bool = True,
        top_k_branches: int = None,
    ) -> np.ndarray:
        """Predict class probabilities for all samples.

        Parameters
        ----------
        branch_probs : np.ndarray [n_samples, n_branches]
        X : np.ndarray [n_samples, n_features]
            Required for "full" and "full_problog" modes.
        verbose : bool
        top_k_branches : int, optional
            For "full_problog" mode only.

        Returns
        -------
        np.ndarray [n_samples, n_classes]
        """
        if self.mode == "fast":
            return _compute_noisy_or_fast(
                self.branches, branch_probs, self.n_classes, self.min_theta,
            )

        if self.mode == "full":
            if X is None:
                raise ValueError("X is required for mode='full'")
            return _compute_analytical_posterior(
                branches=self.branches,
                branch_probs=branch_probs,
                X=X,
                n_classes=self.n_classes,
                p_high=self.p_high,
                p_low=self.p_low,
                min_theta=self.min_theta,
            )

        # mode == "full_problog"
        return self._predict_proba_problog(branch_probs, X, verbose, top_k_branches)

    def predict(
        self,
        branch_probs: np.ndarray,
        X: np.ndarray = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(branch_probs, X, verbose=verbose)
        return np.argmax(proba, axis=1)

    # ── Diagnostics ────────────────────────────────────────

    def get_posterior_z(
        self,
        branch_probs: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return posterior P(z(b,x) | evidence) for all branches and samples.

        Uses depth-adjusted per-condition probabilities:
        ``p_high_b = p_high^(1/m_b)`` for branch b with m_b conditions.
        This is consistent with the ProbLog program generated by
        ``export_full_problog_program``.

        Returns
        -------
        np.ndarray [n_samples, n_branches]
        """
        probs = np.asarray(branch_probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        n_match, n_total = _evaluate_conditions_batch(self.branches, X)
        n_miss = n_total[np.newaxis, :] - n_match

        # Depth-adjusted per-condition probabilities
        n_total_safe = np.where(n_total > 0, n_total, 1.0)  # [n_branches]
        p_high_adj = self.p_high ** (1.0 / n_total_safe)
        p_low_adj  = self.p_low  ** (1.0 / n_total_safe)

        log_lik_true  = (n_match * np.log(p_high_adj)[np.newaxis, :]
                         + n_miss * np.log(1.0 - p_high_adj)[np.newaxis, :])
        log_lik_false = (n_match * np.log(p_low_adj)[np.newaxis, :]
                         + n_miss * np.log(1.0 - p_low_adj)[np.newaxis, :])

        pz = np.clip(probs, 1e-15, 1.0 - 1e-15)
        log_jt = np.log(pz) + log_lik_true
        log_jf = np.log(1.0 - pz) + log_lik_false
        max_log = np.maximum(log_jt, log_jf)
        log_ev = max_log + np.log(np.exp(log_jt - max_log) + np.exp(log_jf - max_log))

        return np.exp(log_jt - log_ev)

    def predict_with_explanations(
        self,
        branch_probs: np.ndarray,
        X: np.ndarray,
        top_k_branches: int = 5,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Predict with per-sample explanations.

        Returns the top-k branches supporting the predicted class,
        using posterior z (if mode=="full") or prior z (if mode=="fast").
        """
        proba = self.predict_proba(branch_probs, X, verbose=False)
        predictions = np.argmax(proba, axis=1)

        # Posterior z for explanations
        if self.mode in ("full", "full_problog"):
            z_for_explain = self.get_posterior_z(branch_probs, X)
        else:
            z_for_explain = np.asarray(branch_probs)
            if z_for_explain.ndim == 1:
                z_for_explain = z_for_explain[np.newaxis, :]

        explanations = []
        for i in range(len(X)):
            pred_class = int(predictions[i])
            branch_support = []
            for br_idx, branch in enumerate(self.branches):
                theta = _class_proportions_to_theta(branch)
                if theta and pred_class < len(theta):
                    pz_val = float(z_for_explain[i, br_idx])
                    support_score = theta[pred_class] * pz_val
                    branch_support.append({
                        'branch_id': branch.branch_id,
                        'tree_id': branch.tree_id,
                        'theta_k': theta[pred_class],
                        'p_z_posterior': pz_val,
                        'support_score': support_score,
                        'conditions': [
                            f"f{c.feature_idx} {c.direction} {c.threshold:.4f}"
                            for c in branch.conditions
                        ],
                    })
            branch_support.sort(key=lambda x: x['support_score'], reverse=True)
            explanations.append({
                'predicted_class': pred_class,
                'class_probabilities': proba[i].tolist(),
                'top_branches': branch_support[:top_k_branches],
            })

        return predictions, explanations

    # ── Private: ProbLog engine (slow) ─────────────────────

    def _predict_proba_problog(
        self,
        branch_probs: np.ndarray,
        X: np.ndarray,
        verbose: bool,
        top_k_branches: int,
    ) -> np.ndarray:
        """Per-sample ProbLog inference (slow, for verification)."""
        if X is None:
            raise ValueError("X is required for mode='full_problog'")

        n_samples = X.shape[0]
        all_probs = np.zeros((n_samples, self.n_classes))

        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator, desc="ProbLog inference (full_problog)")

        for i in iterator:
            bp_i = branch_probs[i]
            branches_i = self.branches
            if top_k_branches is not None and top_k_branches < len(self.branches):
                top_idx = np.argsort(bp_i)[-top_k_branches:]
                branches_i = [self.branches[j] for j in top_idx]
                bp_i = bp_i[top_idx]

            program = export_full_problog_program(
                branches=branches_i,
                branch_probs_single=bp_i,
                observed_row=X[i],
                x_id=i,
                n_classes=self.n_classes,
                p_high=self.p_high,
                p_low=self.p_low,
                min_theta=self.min_theta,
            )
            all_probs[i] = _run_problog_inference(program, self.n_classes, i)

        return all_probs


# ═════════════════════════════════════════════════════════════
# Modular building blocks for experimental aggregation variants
# ═════════════════════════════════════════════════════════════

def build_theta_matrix(
    branches: List[Branch],
    n_classes: int,
    min_theta: float = 1e-6,
    temperature: float = 1.0,
) -> np.ndarray:
    """Build theta matrix [n_branches, n_classes] with optional temperature sharpening.

    Parameters
    ----------
    temperature : float
        T < 1 sharpens (makes theta more peaked).
        T = 1 leaves as-is.
        T > 1 smooths (makes theta more uniform).
        Applied as: theta_sharp = theta^(1/T) / sum(theta^(1/T)) per row.
    """
    n_branches = len(branches)
    theta = np.zeros((n_branches, n_classes), dtype=np.float64)
    for br_idx, branch in enumerate(branches):
        t = _class_proportions_to_theta(branch)
        if not t:
            continue
        for k in range(min(n_classes, len(t))):
            theta[br_idx, k] = max(min(t[k], 1.0 - min_theta), min_theta)

    if temperature != 1.0 and temperature > 0:
        theta_sharp = theta ** (1.0 / temperature)
        row_sums = theta_sharp.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        theta = theta_sharp / row_sums

    return theta


def aggregate_noisy_or(
    z: np.ndarray,
    theta: np.ndarray,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Noisy-or aggregation: P(class=k) = 1 - Π_b (1 - θ_bk · z_b).

    Parameters
    ----------
    z : [n_samples, n_branches] — branch activation values (prior or posterior)
    theta : [n_branches, n_classes]
    top_k : optional int — keep only top-K branches per sample (by z value)

    Returns
    -------
    [n_samples, n_classes] — normalized class probabilities
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]

    if top_k is not None and top_k < z.shape[1]:
        z_filtered = np.zeros_like(z)
        for i in range(z.shape[0]):
            idx = np.argsort(z[i])[-top_k:]
            z_filtered[i, idx] = z[i, idx]
        z = z_filtered

    p_support = z[:, :, np.newaxis] * theta[np.newaxis, :, :]
    log_complement = np.log1p(-np.clip(p_support, 0, 1 - 1e-15))
    log_noisy_or = np.sum(log_complement, axis=1)
    class_probs = 1.0 - np.exp(log_noisy_or)

    row_sums = class_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return class_probs / row_sums


def aggregate_weighted_mean(
    z: np.ndarray,
    theta: np.ndarray,
    top_k: Optional[int] = None,
    conflict_penalty: float = 0.0,
) -> np.ndarray:
    """Weighted-mean aggregation: P(class=k) ∝ Σ_b θ_bk · z_b.

    Unlike noisy-or, this does not saturate with many branches.

    Parameters
    ----------
    z : [n_samples, n_branches]
    theta : [n_branches, n_classes]
    top_k : optional int — keep only top-K branches per sample

    Returns
    -------
    [n_samples, n_classes] — normalized class probabilities
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]

    if top_k is not None and top_k < z.shape[1]:
        z_filtered = np.zeros_like(z)
        for i in range(z.shape[0]):
            idx = np.argsort(z[i])[-top_k:]
            z_filtered[i, idx] = z[i, idx]
        z = z_filtered

    # [n, B, 1] * [1, B, K] → [n, B, K] → sum over B → [n, K]
    weighted = z[:, :, np.newaxis] * theta[np.newaxis, :, :]
    numerator = weighted.sum(axis=1)

    row_sums = numerator.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    proba = numerator / row_sums
    if conflict_penalty > 0.0 and proba.shape[1] > 1:
        conflict = 1.0 - np.max(proba, axis=1, keepdims=True)
        uniform = np.ones_like(proba) / proba.shape[1]
        shrink = np.clip(float(conflict_penalty) * conflict, 0.0, 1.0)
        proba = (1.0 - shrink) * proba + shrink * uniform
    return proba


def compute_match_ratio(
    branches: List[Branch],
    X: np.ndarray,
) -> np.ndarray:
    """Compute condition match ratio per (sample, branch).

    Returns
    -------
    [n_samples, n_branches] — values in [0, 1], fraction of conditions satisfied.
    Branches with 0 conditions get ratio = 1.0.
    """
    n_match, n_total = _evaluate_conditions_batch(branches, X)
    n_total_safe = np.where(n_total > 0, n_total, 1.0)[np.newaxis, :]
    ratio = n_match / n_total_safe
    ratio[:, n_total == 0] = 1.0
    return ratio


def compute_condition_activation(
    branches: List[Branch],
    X: np.ndarray,
    tau: float = 1.0,
    soft_and: str = "geomean",
) -> np.ndarray:
    """Compute condition-aware branch activations from explicit rule atoms.

    Mirrors ``nstoolkit.engine.inference.compute_condition_activation``.
    ``geomean`` keeps shallow and deep rules comparable; ``product`` is a
    stricter differentiable AND; ``mean`` is a softer coverage score.
    Branches with no conditions are always active.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    out = np.ones((n_samples, len(branches)), dtype=np.float64)

    def _sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -50.0, 50.0)))

    tau = max(float(tau), 1e-12)
    for b_idx, branch in enumerate(branches):
        if not branch.conditions:
            out[:, b_idx] = 1.0
            continue
        scores = []
        for cond in branch.conditions:
            vals = X[:, cond.feature_idx]
            if cond.direction == "le":
                scores.append(_sigmoid((cond.threshold - vals) / tau))
            else:
                scores.append(_sigmoid((vals - cond.threshold) / tau))
        score = np.stack(scores, axis=1)
        if soft_and == "product":
            out[:, b_idx] = np.prod(score, axis=1)
        elif soft_and == "mean":
            out[:, b_idx] = score.mean(axis=1)
        elif soft_and == "geomean":
            out[:, b_idx] = np.exp(
                np.log(np.clip(score, 1e-12, 1.0)).mean(axis=1)
            )
        else:
            raise ValueError(f"Unsupported soft_and: {soft_and}")
    return np.clip(out, 0.0, 1.0)


def combine_rule_activations(
    neural_z: np.ndarray,
    condition_z: np.ndarray,
    mode: str = "hybrid",
    lam: float = 0.5,
) -> np.ndarray:
    """Combine neural and condition-aware rule activations.

    ``mode='neural'`` preserves the learned hidden activations,
    ``mode='condition'`` uses explicit soft rule satisfaction, and
    ``mode='hybrid'`` geometrically combines both, exactly matching the
    public NSToolkit rule-head behavior.
    """
    neural_z = np.asarray(neural_z, dtype=np.float64)
    condition_z = np.asarray(condition_z, dtype=np.float64)
    if mode == "neural":
        return neural_z
    if mode == "condition":
        return condition_z
    if mode == "hybrid":
        lam = float(np.clip(lam, 0.0, 1.0))
        return (
            np.clip(neural_z, 1e-12, 1.0) ** lam
            * np.clip(condition_z, 1e-12, 1.0) ** (1.0 - lam)
        )
    raise ValueError(f"Unsupported activation mode: {mode}")


def estimate_rule_reliability(
    branches: List[Branch],
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    theta: Optional[np.ndarray] = None,
    min_coverage: float = 1e-3,
) -> np.ndarray:
    """Estimate per-rule reliability from validation coverage and precision."""
    X = np.asarray(X)
    y = np.asarray(y).ravel().astype(int)
    if theta is None:
        theta = build_theta_matrix(branches, n_classes)
    theta = np.asarray(theta, dtype=np.float64)
    theta_safe = np.clip(theta, 1e-12, 1.0)
    theta_safe = theta_safe / np.maximum(theta_safe.sum(axis=1, keepdims=True), 1e-12)

    entropy = -(theta_safe * np.log(theta_safe)).sum(axis=1)
    purity = 1.0 - entropy / max(np.log(max(n_classes, 2)), 1e-12)
    branch_pred = theta_safe.argmax(axis=1)
    hard_match = compute_match_ratio(branches, X) >= 1.0
    reliability = np.zeros(len(branches), dtype=np.float64)

    for b_idx in range(len(branches)):
        mask = hard_match[:, b_idx]
        coverage = float(mask.mean())
        precision = (
            float(np.mean(y[mask] == branch_pred[b_idx]))
            if mask.any()
            else float(purity[b_idx])
        )
        coverage_gate = coverage / (coverage + min_coverage)
        reliability[b_idx] = coverage_gate * (0.5 * precision + 0.5 * purity[b_idx])

    return np.clip(reliability, 0.0, 1.0)


def apply_rule_reliability(
    z: np.ndarray,
    reliability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multiply branch activations by per-rule reliability weights."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    if reliability is None:
        return z
    r = np.asarray(reliability, dtype=np.float64).reshape(1, -1)
    return z * np.clip(r, 0.0, 1.0)


def compute_rule_conflict(
    z: np.ndarray,
    theta: np.ndarray,
    reliability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score disagreement among active rules for each sample."""
    z_eff = apply_rule_reliability(z, reliability)
    theta = np.asarray(theta, dtype=np.float64)
    support = z_eff @ theta
    if support.shape[1] <= 1:
        return np.zeros(support.shape[0], dtype=np.float64)
    total = support.sum(axis=1)
    top2 = np.partition(support, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return np.clip(1.0 - margin / np.maximum(total, 1e-12), 0.0, 1.0)


def compute_rule_uncertainty(
    proba: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    reliability: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute uncertainty/conflict diagnostics for rule-head predictions."""
    p = np.asarray(proba, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    entropy = entropy / max(np.log(max(p.shape[1], 2)), 1e-12)
    if p.shape[1] > 1:
        top2 = np.partition(p, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = np.ones(p.shape[0], dtype=np.float64)
    conflict = compute_rule_conflict(z, theta, reliability)
    z_eff = apply_rule_reliability(z, reliability)
    active_mass = z_eff.sum(axis=1)
    low_support = 1.0 / (1.0 + active_mass)
    uncertainty = np.clip(
        0.45 * entropy + 0.35 * conflict + 0.20 * low_support,
        0.0,
        1.0,
    )
    return {
        "entropy": entropy,
        "margin": margin,
        "conflict": conflict,
        "active_mass": active_mass,
        "low_support": low_support,
        "uncertainty": uncertainty,
    }


def find_optimal_alpha(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y_true: np.ndarray,
    grid: Optional[np.ndarray] = None,
    metric: str = "accuracy",
) -> Tuple[float, float]:
    """Find optimal mixing weight α for ensemble = α·A + (1-α)·B.

    Parameters
    ----------
    proba_a, proba_b : [n_samples, n_classes]
    y_true : [n_samples]
    grid : array of α values to try (default: 0.0, 0.05, ..., 1.0)
    metric : "accuracy" or "log_loss"

    Returns
    -------
    (best_alpha, best_score)
    """
    if grid is None:
        grid = np.arange(0.0, 1.01, 0.05)

    best_alpha, best_score = 0.5, -1e9
    for alpha in grid:
        proba = alpha * proba_a + (1 - alpha) * proba_b
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        proba = proba / row_sums

        if metric == "accuracy":
            score = float(np.mean(np.argmax(proba, axis=1) == y_true))
        elif metric == "log_loss":
            from sklearn.metrics import log_loss as _ll
            pc = np.clip(proba, 1e-15, 1 - 1e-15)
            pc = pc / pc.sum(axis=1, keepdims=True)
            score = -_ll(y_true, pc)  # negative because we maximise
        else:
            score = float(np.mean(np.argmax(proba, axis=1) == y_true))

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score


def find_optimal_3way(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    proba_c: np.ndarray,
    y_true: np.ndarray,
    step: float = 0.1,
) -> Tuple[Tuple[float, float, float], float]:
    """Find optimal weights (w1, w2, w3) for 3-way ensemble = w1·A + w2·B + w3·C.

    Grid search over simplex with given step. Returns (best_weights, best_accuracy).
    """
    best_w, best_score = (1/3, 1/3, 1/3), -1.0
    vals = np.arange(0.0, 1.0 + step/2, step)

    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            w3 = max(w3, 0.0)
            proba = w1 * proba_a + w2 * proba_b + w3 * proba_c
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            proba = proba / row_sums
            score = float(np.mean(np.argmax(proba, axis=1) == y_true))
            if score > best_score:
                best_score = score
                best_w = (round(w1, 2), round(w2, 2), round(w3, 2))

    return best_w, best_score


# ═════════════════════════════════════════════════════════════
# Improvement building blocks: Top-K filter, Temperature, Learned α
# ═════════════════════════════════════════════════════════════

def apply_topk_filter(z: np.ndarray, K: int) -> np.ndarray:
    """Zero out all but top-K branches per sample (by activation value).

    Unlike the ``top_k`` parameter in ``aggregate_*`` functions, this returns
    the filtered z directly so it can be fed into ``learn_branch_weights``.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    if K >= z.shape[1]:
        return z.copy()
    z_f = np.zeros_like(z)
    idx = np.argpartition(-z, K, axis=1)[:, :K]
    np.put_along_axis(z_f, idx, np.take_along_axis(z, idx, axis=1), axis=1)
    return z_f


def learn_temperature(proba_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Learn temperature *T* that minimises NLL on calibration data.

    Temperature scaling: ``P_cal(k) = softmax(log P(k) / T)``.
    T > 1 → smoother (less confident),  T < 1 → sharper,  T = 1 → unchanged.
    """
    from scipy.optimize import minimize_scalar

    log_p = np.log(np.clip(proba_cal, 1e-15, 1.0))
    y = y_cal.ravel().astype(int)

    def nll(T):
        s = log_p / max(T, 0.01)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-15))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling: ``softmax(log P / T)``."""
    log_p = np.log(np.clip(proba, 1e-15, 1.0))
    s = log_p / max(T, 0.01)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def learn_branch_weights(
    z_train: np.ndarray,
    theta: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 300,
    lr: float = 0.01,
) -> np.ndarray:
    """Learn per-branch attention weights α via gradient descent on CE loss.

    Optimises:  ``P(k|X) = Σ(α_b · z_b · θ_{b,k}) / Σ(α_b · z_b)``
    where α_b ≥ 0 are learnable weights (one per branch).

    Returns
    -------
    alpha : [n_branches] — positive weights (softplus of learned logits)
    """
    import torch
    import torch.nn.functional as F_

    n_br = z_train.shape[1]
    z_t = torch.tensor(z_train, dtype=torch.float32)
    th_t = torch.tensor(theta, dtype=torch.float32)
    y_t = torch.tensor(y_train.ravel(), dtype=torch.long)

    log_a = torch.nn.Parameter(torch.zeros(n_br))
    opt = torch.optim.Adam([log_a], lr=lr)

    for _ in range(epochs):
        a = F_.softplus(log_a)
        wz = z_t * a.unsqueeze(0)
        num = wz @ th_t
        den = wz.sum(dim=1, keepdim=True) + 1e-15
        p = num / den
        loss = F_.nll_loss(torch.log(p + 1e-15), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return F_.softplus(log_a).detach().numpy()


def extract_w2_as_theta(model) -> np.ndarray:
    """Extract W2 from a trained RuleNetwork and normalise it as a valid θ matrix.

    W2 shape = [n_classes, n_branches].  θ shape = [n_branches, n_classes].
    W2 already encodes class proportions per branch (from tree leaves), so
    transposing and row-normalising gives a θ that is *perfectly aligned*
    with the W1 that was trained to optimise through W2.

    Returns
    -------
    theta : [n_branches, n_classes] — row-normalised, non-negative
    """
    w2 = model.w2.detach().cpu().numpy()          # [K, B]
    theta = w2.T.copy()                            # [B, K]
    theta = np.clip(theta, 0.0, None)              # ensure non-negative
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return theta / row_sums


def learn_theta_alpha(
    z_train: np.ndarray,
    theta_init: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 500,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Learn θ (branch→class mapping) AND α (branch weights) end-to-end.

    Instead of using the fixed θ from tree proportions, we jointly optimise
    both θ and α to minimise cross-entropy loss on the training set.

    θ is parameterised via softmax over logits (ensures valid probability rows).
    α is parameterised via softplus (ensures positivity).

    This addresses the core train-test mismatch: W1 was trained for W2, but
    ProbLog inference uses θ — by learning θ, we align the ProbLog head with W1.

    Returns
    -------
    theta_learned : [n_branches, n_classes] — learned class proportions
    alpha_learned : [n_branches] — learned positive branch weights
    """
    import torch
    import torch.nn.functional as F_

    n_br = z_train.shape[1]
    z_t = torch.tensor(z_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.ravel(), dtype=torch.long)

    # Initialize θ-logits from tree proportions (log-space)
    theta_clamped = np.clip(theta_init, 1e-6, 1.0)
    theta_logits = torch.nn.Parameter(
        torch.tensor(np.log(theta_clamped), dtype=torch.float32)
    )
    log_a = torch.nn.Parameter(torch.zeros(n_br))

    opt = torch.optim.Adam([theta_logits, log_a], lr=lr)

    for _ in range(epochs):
        theta_soft = F_.softmax(theta_logits, dim=1)   # [B, K]
        a = F_.softplus(log_a)                          # [B]

        wz = z_t * a.unsqueeze(0)                       # [N, B]
        num = wz @ theta_soft                            # [N, K]
        den = wz.sum(dim=1, keepdim=True) + 1e-15
        p = num / den

        loss = F_.nll_loss(torch.log(p + 1e-15), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    theta_out = F_.softmax(theta_logits, dim=1).detach().numpy()
    alpha_out = F_.softplus(log_a).detach().numpy()
    return theta_out, alpha_out


def aggregate_weighted_mean_alpha(
    z: np.ndarray,
    theta: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Weighted mean with per-branch learned weights α.

    ``P(k|X) = Σ(α_b · z_b · θ_{b,k}) / Σ(α_b · z_b)``
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    wz = z * alpha[np.newaxis, :]
    num = wz @ theta
    rs = num.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    return num / rs


# ═════════════════════════════════════════════════════════════
# New aggregation modes for PL-fast / PL-full modernisation
# ═════════════════════════════════════════════════════════════

def aggregate_max_product(
    z: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Max-product (MAP) aggregation: P(class=k) ∝ max_b(θ_bk · z_b).

    Instead of noisy-or (which saturates with many branches), each class
    probability is determined by the *single most confident branch*.
    This corresponds to MAP inference in factor graphs / Viterbi decoding.

    Parameters
    ----------
    z : [n_samples, n_branches] — branch activations (prior or posterior)
    theta : [n_branches, n_classes]

    Returns
    -------
    [n_samples, n_classes] — normalised class probabilities
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]

    # [n, B, 1] * [1, B, K] → [n, B, K]  →  max over B  → [n, K]
    support = z[:, :, np.newaxis] * theta[np.newaxis, :, :]
    class_scores = support.max(axis=1)  # max over branches

    row_sums = class_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return class_scores / row_sums


def aggregate_geometric_mean(
    z: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Geometric-mean (log-linear) aggregation.

    P(class=k) ∝ exp( Σ_b z_b · log θ_bk )

    Each branch votes for a class via log(θ_bk), weighted by its activation
    z_b.  Unlike noisy-or, contributions are additive in log-space, so there
    is **no saturation** regardless of the number of branches.

    This corresponds to a log-linear / energy-based model where branches
    define potentials φ_bk = θ_bk^(z_b).

    Parameters
    ----------
    z : [n_samples, n_branches] — branch activations (prior or posterior)
    theta : [n_branches, n_classes]
    eps : float — floor for theta to avoid log(0)

    Returns
    -------
    [n_samples, n_classes] — normalised class probabilities
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]

    log_theta = np.log(np.clip(theta, eps, 1.0))  # [B, K]

    # Σ_b z_b · log θ_bk  →  z @ log_theta  →  [n, K]
    log_scores = z @ log_theta

    # Softmax for numerical stability
    log_scores = log_scores - log_scores.max(axis=1, keepdims=True)
    scores = np.exp(log_scores)

    row_sums = scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return scores / row_sums


def compute_entropy_gate(
    theta: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute per-branch entropy of θ distribution and return a gate mask.

    Branches with near-uniform θ (high entropy) carry no class information
    and should be suppressed.  The gate is:
        gate_b = 1 − H(θ_b) / H_max

    where H_max = log(n_classes) is the maximum entropy (uniform).
    gate = 1 for a perfectly peaked branch, gate → 0 for uniform.

    Parameters
    ----------
    theta : [n_branches, n_classes]
    n_classes : int

    Returns
    -------
    gate : [n_branches] — values in [0, 1]
    """
    eps = 1e-15
    theta_safe = np.clip(theta, eps, 1.0)
    # Normalise rows (should already sum to ~1, but just in case)
    row_sums = theta_safe.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    p = theta_safe / row_sums

    H = -np.sum(p * np.log(p + eps), axis=1)  # [n_branches]
    H_max = np.log(n_classes) + eps
    gate = 1.0 - H / H_max
    return np.clip(gate, 0.0, 1.0)


# ═════════════════════════════════════════════════════════════
# Differentiable Bayesian Posterior for end-to-end training
# ═════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def aggregate_calibrated_noisy_or(
    z: np.ndarray,
    theta: np.ndarray,
    leak: Optional[np.ndarray] = None,
    class_bias: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    branch_gate: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Noisy-or with class leak, class bias, temperature and branch gates.

    This is the NumPy inference counterpart of the calibrated e2e-NoisyOr
    training path.  The raw noisy-or class evidences are converted to logits
    with ``log(p_class) + class_bias`` and normalised by a softmax temperature.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    theta = np.asarray(theta, dtype=np.float64)
    if branch_gate is not None:
        gate = np.asarray(branch_gate, dtype=np.float64).reshape(1, -1)
        z = z * np.clip(gate, 0.0, 1.0)

    n_classes = theta.shape[1]
    leak_arr = (
        np.zeros(n_classes, dtype=np.float64)
        if leak is None
        else np.asarray(leak, dtype=np.float64).reshape(-1)
    )
    bias_arr = (
        np.zeros(n_classes, dtype=np.float64)
        if class_bias is None
        else np.asarray(class_bias, dtype=np.float64).reshape(-1)
    )
    if leak_arr.shape[0] != n_classes:
        raise ValueError("leak must have one value per class")
    if bias_arr.shape[0] != n_classes:
        raise ValueError("class_bias must have one value per class")

    p_support = z[:, :, np.newaxis] * theta[np.newaxis, :, :]
    log_no_event = np.log1p(-np.clip(p_support, 0.0, 1.0 - 1e-15)).sum(axis=1)
    log_no_event += np.log1p(-np.clip(leak_arr, 0.0, 1.0 - 1e-15))[np.newaxis, :]
    class_prob = np.clip(1.0 - np.exp(log_no_event), 1e-15, 1.0)

    temp = max(float(temperature), 1e-6)
    logits = (np.log(class_prob) + bias_arr[np.newaxis, :]) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(logits)
    return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-15)


class DifferentiablePosterior(nn.Module):
    """Differentiable Bayesian posterior P(z|evidence) in PyTorch.

    Converts the analytical posterior computation into a fully
    differentiable PyTorch operation so that gradients can flow
    from the loss through the posterior back to W1.

    Condition evaluation uses soft sigmoid matching:
        match = σ((threshold - x) / τ)   for "le" conditions
        match = σ((x - threshold) / τ)   for "gt" conditions

    This makes the evidence evaluation differentiable while preserving
    the same Bayesian structure as the NumPy analytical posterior.
    """

    def __init__(
        self,
        branches: List[Branch],
        p_high: float = 0.95,
        p_low: float = 0.05,
        tau: float = 0.1,
        learn_reliability: bool = False,
        reliability_init: float = 1.0,
        reliability_max: float = 2.0,
    ):
        super().__init__()
        self.branches = branches
        self.p_high = p_high
        self.p_low = p_low
        self.tau = tau
        self.learn_reliability = bool(learn_reliability)
        self.reliability_max = float(max(reliability_max, 1e-6))

        # Pre-extract all conditions into flat tensors for vectorization
        all_feat_idx = []
        all_threshold = []
        all_direction = []  # +1 for "le", -1 for "gt"
        all_branch_idx = []
        branch_n_conds = []

        for b_idx, branch in enumerate(branches):
            n_conds = len(branch.conditions)
            branch_n_conds.append(n_conds)
            for cond in branch.conditions:
                all_feat_idx.append(cond.feature_idx)
                all_threshold.append(cond.threshold)
                all_direction.append(1.0 if cond.direction == "le" else -1.0)
                all_branch_idx.append(b_idx)

        self.n_branches = len(branches)
        self.n_total_conds = len(all_feat_idx)

        self._has_conditions = self.n_total_conds > 0

        self.register_buffer("feat_idx", torch.tensor(all_feat_idx, dtype=torch.long))
        self.register_buffer("threshold", torch.tensor(all_threshold, dtype=torch.float32))
        self.register_buffer("direction", torch.tensor(all_direction, dtype=torch.float32))
        self.register_buffer("branch_idx", torch.tensor(all_branch_idx, dtype=torch.long))
        self.register_buffer(
            "branch_n_conds",
            torch.tensor(branch_n_conds, dtype=torch.float32),
        )

        # Depth-adjusted per-condition log-probabilities
        # For condition c belonging to branch b with m_b conditions:
        #   p_h = p_high^(1/m_b),   p_l = p_low^(1/m_b)
        safe_branch_n_conds = [max(n, 1) for n in branch_n_conds]
        p_h_per_cond = torch.tensor(
            [p_high ** (1.0 / safe_branch_n_conds[bi]) for bi in all_branch_idx],
            dtype=torch.float32,
        )
        p_l_per_cond = torch.tensor(
            [p_low ** (1.0 / safe_branch_n_conds[bi]) for bi in all_branch_idx],
            dtype=torch.float32,
        )

        # Pre-compute log constants (these are fixed, no gradient needed)
        self.register_buffer("log_ph", torch.log(p_h_per_cond))
        self.register_buffer("log_1m_ph", torch.log(1.0 - p_h_per_cond))
        self.register_buffer("log_pl", torch.log(p_l_per_cond))
        self.register_buffer("log_1m_pl", torch.log(1.0 - p_l_per_cond))

        if self.learn_reliability:
            init = float(np.clip(reliability_init / self.reliability_max, 1e-4, 1.0 - 1e-4))
            init_logit = math.log(init / (1.0 - init))
            self.evidence_reliability_logit = nn.Parameter(
                torch.full((self.n_branches,), init_logit, dtype=torch.float32)
            )
        else:
            self.register_parameter("evidence_reliability_logit", None)

    def evidence_reliability(self) -> torch.Tensor:
        """Return per-branch evidence scale r_b.

        ``r_b = 1`` recovers the fixed Bayesian posterior.  Values below one
        temper noisy evidence; values above one let reliable branches sharpen
        the posterior.  The scale is bounded for stable paper sweeps.
        """
        if self.evidence_reliability_logit is None:
            return torch.ones(self.n_branches, dtype=torch.float32, device=self.branch_n_conds.device)
        return self.reliability_max * torch.sigmoid(self.evidence_reliability_logit)

    def evidence_regularization(self) -> torch.Tensor:
        """Small prior that keeps learned evidence reliability near 1."""
        r = self.evidence_reliability()
        return (r - 1.0).pow(2).mean()

    def _condition_match(self, X: torch.Tensor) -> torch.Tensor:
        X_sel = X[:, self.feat_idx.to(X.device)]
        diff = self.direction.to(X.device).unsqueeze(0) * (
            self.threshold.to(X.device).unsqueeze(0) - X_sel
        )
        return torch.sigmoid(diff / self.tau)

    def branch_truth(
        self,
        X: torch.Tensor,
        soft_and: str = "geomean",
    ) -> torch.Tensor:
        """Differentiable branch-condition truth target for auxiliary losses."""
        out = X.new_ones((X.shape[0], self.n_branches))
        if not self._has_conditions:
            return out
        match = self._condition_match(X)
        branch_exp = self.branch_idx.to(X.device).unsqueeze(0).expand(X.shape[0], -1)
        has_conditions = self.branch_n_conds.to(X.device) > 0
        n_safe = self.branch_n_conds.to(X.device).clamp_min(1.0).unsqueeze(0)

        if soft_and == "product":
            log_match = torch.log(match.clamp_min(1e-12))
            acc = X.new_zeros((X.shape[0], self.n_branches))
            acc.scatter_add_(1, branch_exp, log_match)
            out[:, has_conditions] = torch.exp(acc[:, has_conditions])
        elif soft_and == "mean":
            acc = X.new_zeros((X.shape[0], self.n_branches))
            acc.scatter_add_(1, branch_exp, match)
            out[:, has_conditions] = (acc / n_safe)[:, has_conditions]
        elif soft_and == "geomean":
            log_match = torch.log(match.clamp_min(1e-12))
            acc = X.new_zeros((X.shape[0], self.n_branches))
            acc.scatter_add_(1, branch_exp, log_match)
            out[:, has_conditions] = torch.exp((acc / n_safe)[:, has_conditions])
        else:
            raise ValueError(f"Unsupported soft_and: {soft_and}")
        return out.clamp(0.0, 1.0)

    def forward(
        self,
        z_prior: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Compute differentiable posterior P(z|evidence).

        Parameters
        ----------
        z_prior : torch.Tensor [batch, n_branches]
            Prior branch probabilities from the neural network (requires_grad).
        X : torch.Tensor [batch, n_features]
            Input features.

        Returns
        -------
        z_posterior : torch.Tensor [batch, n_branches]
            Posterior branch probabilities (gradients flow through).
        """
        return self.forward_with_attribution(z_prior, X, return_intermediates=False)

    def forward_with_attribution(
        self,
        z_prior: torch.Tensor,
        X: torch.Tensor,
        return_intermediates: bool = True,
    ):
        """Same forward as ``__call__`` but optionally returns per-condition
        intermediates used for attribution maps.

        When ``return_intermediates=True`` returns a tuple
        ``(z_posterior, info)`` where ``info`` is a dict with:

        * ``match`` [batch, n_total_conds] — soft truth value of every
          condition, σ((thr−x)/τ) for ``le`` and σ((x−thr)/τ) for ``gt``.
        * ``log_lik_z1_cond`` [batch, n_total_conds] — per-condition
          log-likelihood under z=1.
        * ``log_lik_z0_cond`` [batch, n_total_conds] — same under z=0.
        * ``cond_log_lr`` [batch, n_total_conds] — log-likelihood ratio
          contributed by every condition (log_lik_z1_cond − log_lik_z0_cond).
          The branch-level log-LR is exactly the scatter-sum of this.
        * ``log_lik_z1`` [batch, n_branches], ``log_lik_z0`` [batch, n_branches]
          — branch-level totals (matching the originals used internally).
        * ``log_evidence`` [batch, n_branches] — Bayesian normaliser.
        * ``branch_idx`` [n_total_conds] — branch each condition belongs to.

        These are the exact quantities that feed the Bayesian update, so
        decomposing them gives an exact attribution of the posterior shift
        back to its originating branch conditions.
        """
        if not self._has_conditions:
            if return_intermediates:
                return z_prior, {
                    "match": None, "log_lik_z1_cond": None, "log_lik_z0_cond": None,
                    "cond_log_lr": None, "log_lik_z1": None, "log_lik_z0": None,
                    "log_evidence": None, "branch_idx": self.branch_idx,
                    "evidence_reliability": self.evidence_reliability(),
                }
            return z_prior

        batch_size = z_prior.shape[0]

        # ── Soft condition evaluation (differentiable) ──────────
        match = self._condition_match(X)  # [batch, n_total_conds]

        # ── Per-condition log-likelihoods ───────────────────────
        # P(obs|z=1): match·p_h + (1-match)·(1-p_h)
        log_ph = self.log_ph.to(X.device).unsqueeze(0)
        log_1m_ph = self.log_1m_ph.to(X.device).unsqueeze(0)
        log_pl = self.log_pl.to(X.device).unsqueeze(0)
        log_1m_pl = self.log_1m_pl.to(X.device).unsqueeze(0)
        log_lik_z1_cond = (
            match * log_ph
            + (1 - match) * log_1m_ph
        )  # [batch, n_total_conds]

        log_lik_z0_cond = (
            match * log_pl
            + (1 - match) * log_1m_pl
        )  # [batch, n_total_conds]

        r_branch = self.evidence_reliability().to(X.device)
        r_cond = r_branch[self.branch_idx.to(X.device)].unsqueeze(0)
        log_lik_z1_cond = log_lik_z1_cond * r_cond
        log_lik_z0_cond = log_lik_z0_cond * r_cond

        # ── Scatter-add to branches ─────────────────────────────
        log_lik_z1 = z_prior.new_zeros(batch_size, self.n_branches)
        log_lik_z0 = z_prior.new_zeros(batch_size, self.n_branches)

        branch_exp = self.branch_idx.to(X.device).unsqueeze(0).expand(batch_size, -1)
        log_lik_z1.scatter_add_(1, branch_exp, log_lik_z1_cond)
        log_lik_z0.scatter_add_(1, branch_exp, log_lik_z0_cond)

        # ── Bayesian update ─────────────────────────────────────
        pz = torch.clamp(z_prior, 1e-7, 1.0 - 1e-7)

        log_joint_true = torch.log(pz) + log_lik_z1
        log_joint_false = torch.log(1.0 - pz) + log_lik_z0

        # Log-sum-exp for numerical stability
        max_log = torch.maximum(log_joint_true, log_joint_false)
        log_evidence = max_log + torch.log(
            torch.exp(log_joint_true - max_log)
            + torch.exp(log_joint_false - max_log)
        )

        z_posterior = torch.exp(log_joint_true - log_evidence)

        if not return_intermediates:
            return z_posterior

        info = {
            "match": match,
            "log_lik_z1_cond": log_lik_z1_cond,
            "log_lik_z0_cond": log_lik_z0_cond,
            "cond_log_lr": log_lik_z1_cond - log_lik_z0_cond,
            "log_lik_z1": log_lik_z1,
            "log_lik_z0": log_lik_z0,
            "log_evidence": log_evidence,
            "branch_idx": self.branch_idx,
            "evidence_reliability": r_branch,
        }
        return z_posterior, info
