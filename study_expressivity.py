"""Empirical study of the formal expressivity of ``DifferentiablePosterior``.

Companion file: ``EXPRESSIVITY.md`` (theoretical statements).

Four probes:

* **E1 — τ-consistency.**  As τ → 0 the soft sigmoid match converges to
  the indicator of the threshold, so the differentiable posterior should
  converge pointwise to the exact analytical posterior used by ProbLog.
  We fit RuleNetwork on Iris/Wine, then sweep τ and measure
  ``max_b |z_diff(τ) − z_analytical|``, ``RMSE``, and the symmetric KL
  on the noisy-or class probabilities.

* **E2 — XOR independence break.**  Construct a synthetic dataset with
  two branches whose joint pattern is XOR.  Bayes-optimal P(c|x) is the
  XOR of evidences, but a noisy-or class head with ``P(c=1|x) = 1 −
  Π_b (1 − θ_{b1}·z_b)`` is symmetric (and monotone in each z_b), so
  there exists *no* (θ_1, θ_2) pair that can represent XOR.  We fit θ
  on the synthetic data and report the irreducible error gap.

* **E3 — depth-adjusted likelihood ratio.**  For a single branch with
  m conditions, the per-condition LR contribution is
  ``Δ(match=1) = log(p_h^{1/m} / p_l^{1/m}) = (1/m) · log(p_h/p_l)``.
  We tabulate posterior concentration (logit shift when *all* conditions
  fire, prior=0.5) for m ∈ {1..6} and (p_high, p_low) ∈ {(.95,.05),
  (.99,.01), (.8,.2)}.

* **E4 — branch independence (theoretical).**  Compute, for every
  pair of branches in a trained model, the empirical
  conditional dependence ``corr(z_b1 | y, z_b2 | y) − corr(z_b1, z_b2)``;
  noisy-or assumes 0.  This is reported as a *limitation diagnostic*.

Run:

    python study_expressivity.py --output expressivity_report.json

Outputs a JSON report and a human-readable summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from rule_network_model import RuleNetworkModel
from problog_inference import (
    ProbLogClassifier,
    aggregate_noisy_or,
    build_theta_matrix,
    DifferentiablePosterior,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _train_rule_network(X, y, n_classes: int, n_features: int):
    et = ExtraTreesClassifier(
        n_estimators=int(n_classes + round(np.log2(max(n_features, 2)))),
        max_leaf_nodes=2 ** (round(np.log2(max(n_features, 2))) + 4),
        random_state=0,
    )
    et.fit(X, y)
    model = RuleNetworkModel()
    model.build_model_from_ensemble(et)
    theta = build_theta_matrix(model.branches, n_classes)
    return model, theta


def _safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))


def _sym_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return np.sum(p * np.log(p / q) + q * np.log(q / p), axis=-1)


# ──────────────────────────────────────────────────────────────────
# E1 — τ-consistency
# ──────────────────────────────────────────────────────────────────

def probe_tau_consistency(dataset_name: str, X, y, taus=None) -> Dict:
    """For a fixed trained model, sweep τ and compare diff. posterior
    against the exact analytical posterior used by native ProbLog."""
    if taus is None:
        taus = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

    n_classes = int(np.max(y) + 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    model, theta = _train_rule_network(X_tr, y_tr, n_classes, X.shape[1])
    bp_te = model.predict_branch_proba(X_te)
    if isinstance(bp_te, torch.Tensor):
        bp_te_np = bp_te.detach().cpu().numpy()
    else:
        bp_te_np = np.asarray(bp_te)

    clf_full = ProbLogClassifier(model.branches, n_classes, mode="full")
    z_post_analytical = clf_full.get_posterior_z(bp_te_np, X_te)
    proba_analytical = aggregate_noisy_or(z_post_analytical, theta)

    rows = []
    for tau in taus:
        diff = DifferentiablePosterior(
            model.branches, p_high=0.95, p_low=0.05, tau=tau
        )
        with torch.no_grad():
            z_post_diff = diff(
                bp_te if isinstance(bp_te, torch.Tensor)
                else torch.from_numpy(bp_te_np).float(),
                torch.from_numpy(X_te).float(),
            ).cpu().numpy()
        proba_diff = aggregate_noisy_or(z_post_diff, theta)

        z_max_abs = float(np.max(np.abs(z_post_diff - z_post_analytical)))
        z_rmse = float(np.sqrt(np.mean((z_post_diff - z_post_analytical) ** 2)))
        cls_kl = float(np.mean(_sym_kl(proba_diff, proba_analytical)))
        cls_max_abs = float(np.max(np.abs(proba_diff - proba_analytical)))
        argmax_agree = float(np.mean(
            np.argmax(proba_diff, axis=1) == np.argmax(proba_analytical, axis=1)
        ))
        rows.append({
            "tau": tau,
            "z_max_abs": z_max_abs,
            "z_rmse": z_rmse,
            "class_kl_sym_mean": cls_kl,
            "class_max_abs": cls_max_abs,
            "argmax_agreement": argmax_agree,
        })

    return {
        "dataset": dataset_name,
        "n_branches": len(model.branches),
        "n_test": int(len(X_te)),
        "rows": rows,
    }


# ──────────────────────────────────────────────────────────────────
# E2 — XOR independence break
# ──────────────────────────────────────────────────────────────────

def probe_xor_independence(n_per_quadrant: int = 200) -> Dict:
    """Two-feature XOR.  Bayes-optimal Boolean classifier is XOR(x1>0, x2>0).
    Noisy-or with two branches B1='x1>0', B2='x2>0' and any θ ∈ [0,1]^{2×2}
    is monotone in each z_b → cannot represent XOR.

    We compute the achievable accuracy (over a grid search of θ) and the
    irreducible decision-boundary error.
    """
    rng = np.random.RandomState(0)
    X_parts, y_parts = [], []
    for q1 in (-1, 1):
        for q2 in (-1, 1):
            mu = np.array([q1 * 1.5, q2 * 1.5])
            x = rng.randn(n_per_quadrant, 2) * 0.5 + mu
            label = int(((q1 > 0) != (q2 > 0)))  # XOR
            X_parts.append(x)
            y_parts.append(np.full(n_per_quadrant, label))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    # Hard branches (deterministic AND of single conditions)
    z1 = (X[:, 0] > 0).astype(np.float64)
    z2 = (X[:, 1] > 0).astype(np.float64)
    z = np.stack([z1, z2], axis=1)        # [n, 2]

    # Search θ ∈ [0,1]^{2 × 2}
    best_acc, best_theta = 0.0, None
    grid = np.linspace(0.0, 1.0, 21)
    for t11 in grid:
        for t12 in grid:
            for t21 in grid:
                for t22 in grid:
                    theta = np.array([[t11, t12], [t21, t22]])
                    proba = aggregate_noisy_or(z, theta)
                    pred = np.argmax(proba, axis=1)
                    acc = float(np.mean(pred == y))
                    if acc > best_acc:
                        best_acc, best_theta = acc, theta

    # Bayes-optimal achievable accuracy: 1 - misclassified XOR boundary.
    # With these means, optimal Bayes classifier ≈ 1.0 (well-separated).
    bayes_acc = 1.0
    return {
        "best_noisy_or_acc": best_acc,
        "bayes_optimal_acc": bayes_acc,
        "irreducible_gap": bayes_acc - best_acc,
        "best_theta": None if best_theta is None else best_theta.tolist(),
        "n": int(len(X)),
    }


# ──────────────────────────────────────────────────────────────────
# E3 — depth-adjusted likelihood ratio
# ──────────────────────────────────────────────────────────────────

def probe_depth_likelihood(
    depths=(1, 2, 3, 4, 5, 6),
    p_pairs=((0.95, 0.05), (0.99, 0.01), (0.8, 0.2)),
) -> Dict:
    """Per-branch logit shift when all m conditions fire (match=1) starting
    from prior z=0.5.  Expected: logit_shift = log(p_h/p_l), independent
    of depth, because depth-adjustment cancels (m · (1/m)·log(p_h/p_l)).
    Conversely, the *per-condition* contribution shrinks as 1/m — i.e. a
    deeper branch is less sensitive to losing any single condition match.
    """
    rows = []
    for p_h, p_l in p_pairs:
        for m in depths:
            p_h_per = p_h ** (1.0 / m)
            p_l_per = p_l ** (1.0 / m)
            per_cond_lr = np.log(p_h_per / p_l_per)
            full_branch_lr = m * per_cond_lr
            rows.append({
                "p_high": p_h,
                "p_low": p_l,
                "depth": m,
                "p_high_per_cond": float(p_h_per),
                "p_low_per_cond": float(p_l_per),
                "per_cond_log_lr": float(per_cond_lr),
                "branch_log_lr_all_fire": float(full_branch_lr),
                "z_post_all_fire_from_prior_0.5": float(
                    1.0 / (1.0 + np.exp(-full_branch_lr))
                ),
            })
    return {"rows": rows}


# ──────────────────────────────────────────────────────────────────
# E4 — branch independence diagnostic
# ──────────────────────────────────────────────────────────────────

def probe_branch_independence(dataset_name: str, X, y) -> Dict:
    """Empirical: how independent are branches (the assumption baked into
    noisy-or aggregation)?  Compute, on training data:
    *  joint = E[z_b1 · z_b2]
    *  prod  = E[z_b1] · E[z_b2]
    *  excess = joint − prod
    Aggregated to mean / max / fraction-above-0.05.
    """
    n_classes = int(np.max(y) + 1)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    model, _ = _train_rule_network(X_tr, y_tr, n_classes, X.shape[1])
    bp = model.predict_branch_proba(X_tr)
    z = bp.detach().cpu().numpy() if isinstance(bp, torch.Tensor) else np.asarray(bp)

    n_b = z.shape[1]
    means = z.mean(axis=0)
    joint = (z.T @ z) / z.shape[0]                    # [n_b, n_b]
    prod = np.outer(means, means)
    excess = joint - prod
    iu = np.triu_indices(n_b, k=1)
    excess_pairs = excess[iu]

    return {
        "dataset": dataset_name,
        "n_branches": n_b,
        "n_pairs": int(len(excess_pairs)),
        "mean_excess_dependence": float(np.mean(excess_pairs)),
        "max_abs_excess_dependence": float(np.max(np.abs(excess_pairs))),
        "frac_pairs_above_0.05": float(np.mean(np.abs(excess_pairs) > 0.05)),
    }


# ──────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────

def main(output_path: str):
    iris = load_iris()
    wine = load_wine()

    report = {
        "E1_tau_consistency": [
            probe_tau_consistency("Iris", iris.data, iris.target),
            probe_tau_consistency("Wine", wine.data, wine.target),
        ],
        "E2_xor_independence": probe_xor_independence(),
        "E3_depth_likelihood": probe_depth_likelihood(),
        "E4_branch_independence": [
            probe_branch_independence("Iris", iris.data, iris.target),
            probe_branch_independence("Wine", wine.data, wine.target),
        ],
    }

    # Pretty-print summary tables
    print("=" * 80)
    print("E1: τ-consistency (diff. posterior → analytical posterior)")
    print("=" * 80)
    for ds in report["E1_tau_consistency"]:
        print(f"\n  Dataset: {ds['dataset']}  (n_branches={ds['n_branches']}, n_test={ds['n_test']})")
        print(f"    {'τ':>8}  {'z_max_abs':>10}  {'z_rmse':>10}  "
              f"{'class_KL':>10}  {'p_max_abs':>10}  {'argmax_agree':>14}")
        for r in ds["rows"]:
            print(f"    {r['tau']:>8.3f}  {r['z_max_abs']:>10.4f}  "
                  f"{r['z_rmse']:>10.4f}  {r['class_kl_sym_mean']:>10.4f}  "
                  f"{r['class_max_abs']:>10.4f}  {r['argmax_agreement']:>14.3f}")

    print("\n" + "=" * 80)
    print("E2: XOR — noisy-or expressivity ceiling")
    print("=" * 80)
    e2 = report["E2_xor_independence"]
    print(f"  Bayes-optimal acc           : {e2['bayes_optimal_acc']:.4f}")
    print(f"  Best noisy-or acc (θ-grid)  : {e2['best_noisy_or_acc']:.4f}")
    print(f"  Irreducible gap (XOR is not noisy-or representable): "
          f"{e2['irreducible_gap']:.4f}")
    print(f"  Best θ found: {e2['best_theta']}")

    print("\n" + "=" * 80)
    print("E3: depth-adjusted likelihood ratio")
    print("=" * 80)
    print(f"  {'(p_h, p_l)':>14}  {'depth':>6}  {'per-cond LR':>12}  "
          f"{'branch LR':>10}  {'z_post(all fire)':>18}")
    for r in report["E3_depth_likelihood"]["rows"]:
        pp = f"({r['p_high']}, {r['p_low']})"
        print(f"  {pp:>14}  {r['depth']:>6}  {r['per_cond_log_lr']:>12.4f}  "
              f"{r['branch_log_lr_all_fire']:>10.4f}  "
              f"{r['z_post_all_fire_from_prior_0.5']:>18.6f}")

    print("\n" + "=" * 80)
    print("E4: empirical branch independence on z_prior")
    print("=" * 80)
    for ds in report["E4_branch_independence"]:
        print(f"  {ds['dataset']:<10}  n_branches={ds['n_branches']:<4}  "
              f"mean_excess={ds['mean_excess_dependence']:+.4f}  "
              f"max|excess|={ds['max_abs_excess_dependence']:.4f}  "
              f"frac>0.05={ds['frac_pairs_above_0.05']:.3f}")

    print()
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="expressivity_report.json")
    args = parser.parse_args()
    main(args.output)
