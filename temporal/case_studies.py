"""§6.2 — case studies: per-sample interpretability for L3 / L4.

For a held-out subset of a temporal dataset, this script:

1. Trains the requested level (L3 interval-forest backbone or L4
   PPθ-Post-Temporal) on the train split.
2. Selects up to ``--n-samples`` samples per class from the validation
   split.
3. Computes the predicted class, posterior support per branch, and the
   top-K branches by ``θ_bk · z_b`` (or ``θ_bk · max_t z_b,t`` for L4).
4. Renders each branch as a human-readable rule using
   :class:`IntervalFeatureExtractor.feature_meta` (L3) or per-timestep
   raw conditions (L4).
5. Dumps the result to JSON for inclusion in the paper / supplementary.

Example
-------
::

    python -m temporal.case_studies \
        --dataset p12 --level L3 --n-samples 4 --top-k 5

Output JSON schema
------------------
::

    {
      "dataset": "synthetic_p12",
      "level": "L3",
      "samples": [
        {
          "x_id": 12,
          "true_class": 1,
          "predicted_class": 1,
          "class_proba": [0.18, 0.82],
          "top_branches": [
            {
              "branch_id": "b142",
              "theta_k": 0.94,
              "p_z_posterior": 0.97,
              "support_score": 0.91,
              "rule": "mean(HR, [12:36]) > 95.4 AND slope(Lactate, [24:48]) > 0.3"
            },
            ...
          ]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from rule_network_model import RuleNetworkModel  # noqa: E402
from branch_schema import Branch  # noqa: E402
from problog_inference import (  # noqa: E402
    ProbLogClassifier,
    aggregate_weighted_mean,
    build_theta_matrix,
)
from problog_export import _class_proportions_to_theta  # noqa: E402

from .compare_temporal import _train_rule_network_static  # noqa: E402
from .datasets import load_temporal_dataset  # noqa: E402
from .interval_forest import (  # noqa: E402
    IntervalFeatureExtractor,
    IntervalFeatureMeta,
)
from .pp_theta_post_temporal import (  # noqa: E402
    PPThetaPostTemporal,
    temporal_aggregate,
)


# ─────────────────────────────────────────────────────────────────────────
# Rule rendering
# ─────────────────────────────────────────────────────────────────────────

def _render_l3_condition(
    branch: Branch,
    cond_idx: int,
    feature_meta: Sequence[IntervalFeatureMeta],
) -> str:
    cond = branch.conditions[cond_idx]
    meta = feature_meta[cond.feature_idx]
    op = "≤" if cond.direction == "le" else ">"
    return (
        f"{meta.stat}({meta.variable_name}, "
        f"[{meta.interval_start}:{meta.interval_end}]) {op} {cond.threshold:.4g}"
    )


def render_l3_rule(
    branch: Branch,
    feature_meta: Sequence[IntervalFeatureMeta],
) -> str:
    if not branch.conditions:
        return "TRUE"
    return " AND ".join(
        _render_l3_condition(branch, i, feature_meta)
        for i in range(len(branch.conditions))
    )


def _render_l4_condition(
    branch: Branch,
    cond_idx: int,
    var_names: Sequence[str],
) -> str:
    """For L4 the per-snapshot trees split on ``[value_v0..valueV-1,
    mask_v0..maskV-1]``; map ``feature_idx`` back to a variable name.
    """
    cond = branch.conditions[cond_idx]
    n_vars = len(var_names)
    if cond.feature_idx < n_vars:
        var_name = var_names[cond.feature_idx]
        kind = "value"
    else:
        var_name = var_names[cond.feature_idx - n_vars]
        kind = "mask"
    op = "≤" if cond.direction == "le" else ">"
    return f"{kind}({var_name}) {op} {cond.threshold:.4g}"


def render_l4_rule(
    branch: Branch,
    var_names: Sequence[str],
) -> str:
    if not branch.conditions:
        return "TRUE"
    return " AND ".join(
        _render_l4_condition(branch, i, var_names)
        for i in range(len(branch.conditions))
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-level case-study generation
# ─────────────────────────────────────────────────────────────────────────

def _normalise_k_values(k_values: Sequence[int]) -> Tuple[int, ...]:
    out = sorted({int(k) for k in k_values if int(k) > 0})
    return tuple(out or (1, 3, 5))


def _parse_int_list(raw: str, default: Sequence[int] = (1, 3, 5)) -> Tuple[int, ...]:
    if raw is None or str(raw).strip() == "":
        return _normalise_k_values(default)
    vals: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return _normalise_k_values(vals or default)


def _select_samples_per_class(
    y_val: np.ndarray, n_per_class: int, seed: int = 0,
) -> List[Tuple[int, str]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[int, str]] = []
    for cls in np.unique(y_val):
        idx = np.flatnonzero(y_val == cls)
        rng.shuffle(idx)
        out.extend((int(i), f"true_class_{int(cls)}") for i in idx[:n_per_class])
    return out


def _binary_predictions(proba: np.ndarray, decision_threshold: Optional[float]) -> np.ndarray:
    if decision_threshold is not None and proba.ndim == 2 and proba.shape[1] == 2:
        return (proba[:, 1] >= float(decision_threshold)).astype(int)
    return np.argmax(proba, axis=1)


def _select_case_samples(
    y_val: np.ndarray,
    pred: np.ndarray,
    n_per_bucket: int,
    seed: int = 0,
    mode: str = "balanced_true",
) -> List[Tuple[int, str]]:
    if mode == "balanced_true":
        return _select_samples_per_class(y_val, n_per_bucket, seed)
    if mode != "confusion":
        raise ValueError(f"unknown case selection mode: {mode}")

    rng = np.random.default_rng(seed)
    out: List[Tuple[int, str]] = []
    unique = set(int(x) for x in np.unique(y_val))
    if unique.issubset({0, 1}):
        specs = [
            ("tn", 0, 0),
            ("tp", 1, 1),
            ("fp", 0, 1),
            ("fn", 1, 0),
        ]
        for label, true_cls, pred_cls in specs:
            idx = np.flatnonzero((y_val == true_cls) & (pred == pred_cls))
            rng.shuffle(idx)
            out.extend((int(i), label) for i in idx[:n_per_bucket])
        return out

    for label, mask in (
        ("correct", y_val == pred),
        ("incorrect", y_val != pred),
    ):
        idx = np.flatnonzero(mask)
        rng.shuffle(idx)
        out.extend((int(i), label) for i in idx[:n_per_bucket])
    return out


def _branch_family(branch: Branch, max_conditions: int = 4) -> str:
    if not branch.conditions:
        return f"tree{int(branch.tree_id)}:TRUE"
    parts = [
        f"f{int(cond.feature_idx)}:{cond.direction}"
        for cond in branch.conditions[:max_conditions]
    ]
    if len(branch.conditions) > max_conditions:
        parts.append(f"len{len(branch.conditions)}")
    return "|".join(parts)


def _normalize_probability(scores: np.ndarray, fallback: np.ndarray) -> List[float]:
    scores = np.asarray(scores, dtype=np.float64)
    total = float(scores.sum())
    if np.isfinite(total) and total > 0:
        return (scores / total).tolist()
    return np.asarray(fallback, dtype=np.float64).tolist()


def _subset_weighted_mean_proba(
    z_row: np.ndarray,
    theta: np.ndarray,
    indices: Sequence[int],
    fallback: np.ndarray,
) -> List[float]:
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return np.asarray(fallback, dtype=np.float64).tolist()
    z_sub = np.clip(np.asarray(z_row, dtype=np.float64)[idx], 0.0, None)
    denom = float(z_sub.sum())
    if not np.isfinite(denom) or denom <= 0:
        return np.asarray(fallback, dtype=np.float64).tolist()
    scores = (z_sub[:, None] * theta[idx]).sum(axis=0) / denom
    return _normalize_probability(scores, fallback)


def _audit_counterfactual_fields(
    z_row: np.ndarray,
    theta: np.ndarray,
    pred_class: int,
    ranked_indices: np.ndarray,
    k_values: Sequence[int] = (1, 3, 5),
    branch_families: Optional[Sequence[str]] = None,
    random_seed: int = 0,
    random_baseline_samples: int = 20,
) -> Dict:
    """Fields required by Section 31 for coverage/sufficiency/deletion."""
    z_row = np.asarray(z_row, dtype=np.float64)
    support = z_row[:, None] * theta
    uniform = np.ones(theta.shape[1], dtype=np.float64) / theta.shape[1]
    raw_prior = theta.mean(axis=0) if theta.shape[0] else uniform
    prior = np.asarray(_normalize_probability(raw_prior, uniform), dtype=np.float64)
    all_indices = np.arange(theta.shape[0], dtype=np.int64)
    ranked_indices = np.asarray(ranked_indices, dtype=np.int64)
    k_values = _normalise_k_values(k_values)
    families = list(branch_families or [])
    if len(families) != theta.shape[0]:
        families = []

    out = {
        "total_support_mass": float(support[:, pred_class].sum()),
        "total_support_mass_by_class": support.sum(axis=0).tolist(),
        "proba_top_rules_only": {},
        "proba_without_top_rules": {},
        "proba_without_random_rules": {},
        "proba_without_rule_families": {},
        "n_random_rule_sets": int(max(0, random_baseline_samples)),
        "n_deleted_rule_families": {},
        "n_deleted_rule_family_branches": {},
    }
    for k in k_values:
        top = np.asarray(ranked_indices[: min(k, len(ranked_indices))], dtype=np.int64)
        keep = np.ones(theta.shape[0], dtype=bool)
        keep[top] = False
        out["proba_top_rules_only"][str(k)] = _subset_weighted_mean_proba(z_row, theta, top, prior)
        out["proba_without_top_rules"][str(k)] = _subset_weighted_mean_proba(
            z_row,
            theta,
            all_indices[keep],
            prior,
        )

        if theta.shape[0] > 0 and random_baseline_samples > 0:
            rng = np.random.default_rng(int(random_seed) + 10007 * int(k))
            remove_count = min(int(k), theta.shape[0])
            random_probas = []
            for _ in range(int(random_baseline_samples)):
                remove = rng.choice(all_indices, size=remove_count, replace=False)
                random_keep = np.ones(theta.shape[0], dtype=bool)
                random_keep[remove] = False
                random_probas.append(_subset_weighted_mean_proba(z_row, theta, all_indices[random_keep], prior))
            out["proba_without_random_rules"][str(k)] = np.asarray(random_probas, dtype=np.float64).mean(axis=0).tolist()

        if families:
            selected_families: List[str] = []
            seen = set()
            for idx in ranked_indices:
                fam = families[int(idx)]
                if fam not in seen:
                    selected_families.append(fam)
                    seen.add(fam)
                if len(selected_families) >= int(k):
                    break
            delete_family = np.asarray([fam in seen for fam in families], dtype=bool)
            out["proba_without_rule_families"][str(k)] = _subset_weighted_mean_proba(
                z_row,
                theta,
                all_indices[~delete_family],
                prior,
            )
            out["n_deleted_rule_families"][str(k)] = int(len(selected_families))
            out["n_deleted_rule_family_branches"][str(k)] = int(delete_family.sum())
    return out


def case_studies_l3(
    X_ts: np.ndarray, mask: np.ndarray, y: np.ndarray,
    var_names: Sequence[str], dataset_name: str,
    n_samples_per_class: int, top_k: int, seed: int, epochs: int,
    audit_k_values: Sequence[int] = (1, 3, 5),
    case_selection: str = "balanced_true",
    decision_threshold: Optional[float] = None,
    random_baseline_samples: int = 20,
) -> Dict:
    n_classes = int(np.unique(y).size)
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed,
    )
    extractor = IntervalFeatureExtractor(
        var_names=var_names, T=X_ts.shape[1], n_intervals=12, seed=seed,
    )
    X_train = extractor.transform(X_ts[train_idx], mask[train_idx])
    X_val = extractor.transform(X_ts[val_idx], mask[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    model = _train_rule_network_static(
        X_train, y_train, X_val, y_val, seed, epochs, n_classes,
    )
    bp_val = model.predict_branch_proba(X_val).numpy()
    branches = model.branches
    theta = build_theta_matrix(branches, n_classes)
    branch_rules = [render_l3_rule(br, extractor.feature_meta) for br in branches]
    branch_families = [_branch_family(br) for br in branches]
    proba = aggregate_weighted_mean(bp_val, theta)
    pred = _binary_predictions(proba, decision_threshold)
    print(f"L3 case-study val accuracy ({case_selection}): {accuracy_score(y_val, pred):.3f}")

    audit_k_values = _normalise_k_values(audit_k_values)
    export_top_k = max([int(top_k), *audit_k_values])
    chosen = _select_case_samples(y_val, pred, n_samples_per_class, seed, mode=case_selection)

    samples: List[Dict] = []
    for local_idx, case_type in chosen:
        global_id = int(val_idx[local_idx])
        z_row = bp_val[local_idx]                 # [B]
        support = z_row[:, None] * theta          # [B, K]
        pred_class = int(pred[local_idx])
        order = np.argsort(-support[:, pred_class])
        top = order[:export_top_k]
        top_branches = []
        for br_idx in top:
            br = branches[int(br_idx)]
            theta_k_list = _class_proportions_to_theta(br) or [0.0] * n_classes
            theta_k = float(theta_k_list[pred_class] if pred_class < len(theta_k_list) else 0.0)
            top_branches.append({
                "branch_id": br.branch_id,
                "tree_id": int(br.tree_id),
                "theta_k": theta_k,
                "p_z_posterior": float(z_row[int(br_idx)]),
                "support_score": float(support[int(br_idx), int(pred[local_idx])]),
                "rule": branch_rules[int(br_idx)],
                "rule_family": branch_families[int(br_idx)],
            })
        opposing_branches = []
        if n_classes > 1:
            opp_scores = support.copy()
            opp_scores[:, pred_class] = -np.inf
            opp_class = np.argmax(opp_scores, axis=1)
            opp_order = np.argsort(-opp_scores[np.arange(len(branches)), opp_class])[:top_k]
            for br_idx in opp_order:
                br = branches[int(br_idx)]
                cls = int(opp_class[int(br_idx)])
                theta_k_list = _class_proportions_to_theta(br) or [0.0] * n_classes
                theta_k = float(theta_k_list[cls] if cls < len(theta_k_list) else 0.0)
                opposing_branches.append({
                    "branch_id": br.branch_id,
                    "tree_id": int(br.tree_id),
                    "opposing_class": cls,
                    "theta_k": theta_k,
                    "p_z_posterior": float(z_row[int(br_idx)]),
                    "support_score": float(support[int(br_idx), cls]),
                    "rule": branch_rules[int(br_idx)],
                    "rule_family": branch_families[int(br_idx)],
                })
        audit_fields = _audit_counterfactual_fields(
            z_row,
            theta,
            pred_class,
            order,
            k_values=audit_k_values,
            branch_families=branch_families,
            random_seed=seed + int(global_id) * 1009,
            random_baseline_samples=random_baseline_samples,
        )
        samples.append({
            "x_id": global_id,
            "case_type": case_type,
            "decision_threshold": decision_threshold,
            "true_class": int(y_val[local_idx]),
            "predicted_class": pred_class,
            "class_proba": proba[local_idx].tolist(),
            "top_branches": top_branches,
            "opposing_branches": opposing_branches,
            **audit_fields,
        })

    return {
        "dataset": dataset_name,
        "level": "L3",
        "n_branches": len(branches),
        "case_selection": case_selection,
        "decision_threshold": decision_threshold,
        "audit_k_values": list(audit_k_values),
        "samples": samples,
    }


def case_studies_l4(
    X_ts: np.ndarray, mask: np.ndarray, y: np.ndarray,
    var_names: Sequence[str], dataset_name: str,
    n_samples_per_class: int, top_k: int, seed: int, epochs: int,
    aggregation: str = "mean",
    audit_k_values: Sequence[int] = (1, 3, 5),
    case_selection: str = "balanced_true",
    decision_threshold: Optional[float] = None,
    random_baseline_samples: int = 20,
) -> Dict:
    n_classes = int(np.unique(y).size)
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed,
    )

    tbn = PPThetaPostTemporal(
        var_names=var_names, n_classes=n_classes,
        seed=seed, epochs=epochs, aggregation=aggregation,
    ).fit(
        X_ts[train_idx], mask[train_idx], y[train_idx],
        x_val=(X_ts[val_idx], mask[val_idx], y[val_idx]),
    )

    z_per_time_val = tbn.predict_branch_probs_per_time(
        X_ts[val_idx], mask[val_idx],
    )                                              # [N_val, T, B]
    z_val = temporal_aggregate(z_per_time_val, mode=aggregation)
    theta = build_theta_matrix(tbn.branches, n_classes)
    branch_rules = [render_l4_rule(br, var_names) for br in tbn.branches]
    branch_families = [_branch_family(br) for br in tbn.branches]
    proba = aggregate_weighted_mean(z_val, theta)
    pred = _binary_predictions(proba, decision_threshold)
    y_val = y[val_idx]
    print(f"L4 case-study val accuracy ({case_selection}): {accuracy_score(y_val, pred):.3f}")

    audit_k_values = _normalise_k_values(audit_k_values)
    export_top_k = max([int(top_k), *audit_k_values])
    chosen = _select_case_samples(y_val, pred, n_samples_per_class, seed, mode=case_selection)
    samples: List[Dict] = []
    for local_idx, case_type in chosen:
        global_id = int(val_idx[local_idx])
        z_per_time_row = z_per_time_val[local_idx]   # [T, B]
        z_row = z_val[local_idx]                     # [B]
        support = z_row[:, None] * theta             # [B, K]
        pred_class = int(pred[local_idx])
        order = np.argsort(-support[:, pred_class])
        top = order[:export_top_k]
        top_branches = []
        for br_idx in top:
            br = tbn.branches[int(br_idx)]
            theta_k_list = _class_proportions_to_theta(br) or [0.0] * n_classes
            theta_k = float(theta_k_list[pred_class] if pred_class < len(theta_k_list) else 0.0)
            # peak timestep — when this branch was most active for this sample
            peak_t = int(np.argmax(z_per_time_row[:, int(br_idx)]))
            top_branches.append({
                "branch_id": br.branch_id,
                "tree_id": int(br.tree_id),
                "theta_k": theta_k,
                "p_z_aggregated": float(z_row[int(br_idx)]),
                "p_z_peak_timestep": peak_t,
                "p_z_peak_value": float(z_per_time_row[peak_t, int(br_idx)]),
                "support_score": float(support[int(br_idx), pred_class]),
                "rule": branch_rules[int(br_idx)],
                "rule_family": branch_families[int(br_idx)],
            })
        opposing_branches = []
        if n_classes > 1:
            opp_scores = support.copy()
            opp_scores[:, pred_class] = -np.inf
            opp_class = np.argmax(opp_scores, axis=1)
            opp_order = np.argsort(-opp_scores[np.arange(len(tbn.branches)), opp_class])[:top_k]
            for br_idx in opp_order:
                br = tbn.branches[int(br_idx)]
                cls = int(opp_class[int(br_idx)])
                theta_k_list = _class_proportions_to_theta(br) or [0.0] * n_classes
                theta_k = float(theta_k_list[cls] if cls < len(theta_k_list) else 0.0)
                peak_t = int(np.argmax(z_per_time_row[:, int(br_idx)]))
                opposing_branches.append({
                    "branch_id": br.branch_id,
                    "tree_id": int(br.tree_id),
                    "opposing_class": cls,
                    "theta_k": theta_k,
                    "p_z_aggregated": float(z_row[int(br_idx)]),
                    "p_z_peak_timestep": peak_t,
                    "p_z_peak_value": float(z_per_time_row[peak_t, int(br_idx)]),
                    "support_score": float(support[int(br_idx), cls]),
                    "rule": branch_rules[int(br_idx)],
                    "rule_family": branch_families[int(br_idx)],
                })
        audit_fields = _audit_counterfactual_fields(
            z_row,
            theta,
            pred_class,
            order,
            k_values=audit_k_values,
            branch_families=branch_families,
            random_seed=seed + int(global_id) * 1009,
            random_baseline_samples=random_baseline_samples,
        )
        samples.append({
            "x_id": global_id,
            "case_type": case_type,
            "decision_threshold": decision_threshold,
            "true_class": int(y_val[local_idx]),
            "predicted_class": pred_class,
            "class_proba": proba[local_idx].tolist(),
            "top_branches": top_branches,
            "opposing_branches": opposing_branches,
            **audit_fields,
        })

    return {
        "dataset": dataset_name,
        "level": "L4",
        "aggregation": aggregation,
        "n_branches": len(tbn.branches),
        "case_selection": case_selection,
        "decision_threshold": decision_threshold,
        "audit_k_values": list(audit_k_values),
        "samples": samples,
    }


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L3 / L4 case studies")
    p.add_argument("--dataset", default="p12")
    p.add_argument("--level", choices=["L3", "L4"], default="L3")
    p.add_argument("--n-samples", type=int, default=3,
                   help="Samples per class to extract.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K supporting branches per sample.")
    p.add_argument("--audit-k-values", default="1,3,5",
                   help="Comma-separated K values for sufficiency/deletion fields.")
    p.add_argument("--case-selection", choices=["balanced_true", "confusion"], default="balanced_true",
                   help="Case selection mode: balanced true classes or TP/TN/FP/FN buckets.")
    p.add_argument("--decision-threshold", type=float, default=None,
                   help="Optional binary threshold for confusion-bucket case selection.")
    p.add_argument("--random-baseline-samples", type=int, default=20,
                   help="Random rule deletion replicates per K.")
    p.add_argument("--preview-top-n", type=int, default=5,
                   help="Number of top rules printed in the terminal preview.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aggregation", default="mean",
                   help="L4 only: temporal aggregation mode.")
    p.add_argument(
        "--output-dir",
        default=os.path.join(THIS_DIR, "..", "output", "temporal"),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    X_ts, mask, y, var_names, dataset_name = load_temporal_dataset(args.dataset)
    audit_k_values = _parse_int_list(args.audit_k_values, default=(1, 3, 5))

    if args.level == "L3":
        result = case_studies_l3(
            X_ts, mask, y, var_names, dataset_name,
            n_samples_per_class=args.n_samples, top_k=args.top_k,
            seed=args.seed, epochs=args.epochs,
            audit_k_values=audit_k_values,
            case_selection=args.case_selection,
            decision_threshold=args.decision_threshold,
            random_baseline_samples=args.random_baseline_samples,
        )
    else:
        result = case_studies_l4(
            X_ts, mask, y, var_names, dataset_name,
            n_samples_per_class=args.n_samples, top_k=args.top_k,
            seed=args.seed, epochs=args.epochs,
            aggregation=args.aggregation,
            audit_k_values=audit_k_values,
            case_selection=args.case_selection,
            decision_threshold=args.decision_threshold,
            random_baseline_samples=args.random_baseline_samples,
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir,
        f"case_studies_{args.dataset}_{args.level}_{timestamp}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(result['samples'])} case-study samples → {out_path}")

    print("\n=== Sample preview ===")
    for s in result["samples"][:2]:
        print(
            f"\nx_id={s['x_id']}  true={s['true_class']}  pred={s['predicted_class']}"
            f"  P(class)={['%.3f' % p for p in s['class_proba']]}"
        )
        for br in s["top_branches"][:args.preview_top_n]:
            print(f"  • {br['branch_id']}  θ={br['theta_k']:.3f}  IF {br['rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
