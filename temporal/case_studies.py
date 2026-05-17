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

def _select_samples_per_class(
    y_val: np.ndarray, n_per_class: int, seed: int = 0,
) -> List[int]:
    rng = np.random.default_rng(seed)
    out: List[int] = []
    for cls in np.unique(y_val):
        idx = np.flatnonzero(y_val == cls)
        rng.shuffle(idx)
        out.extend(idx[:n_per_class].tolist())
    return out


def case_studies_l3(
    X_ts: np.ndarray, mask: np.ndarray, y: np.ndarray,
    var_names: Sequence[str], dataset_name: str,
    n_samples_per_class: int, top_k: int, seed: int, epochs: int,
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
    proba = aggregate_weighted_mean(bp_val, theta)
    pred = np.argmax(proba, axis=1)
    print(f"L3 case-study val accuracy: {accuracy_score(y_val, pred):.3f}")

    chosen = _select_samples_per_class(y_val, n_samples_per_class, seed)

    samples: List[Dict] = []
    for local_idx in chosen:
        global_id = int(val_idx[local_idx])
        z_row = bp_val[local_idx]                 # [B]
        support = z_row[:, None] * theta          # [B, K]
        order = np.argsort(-support[:, int(pred[local_idx])])
        top = order[:top_k]
        top_branches = []
        for br_idx in top:
            br = branches[int(br_idx)]
            pred_class = int(pred[local_idx])
            theta_k_list = _class_proportions_to_theta(br) or [0.0] * n_classes
            theta_k = float(theta_k_list[pred_class] if pred_class < len(theta_k_list) else 0.0)
            top_branches.append({
                "branch_id": br.branch_id,
                "tree_id": int(br.tree_id),
                "theta_k": theta_k,
                "p_z_posterior": float(z_row[int(br_idx)]),
                "support_score": float(support[int(br_idx), int(pred[local_idx])]),
                "rule": render_l3_rule(br, extractor.feature_meta),
            })
        samples.append({
            "x_id": global_id,
            "true_class": int(y_val[local_idx]),
            "predicted_class": int(pred[local_idx]),
            "class_proba": proba[local_idx].tolist(),
            "top_branches": top_branches,
        })

    return {
        "dataset": dataset_name,
        "level": "L3",
        "n_branches": len(branches),
        "samples": samples,
    }


def case_studies_l4(
    X_ts: np.ndarray, mask: np.ndarray, y: np.ndarray,
    var_names: Sequence[str], dataset_name: str,
    n_samples_per_class: int, top_k: int, seed: int, epochs: int,
    aggregation: str = "mean",
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
    proba = aggregate_weighted_mean(z_val, theta)
    pred = np.argmax(proba, axis=1)
    y_val = y[val_idx]
    print(f"L4 case-study val accuracy: {accuracy_score(y_val, pred):.3f}")

    chosen = _select_samples_per_class(y_val, n_samples_per_class, seed)
    samples: List[Dict] = []
    for local_idx in chosen:
        global_id = int(val_idx[local_idx])
        z_per_time_row = z_per_time_val[local_idx]   # [T, B]
        z_row = z_val[local_idx]                     # [B]
        support = z_row[:, None] * theta             # [B, K]
        order = np.argsort(-support[:, int(pred[local_idx])])
        top = order[:top_k]
        top_branches = []
        for br_idx in top:
            br = tbn.branches[int(br_idx)]
            pred_class = int(pred[local_idx])
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
                "support_score": float(support[int(br_idx), int(pred[local_idx])]),
                "rule": render_l4_rule(br, var_names),
            })
        samples.append({
            "x_id": global_id,
            "true_class": int(y_val[local_idx]),
            "predicted_class": int(pred[local_idx]),
            "class_proba": proba[local_idx].tolist(),
            "top_branches": top_branches,
        })

    return {
        "dataset": dataset_name,
        "level": "L4",
        "aggregation": aggregation,
        "n_branches": len(tbn.branches),
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

    if args.level == "L3":
        result = case_studies_l3(
            X_ts, mask, y, var_names, dataset_name,
            n_samples_per_class=args.n_samples, top_k=args.top_k,
            seed=args.seed, epochs=args.epochs,
        )
    else:
        result = case_studies_l4(
            X_ts, mask, y, var_names, dataset_name,
            n_samples_per_class=args.n_samples, top_k=args.top_k,
            seed=args.seed, epochs=args.epochs,
            aggregation=args.aggregation,
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
        for br in s["top_branches"]:
            print(f"  • {br['branch_id']}  θ={br['theta_k']:.3f}  IF {br['rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
