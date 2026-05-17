"""§5.4 — Spot-check L3 temporal ProbLog programs against the
analytical posterior used in :mod:`problog_inference`.

The L3 export emits exactly the same Bayesian-update model as the
static `export_full_problog_program`, only with temporal atom names
(``gt_mean(b0,hr,0,12,95.0,X)`` instead of ``gt(b0,fJ,t0_3,X)``).  In
particular, evaluating the program through the native ProbLog engine
must produce the same per-class posterior as
:func:`problog_inference._compute_analytical_posterior`.

This module verifies that equivalence on a small synthetic dataset.
ProbLog's SDD compilation is exponential in the number of branches /
conditions, so the spot-check intentionally uses a small problem
(``n_samples ≈ 30``, ``n_intervals = 4``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from problog import get_evaluatable  # noqa: E402
from problog.program import PrologString  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402

from rule_network_model import RuleNetworkModel  # noqa: E402
from problog_inference import _compute_analytical_posterior  # noqa: E402

from .datasets import load_synthetic_p12  # noqa: E402
from .interval_forest import IntervalFeatureExtractor  # noqa: E402
from .temporal_problog import export_temporal_problog_program  # noqa: E402


def _train_tiny_rule_network(
    X: np.ndarray, y: np.ndarray, seed: int, epochs: int,
    n_estimators: int, max_leaf_nodes: int,
) -> RuleNetworkModel:
    """Build a deliberately small RuleNetwork so the resulting ProbLog
    program compiles in seconds.  We do **not** truncate the branch
    list afterwards — the spot-check operates on the *full* program of
    the trained forest.
    """
    forest = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X, y)
    model = RuleNetworkModel()
    model.build_model_from_ensemble(forest)
    model.fit(
        X.astype(np.float32), y.astype(np.int64),
        X.astype(np.float32), y.astype(np.int64),
        learning_rate=0.01, epochs=epochs,
    )
    return model


def _run_problog(program: str) -> Dict[str, float]:
    """Compile ``program`` with ProbLog and return the dict of query
    probabilities."""
    formula = get_evaluatable().create_from(PrologString(program))
    raw = formula.evaluate()
    return {str(k): float(v) for k, v in raw.items()}


def _engine_class_probs(
    raw: Dict[str, float], x_id: int, n_classes: int,
) -> np.ndarray:
    """Pull ``class(x_id, k)`` queries out of the raw ProbLog dict and
    renormalise to a proper distribution."""
    probs = np.zeros(n_classes)
    for k in range(n_classes):
        key = f"class({x_id},{k})"
        if key in raw:
            probs[k] = raw[key]
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return probs


def spotcheck_l3(
    n_samples: int = 20, n_intervals: int = 2, T: int = 6,
    seed: int = 42, n_check: int = 3, atol: float = 1e-3,
    epochs: int = 15,
    n_estimators: int = 2, max_leaf_nodes: int = 4,
) -> List[Dict]:
    """Run the spot-check; raise ``AssertionError`` if any sample's
    engine-derived posterior is more than ``atol`` away from the
    analytical posterior.

    The defaults intentionally produce a small (≈8–12 branches) but
    **complete** model — every branch and every condition is exported
    into the ProbLog program; nothing is truncated.  The point of the
    spot-check is to demonstrate parity on the full program of a
    reduced-size instance, not to approximate parity on a slice of a
    larger one.
    """
    print(f"\n=== L3 ProbLog spot-check (n_samples={n_samples}, "
          f"T={T}, n_intervals={n_intervals}, "
          f"trees={n_estimators}, max_leaves={max_leaf_nodes}) ===")
    X_ts, mask, y, var_names, _ = load_synthetic_p12(
        n_samples=n_samples, T=T, missing_ratio=0.4, seed=seed,
    )
    n_classes = int(np.unique(y).size)

    extractor = IntervalFeatureExtractor(
        var_names=var_names, T=T, n_intervals=n_intervals, seed=seed,
    )
    X = extractor.transform(X_ts, mask)
    feat_meta = extractor.feature_meta

    model = _train_tiny_rule_network(
        X, y, seed=seed, epochs=epochs,
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
    )
    branches = model.branches
    bp = model.predict_branch_proba(X).numpy()
    print(f"  fitted {len(branches)} branches (full program exported); "
          f"checking first {n_check} samples")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(y), size=min(n_check, len(y)), replace=False)

    results: List[Dict] = []
    max_abs_err = 0.0
    for x_id in chosen:
        program = export_temporal_problog_program(
            branches=branches,
            branch_probs_single=bp[int(x_id)],
            interval_feature_row=X[int(x_id)],
            feature_meta=feat_meta,
            x_id=int(x_id),
            n_classes=n_classes,
        )

        t0 = time.time()
        raw = _run_problog(program)
        engine_secs = time.time() - t0
        engine_probs = _engine_class_probs(raw, int(x_id), n_classes)

        analytic_probs = _compute_analytical_posterior(
            branches=branches,
            branch_probs=bp[int(x_id):int(x_id) + 1],
            X=X[int(x_id):int(x_id) + 1],
            n_classes=n_classes,
        )[0]
        err = float(np.max(np.abs(engine_probs - analytic_probs)))
        max_abs_err = max(max_abs_err, err)
        print(f"  x_id={x_id:>4}  analytic={np.round(analytic_probs, 4).tolist()}  "
              f"engine={np.round(engine_probs, 4).tolist()}  "
              f"|Δ|max={err:.4e}  ProbLog={engine_secs:.1f}s")

        results.append({
            "x_id": int(x_id),
            "analytic": analytic_probs.tolist(),
            "engine": engine_probs.tolist(),
            "max_abs_err": err,
            "engine_seconds": engine_secs,
        })

    print(f"  → max abs error across {len(results)} samples: {max_abs_err:.4e}")
    if max_abs_err > atol:
        raise AssertionError(
            f"L3 ProbLog engine vs analytical posterior diverge by "
            f"{max_abs_err:.4e} > tol={atol:.1e}"
        )
    return results


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spot-check L3 temporal ProbLog program parity. "
                    "Forest size controls SDD compilation cost; reduce "
                    "n-estimators / max-leaf-nodes to keep the full "
                    "program tractable for the engine.",
    )
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--n-intervals", type=int, default=2)
    p.add_argument("--T", type=int, default=6)
    p.add_argument("--n-check", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-estimators", type=int, default=2,
                   help="Forest size — keeps SDD tractable.")
    p.add_argument("--max-leaf-nodes", type=int, default=4,
                   help="Per-tree leaf cap → controls branch count.")
    p.add_argument("--tolerance", type=float, default=1e-3)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    spotcheck_l3(
        n_samples=args.n_samples,
        n_intervals=args.n_intervals,
        T=args.T,
        n_check=args.n_check,
        seed=args.seed,
        epochs=args.epochs,
        n_estimators=args.n_estimators,
        max_leaf_nodes=args.max_leaf_nodes,
        atol=args.tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
