"""Alternative symbolic rule sources for the PPtheta-Post tabular pipeline.

Each rule source fits a base model on ``(X, y)`` and exposes:

* ``branches_per_tree`` — ``List[List[Branch]]`` consumed by
  :meth:`rule_network.RuleNetwork.build_model_from_branches`.
* ``native_model`` — the fitted base model, kept for the ``*_native``
  baseline variant (uses ``predict_proba`` directly without PPtheta-Post
  inference) so the report can show how the source on its own performs.
* ``n_features`` / ``n_classes`` — passed through to RuleNetwork.

The default source is ``ExtraTreesRuleSource`` (drop-in for the legacy
``ExtraTreesClassifier`` block at compare_datasets.py:602).  XGBoost and
CatBoost rule extractors are ported from
``neuro-symbolic-toolkit/nstoolkit/acquisition/{xgb,catboost}_backbone.py``
with two improvements over the original ports:

1. XGBoost multiclass class proportions are no longer uniform — for each
   tree we identify the target class (``tree_idx % num_class`` under
   ``multi:softprob``) and concentrate the soft proportion on that
   class.  Uniform proportions made every branch indistinguishable to
   ``build_theta_matrix`` and starved the PL- variants of signal.
2. Both XGB and CatBoost adapters set sensible defaults
   (``n_estimators=50``, ``depth=4``) so that branch counts stay
   sub-1000 on small benchmark datasets.  Override via ``**kwargs``.

FIGS produces ``imodels.tree.figs.Node`` trees (a recursive structure,
not a sklearn ``Tree``); RuleFit yields textual rule strings.  Both are
walked into ``Branch`` objects with the same parent-of-leaf convention.
RuleFit is binary-only — multiclass datasets raise ``NotImplementedError``
which the driver should catch and surface as ``skipped``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from branch_schema import Branch, Condition


# --------------------------------------------------------------------------- #
# Empirical class proportions — shared post-processor
# --------------------------------------------------------------------------- #


def _subsample_for_refinement(
    X: np.ndarray, y: np.ndarray, max_samples: Optional[int], seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optional stratified down-sampling for refinement on huge datasets.

    Returns ``(X, y)`` unchanged when ``max_samples`` is ``None`` or already
    ≥ ``len(y)``.  Otherwise picks a stratified subset so the empirical
    class proportions stay representative — empirical cp is an estimate, so
    sub-sampling at 50k–100k samples preserves quality at a fraction of the
    runtime.
    """
    n = X.shape[0]
    if not max_samples or max_samples <= 0 or max_samples >= n:
        return X, y
    rng = np.random.default_rng(seed)
    # Stratified pick: at most (max_samples / n_classes) from each class,
    # falling back to uniform if a class is overrepresented.
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, max_samples // len(classes))
    idx_parts: List[np.ndarray] = []
    for cls, cnt in zip(classes, counts):
        pool = np.where(y == cls)[0]
        take = min(per_class, cnt)
        if take >= cnt:
            idx_parts.append(pool)
        else:
            idx_parts.append(rng.choice(pool, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _empirical_class_proportions(
    branches: List[Branch],
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    smooth: float = 1e-3,
) -> int:
    """Vectorised: overwrite ``branch.class_proportions`` with the empirical
    class fraction of training samples that satisfy every condition.

    This is the universal fallback path used by FIGS / RuleFit (which have
    no native ``pred_leaf`` API).  XGBoost / CatBoost short-circuit through
    :func:`_empirical_cp_via_leaf_indices` which scales much better.

    Optimisation: each Condition becomes a column lookup against ``X``.
    We collect every *unique* ``(feature_idx, threshold, direction)`` tuple
    across all branches and compute the boolean column once, then AND the
    relevant columns per branch.  Complexity drops from
    ``O(n_branches × n × depth)`` to
    ``O(n_unique_conditions × n + n_branches × depth)``.
    """
    if X.size == 0 or not branches:
        return 0

    cond_cache: Dict[Tuple[int, float, str], np.ndarray] = {}
    for b in branches:
        for cond in b.conditions:
            key = (int(cond.feature_idx), float(cond.threshold), cond.direction)
            if key in cond_cache:
                continue
            feat, thr, direction = key
            if not (0 <= feat < X.shape[1]):
                cond_cache[key] = np.zeros(X.shape[0], dtype=bool)
                continue
            col = X[:, feat]
            cond_cache[key] = (col <= thr) if direction == "le" else (col > thr)

    return _apply_empirical_cp_with_masks(
        branches, cond_cache, y, n_classes, smooth,
    )


def _apply_empirical_cp_with_masks(
    branches: List[Branch],
    cond_cache: Dict[Tuple[int, float, str], np.ndarray],
    y: np.ndarray,
    n_classes: int,
    smooth: float,
) -> int:
    refined = 0
    y64 = y.astype(np.int64, copy=False)
    n = y.shape[0]
    for b in branches:
        if not b.conditions:
            continue
        mask = np.ones(n, dtype=bool)
        for cond in b.conditions:
            key = (int(cond.feature_idx), float(cond.threshold), cond.direction)
            mask &= cond_cache.get(key, np.zeros(n, dtype=bool))
            if not mask.any():
                break
        if not mask.any():
            continue
        counts = np.bincount(y64[mask], minlength=n_classes).astype(np.float64)
        if smooth > 0:
            counts += smooth
        total = counts.sum()
        if total <= 0:
            continue
        b.class_proportions = (counts / total).tolist()
        refined += 1
    return refined


def _empirical_cp_via_leaf_indices(
    branches_per_tree: List[List[Branch]],
    leaf_ids_per_branch_per_tree: List[List[List[int]]],
    leaf_indices_matrix: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    smooth: float = 1e-3,
) -> int:
    """Fast empirical cp for GBM-style backbones with native leaf-index API.

    ``leaf_indices_matrix[s, t]`` is the leaf id (within tree ``t``) that
    sample ``s`` falls into.  For each branch we already know the leaf ids
    of its leaf children (``leaf_ids_per_branch_per_tree[t][i]``) so the
    mask is a vectorised ``np.isin`` — O(n × n_trees) regardless of
    branch count.

    Falls back silently to the generic condition-eval path for any branch
    without recorded leaf ids.
    """
    if y.size == 0:
        return 0
    refined = 0
    y64 = y.astype(np.int64, copy=False)
    n = y64.shape[0]
    n_trees_in_matrix = leaf_indices_matrix.shape[1] if leaf_indices_matrix.ndim == 2 else 0

    for t, (br_t, leaf_lists) in enumerate(
        zip(branches_per_tree, leaf_ids_per_branch_per_tree)
    ):
        if t >= n_trees_in_matrix:
            break
        col = leaf_indices_matrix[:, t]
        for branch, leaf_ids in zip(br_t, leaf_lists):
            if not leaf_ids:
                continue
            mask = np.isin(col, leaf_ids)
            if not mask.any():
                continue
            counts = np.bincount(
                y64[mask], minlength=n_classes,
            ).astype(np.float64)
            if smooth > 0:
                counts += smooth
            total = counts.sum()
            if total <= 0:
                continue
            branch.class_proportions = (counts / total).tolist()
            refined += 1
    return refined


# --------------------------------------------------------------------------- #
# Common contract
# --------------------------------------------------------------------------- #


@dataclass
class FittedRuleSource:
    """Container for everything a rule source produces.

    Attributes
    ----------
    name
        Short key, e.g. ``"xgb"`` (used in CSV column ``rule_source``).
    branches_per_tree
        Outer list = tree id, inner = parent-of-leaf branches in DFS order.
    n_features, n_classes
        Required to build the RuleNetwork weight matrices.
    native_model
        Fitted base estimator — used by the ``*_native`` variant.
    fit_seconds
        Wall-clock fit time (for the base estimator only, not for the
        downstream RuleNetwork fit).
    extra
        Source-specific diagnostics (e.g. ``{"n_oblivious_trees": 50}``).
    """

    name: str
    branches_per_tree: List[List[Branch]]
    n_features: int
    n_classes: int
    native_model: Any
    fit_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_branches(self) -> int:
        return sum(len(t) for t in self.branches_per_tree)


class RuleSource(ABC):
    """Abstract base.  Subclasses implement :meth:`_fit` only."""

    name: str = "abstract"
    supports_multiclass: bool = True

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
        n_classes: int,
        seed: int,
        refinement_max_samples: Optional[int] = None,
    ) -> FittedRuleSource:
        if n_classes > 2 and not self.supports_multiclass:
            raise NotImplementedError(
                f"{self.name!r} does not support multiclass classification "
                f"(n_classes={n_classes})"
            )
        self._refinement_max_samples = refinement_max_samples
        t0 = time.time()
        branches_per_tree, native_model, extra = self._fit(
            X, y, n_features=n_features, n_classes=n_classes, seed=seed,
        )
        return FittedRuleSource(
            name=self.name,
            branches_per_tree=branches_per_tree,
            n_features=n_features,
            n_classes=n_classes,
            native_model=native_model,
            fit_seconds=time.time() - t0,
            extra=extra,
        )

    def _refine_X(self, X: np.ndarray, y: np.ndarray, seed: int):
        """Return possibly-subsampled (X, y) for empirical cp refinement."""
        return _subsample_for_refinement(
            X, y, getattr(self, "_refinement_max_samples", None), seed,
        )

    @abstractmethod
    def _fit(
        self, X, y, *, n_features, n_classes, seed,
    ) -> Tuple[List[List[Branch]], Any, Dict[str, Any]]:
        ...

    def predict_proba_native(
        self, native_model: Any, X: np.ndarray, n_classes: int,
    ) -> np.ndarray:
        """Default: assume sklearn-like ``predict_proba`` with all columns."""
        proba = native_model.predict_proba(X)
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        if proba.shape[1] < n_classes:
            full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
            classes = getattr(native_model, "classes_", np.arange(proba.shape[1]))
            for col, cls in enumerate(classes):
                full[:, int(cls)] = proba[:, col]
            proba = full
        return proba


# --------------------------------------------------------------------------- #
# ExtraTrees (the legacy default — kept as a RuleSource for symmetry)
# --------------------------------------------------------------------------- #


class ExtraTreesRuleSource(RuleSource):
    name = "extratrees"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        from sklearn.ensemble import ExtraTreesClassifier
        from rule_network import extract_branches_from_sklearn_ensemble

        params = dict(
            n_estimators=self.kwargs.get("n_estimators", 8),
            max_leaf_nodes=self.kwargs.get("max_leaf_nodes", 24),
            random_state=seed,
            n_jobs=self.kwargs.get("n_jobs", 1),
        )
        et = ExtraTreesClassifier(**params).fit(X, y)
        branches_per_tree = extract_branches_from_sklearn_ensemble(et)
        return branches_per_tree, et, {"estimator": "ExtraTreesClassifier"}


# --------------------------------------------------------------------------- #
# XGBoost — recursive JSON tree walk
# --------------------------------------------------------------------------- #


def _xgb_class_proportions(leaf_weights: List[float], n_classes: int,
                            tree_idx: int) -> np.ndarray:
    """Map a tree's leaf weights to a soft class distribution.

    Under ``multi:softprob`` XGBoost lays out trees as
    ``[iter0/class0, iter0/class1, ..., iter0/classK-1, iter1/class0, ...]``
    so ``class_for_tree = tree_idx % n_classes``.  We push a fraction of
    mass onto that class proportional to ``sigmoid(mean(leaf_weights))``
    and split the remainder uniformly.  This is a heuristic — recovering
    the true posterior would require all sibling trees of the iteration
    — but it is strictly better than the uniform fallback from the
    original neuro-symbolic-toolkit port, which made every branch
    indistinguishable to ``build_theta_matrix``.
    """
    if not leaf_weights:
        return np.ones(n_classes, dtype=np.float64) / n_classes
    w = float(np.mean(leaf_weights))
    p = 1.0 / (1.0 + np.exp(-abs(w)))  # |w| so direction collapses to magnitude
    if n_classes == 2:
        p1 = 1.0 / (1.0 + np.exp(-w))
        return np.array([1.0 - p1, p1], dtype=np.float64)
    cp = np.full(n_classes, (1.0 - p) / (n_classes - 1), dtype=np.float64)
    cp[tree_idx % n_classes] = p
    cp /= cp.sum()
    return cp


def _xgb_walk(node: dict, *, n_features: int, n_classes: int, tree_idx: int,
              branch_offset: int, path: List[Condition],
              out: List[Branch],
              leaf_ids_per_branch: List[List[int]]) -> None:
    if "leaf" in node:
        return
    children = node.get("children") or []
    yes_id, no_id = node.get("yes"), node.get("no")
    has_leaf_child = any(
        "leaf" in c for c in children
        if c.get("nodeid") in (yes_id, no_id)
    )

    feat_str = str(node.get("split", "f0"))
    try:
        split_feature = int(feat_str.lstrip("f"))
    except ValueError:
        split_feature = -1
    split_threshold = float(node.get("split_condition", 0.0))

    if has_leaf_child:
        leaf_weights = [
            float(c["leaf"]) for c in children if "leaf" in c
        ]
        cp = _xgb_class_proportions(leaf_weights, n_classes, tree_idx)
        # Record leaf-ids of this parent's leaf children (for pred_leaf-based
        # empirical cp refinement; falls back to condition-eval if missing).
        leaf_ids = [
            int(c.get("nodeid", -1)) for c in children if "leaf" in c
        ]
        leaf_ids_per_branch.append(leaf_ids)
        out.append(Branch(
            branch_id=f"b{branch_offset + len(out)}",
            tree_id=tree_idx,
            parent_node_id=int(node.get("nodeid", -1)),
            conditions=list(path),
            class_proportions=cp.tolist(),
            split_feature_idx=(split_feature if split_feature >= 0 else None),
            split_threshold=split_threshold,
            split_node_id=int(node.get("nodeid", -1)),
        ))

    for c in children:
        if "leaf" in c:
            continue
        cid = c.get("nodeid")
        if cid == yes_id:
            direction = "le"
        elif cid == no_id:
            direction = "gt"
        else:
            continue
        cond = Condition(
            feature_idx=split_feature,
            threshold=split_threshold,
            direction=direction,
            node_id=int(node.get("nodeid", -1)),
        )
        _xgb_walk(
            c, n_features=n_features, n_classes=n_classes,
            tree_idx=tree_idx, branch_offset=branch_offset,
            path=path + [cond], out=out,
            leaf_ids_per_branch=leaf_ids_per_branch,
        )


class XGBoostRuleSource(RuleSource):
    name = "xgb"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        import xgboost as xgb

        params = dict(
            n_estimators=self.kwargs.get("n_estimators", 50),
            max_depth=self.kwargs.get("max_depth", 4),
            learning_rate=self.kwargs.get("learning_rate", 0.1),
            random_state=seed,
            verbosity=0,
            n_jobs=self.kwargs.get("n_jobs", 1),
        )
        if n_classes > 2:
            params.setdefault("objective", "multi:softprob")
            params.setdefault("num_class", n_classes)
        else:
            params.setdefault("objective", "binary:logistic")

        clf = xgb.XGBClassifier(**params).fit(X, y)
        booster = clf.get_booster()
        dump = booster.get_dump(dump_format="json")

        branches_per_tree: List[List[Branch]] = []
        leaf_ids_per_branch_per_tree: List[List[List[int]]] = []
        offset = 0
        for t, tree_json in enumerate(dump):
            node = json.loads(tree_json)
            br_t: List[Branch] = []
            leaf_ids_t: List[List[int]] = []
            _xgb_walk(
                node, n_features=n_features, n_classes=n_classes,
                tree_idx=t, branch_offset=offset, path=[], out=br_t,
                leaf_ids_per_branch=leaf_ids_t,
            )
            branches_per_tree.append(br_t)
            leaf_ids_per_branch_per_tree.append(leaf_ids_t)
            offset += len(br_t)

        # Fast empirical cp via pred_leaf (O(n × n_trees) instead of
        # O(n_branches × n × depth)).  Subsample for very large datasets.
        Xr, yr = self._refine_X(X, y, seed)
        leaf_idx_matrix = booster.predict(
            xgb.DMatrix(Xr), pred_leaf=True,
        )
        leaf_idx_matrix = np.asarray(leaf_idx_matrix, dtype=np.int64)
        if leaf_idx_matrix.ndim == 1:
            leaf_idx_matrix = leaf_idx_matrix.reshape(-1, 1)
        n_refined = _empirical_cp_via_leaf_indices(
            branches_per_tree, leaf_ids_per_branch_per_tree,
            leaf_idx_matrix, yr, n_classes,
        )
        return branches_per_tree, clf, {
            "n_boosters": len(dump),
            "n_branches_refined": n_refined,
            "refinement_mode": "pred_leaf",
            "refinement_samples": int(Xr.shape[0]),
        }


# --------------------------------------------------------------------------- #
# CatBoost — oblivious tree walk
# --------------------------------------------------------------------------- #


def _catboost_leaf_ids_for_parent(parent_idx: int, depth: int) -> List[int]:
    """Map an MSB-first parent index (depth-1 internal node) to the two
    LSB-first CatBoost leaf IDs that share that parent.

    CatBoost ``calc_leaf_indexes`` encodes the leaf id as
    ``sum(bit_d * 2**d for d in range(depth))`` where ``bit_d`` is the
    direction taken at ``splits[d]``.  Our branch walk, on the other hand,
    iterates parents with ``bit_d = (parent_idx >> (depth - 2 - d)) & 1``
    (MSB-first path).  This helper converts between the two conventions.
    Verified empirically against CatBoost 1.2.x on breast_cancer +
    multiclass wine: ``parent_idx=2, depth=3 -> [1, 5]`` matches
    ``np.unique(leaf_indices_for_mask)``.
    """
    base = 0
    for d in range(depth - 1):
        bit = (parent_idx >> (depth - 2 - d)) & 1
        base |= bit << d
    high_bit = 1 << (depth - 1)
    return [base, base | high_bit]


def _catboost_branches_from_oblivious(tree_dict: dict, *, tree_idx: int,
                                       branch_offset: int, n_features: int,
                                       n_classes: int) -> List[Branch]:
    splits = tree_dict.get("splits", [])
    leaf_values = tree_dict.get("leaf_values", [])
    depth = len(splits)
    if depth == 0:
        return []

    branches: List[Branch] = []
    n_internal_at_last = 1 << (depth - 1)
    for parent_idx in range(n_internal_at_last):
        path_conditions: List[Condition] = []
        for d in range(depth - 1):
            split = splits[d]
            feat = int(split.get("float_feature_index",
                                  split.get("feature_index", -1)))
            border = float(split.get("border", 0.0))
            bit = (parent_idx >> (depth - 2 - d)) & 1
            path_conditions.append(Condition(
                feature_idx=feat, threshold=border,
                direction=("le" if bit == 0 else "gt"),
                node_id=d,
            ))

        last_split = splits[-1]
        last_feat = int(last_split.get("float_feature_index",
                                        last_split.get("feature_index", -1)))
        last_border = float(last_split.get("border", 0.0))

        left_leaf = parent_idx * 2
        right_leaf = parent_idx * 2 + 1

        if n_classes > 2:
            cp = np.zeros(n_classes, dtype=np.float64)
            for li in (left_leaf, right_leaf):
                if li * n_classes + n_classes <= len(leaf_values):
                    block = np.asarray(
                        leaf_values[li * n_classes:(li + 1) * n_classes],
                        dtype=np.float64,
                    )
                    e = np.exp(block - block.max())
                    cp += e / e.sum()
            cp /= max(cp.sum(), 1e-15)
        else:
            lv: List[float] = []
            if left_leaf < len(leaf_values):
                lv.append(float(leaf_values[left_leaf]))
            if right_leaf < len(leaf_values):
                lv.append(float(leaf_values[right_leaf]))
            w = float(np.mean(lv)) if lv else 0.0
            p1 = 1.0 / (1.0 + np.exp(-w))
            cp = np.array([1.0 - p1, p1], dtype=np.float64)

        branches.append(Branch(
            branch_id=f"b{branch_offset + len(branches)}",
            tree_id=tree_idx,
            parent_node_id=parent_idx,
            conditions=path_conditions,
            class_proportions=cp.tolist(),
            split_feature_idx=(last_feat if last_feat >= 0 else None),
            split_threshold=last_border,
            split_node_id=parent_idx,
        ))
    return branches


class CatBoostRuleSource(RuleSource):
    name = "catboost"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        from catboost import CatBoostClassifier

        params = dict(
            iterations=self.kwargs.get("iterations", 50),
            depth=self.kwargs.get("depth", 4),
            learning_rate=self.kwargs.get("learning_rate", 0.1),
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=self.kwargs.get("n_jobs", 1),
        )
        clf = CatBoostClassifier(**params).fit(X, y)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.json")
            clf.save_model(path, format="json")
            with open(path, "r") as f:
                model_json = json.load(f)

        trees = (
            model_json.get("oblivious_trees")
            or model_json.get("trees")
            or []
        )
        branches_per_tree: List[List[Branch]] = []
        leaf_ids_per_branch_per_tree: List[List[List[int]]] = []
        offset = 0
        for t, tree_dict in enumerate(trees):
            br_t = _catboost_branches_from_oblivious(
                tree_dict, tree_idx=t, branch_offset=offset,
                n_features=n_features, n_classes=n_classes,
            )
            depth = len(tree_dict.get("splits", []))
            leaf_ids_t = [
                _catboost_leaf_ids_for_parent(int(b.parent_node_id), depth)
                for b in br_t
            ]
            branches_per_tree.append(br_t)
            leaf_ids_per_branch_per_tree.append(leaf_ids_t)
            offset += len(br_t)

        Xr, yr = self._refine_X(X, y, seed)
        leaf_idx_matrix = np.asarray(
            clf.calc_leaf_indexes(Xr), dtype=np.int64,
        )
        if leaf_idx_matrix.ndim == 1:
            leaf_idx_matrix = leaf_idx_matrix.reshape(-1, 1)
        n_refined = _empirical_cp_via_leaf_indices(
            branches_per_tree, leaf_ids_per_branch_per_tree,
            leaf_idx_matrix, yr, n_classes,
        )
        return branches_per_tree, clf, {
            "n_oblivious_trees": len(trees),
            "n_branches_refined": n_refined,
            "refinement_mode": "calc_leaf_indexes",
            "refinement_samples": int(Xr.shape[0]),
        }


# --------------------------------------------------------------------------- #
# FIGS — walk imodels.tree.figs.Node recursive structure
# --------------------------------------------------------------------------- #


def _figs_node_class_proportions(node, n_classes: int) -> np.ndarray:
    """Pull soft class distribution from a FIGS internal node.

    ``Node.value`` is the per-class predicted score at this node; for
    classification it is already a probability-like vector.  We
    L1-normalise to be safe.
    """
    val = node.value
    if val is None:
        return np.ones(n_classes, dtype=np.float64) / n_classes
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 1 and n_classes == 2:
        p = float(arr[0])
        return np.array([1.0 - p, p], dtype=np.float64)
    if arr.shape[0] != n_classes:
        pad = np.zeros(n_classes, dtype=np.float64)
        pad[:min(n_classes, arr.shape[0])] = arr[:n_classes]
        arr = pad
    s = arr.sum()
    return arr / s if s > 0 else np.ones(n_classes) / n_classes


def _figs_walk(node, *, tree_idx: int, branch_offset: int, n_classes: int,
               path: List[Condition], out: List[Branch]) -> None:
    if node is None or getattr(node, "feature", None) is None:
        return
    left, right = node.left, node.right
    is_leaf = lambda n: n is None or getattr(n, "feature", None) is None
    has_leaf_child = is_leaf(left) or is_leaf(right)

    split_feature = int(node.feature)
    split_threshold = float(node.threshold)

    if has_leaf_child:
        cp = _figs_node_class_proportions(node, n_classes)
        out.append(Branch(
            branch_id=f"b{branch_offset + len(out)}",
            tree_id=tree_idx,
            parent_node_id=int(getattr(node, "node_id", -1)),
            conditions=list(path),
            class_proportions=cp.tolist(),
            split_feature_idx=split_feature,
            split_threshold=split_threshold,
            split_node_id=int(getattr(node, "node_id", -1)),
        ))

    if not is_leaf(left):
        _figs_walk(
            left, tree_idx=tree_idx, branch_offset=branch_offset,
            n_classes=n_classes,
            path=path + [Condition(
                feature_idx=split_feature, threshold=split_threshold,
                direction="le", node_id=int(getattr(node, "node_id", -1)),
            )],
            out=out,
        )
    if not is_leaf(right):
        _figs_walk(
            right, tree_idx=tree_idx, branch_offset=branch_offset,
            n_classes=n_classes,
            path=path + [Condition(
                feature_idx=split_feature, threshold=split_threshold,
                direction="gt", node_id=int(getattr(node, "node_id", -1)),
            )],
            out=out,
        )


class FIGSRuleSource(RuleSource):
    name = "figs"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        from imodels import FIGSClassifier

        params = dict(
            max_rules=self.kwargs.get("max_rules", 30),
            max_trees=self.kwargs.get("max_trees", None),
            random_state=seed,
        )
        clf = FIGSClassifier(**{k: v for k, v in params.items() if v is not None})
        clf.fit(X, y)

        branches_per_tree: List[List[Branch]] = []
        offset = 0
        for t, root in enumerate(clf.trees_):
            br_t: List[Branch] = []
            _figs_walk(
                root, tree_idx=t, branch_offset=offset,
                n_classes=n_classes, path=[], out=br_t,
            )
            branches_per_tree.append(br_t)
            offset += len(br_t)
        Xr, yr = self._refine_X(X, y, seed)
        n_refined = sum(
            _empirical_class_proportions(br, Xr, yr, n_classes)
            for br in branches_per_tree
        )
        return branches_per_tree, clf, {
            "n_figs_trees": len(clf.trees_),
            "n_branches_refined": n_refined,
            "refinement_mode": "vectorized_eval",
            "refinement_samples": int(Xr.shape[0]),
        }


# --------------------------------------------------------------------------- #
# RuleFit — parse textual rules into Branch objects (binary only)
# --------------------------------------------------------------------------- #


_RULEFIT_TERM_RE = re.compile(
    r"\s*(?P<feat>[A-Za-z_][A-Za-z0-9_]*|feature_\d+|X\d+|f\d+)\s*"
    r"(?P<op><=|<|>=|>|==)\s*"
    r"(?P<thr>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


def _parse_rulefit_rule(rule_str: str, feature_names: List[str]
                        ) -> Optional[List[Condition]]:
    """Parse a RuleFit rule string into a list of :class:`Condition`.

    RuleFit emits rules like ``"feature_3 <= 1.5 and feature_0 > 0.2"``.
    Returns ``None`` if any term fails to parse — the caller skips such
    rules rather than guessing.
    """
    parts = [p.strip() for p in re.split(r"\band\b", rule_str)]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    conditions: List[Condition] = []
    for term in parts:
        m = _RULEFIT_TERM_RE.match(term)
        if not m:
            return None
        feat = m.group("feat")
        op = m.group("op")
        thr = float(m.group("thr"))
        if feat in name_to_idx:
            feat_idx = name_to_idx[feat]
        else:
            digits = re.findall(r"\d+", feat)
            if not digits:
                return None
            feat_idx = int(digits[-1])
        direction = "le" if op in ("<=", "<") else "gt"
        conditions.append(Condition(
            feature_idx=feat_idx, threshold=thr,
            direction=direction, node_id=len(conditions),
        ))
    return conditions or None


class TabPFNDistillRuleSource(RuleSource):
    """Distil TabPFN into an interpretable tree ensemble, then extract rules.

    TabPFN itself has no symbolic structure — it's a prior-fitted
    transformer that does not expose trees or rules.  Instead of trying
    to interpret it, we use it as a *teacher*:

    1. Fit TabPFN on (X, y) and obtain soft labels
       ``p_soft = TabPFN.predict_proba(X)``.
    2. Fit a tree-ensemble student (XGBoost by default; ExtraTrees /
       CatBoost selectable via the ``student`` kwarg) on the hard
       argmax of ``p_soft`` with sample weights set to TabPFN's
       confidence ``max(p_soft, axis=1)``.  This is the "hard
       distillation with confidence weighting" recipe — cheaper and
       more stable on small datasets than per-class soft regression.
    3. Extract branches from the student exactly as the corresponding
       non-distilled rule source would, with empirical class
       proportions refined from the *original* (X, y) (not from
       TabPFN-induced labels) — that way the PPθ-Post pipeline sees
       the same data-grounded cp it would see for the bare student,
       but the student's *structure* has been guided by TabPFN's
       knowledge of feature interactions.

    Selectable students:

    * ``xgb`` (default) — fast, captures interactions, supports
      sample_weight natively.
    * ``extratrees`` — preserves the legacy sklearn-tree-shape
      branches that early PPθ-Post papers were built on.
    * ``catboost`` — oblivious trees; the same branch budget as the
      plain CatBoost rule source.
    """

    name = "tabpfn_distill"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        tabpfn_kwargs = dict(self.kwargs.get("tabpfn_kwargs", {}))
        if str(tabpfn_kwargs.get("device", "cpu")).lower() == "cpu":
            os.environ.setdefault("TABPFN_EXCLUDE_DEVICES", "mps")
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                "tabpfn is not installed.  `pip install -r requirements.txt` "
                "and download gated weights with "
                "`python download_tabpfn_ts_weights.py --kind classifier`."
            ) from e

        student_key = str(self.kwargs.get("student", "xgb")).lower()
        if student_key not in ("xgb", "extratrees", "et", "catboost", "cb"):
            raise ValueError(
                f"unknown distill student {student_key!r}; "
                "choose from xgb / extratrees / catboost"
            )

        # 1. Teacher: TabPFN.
        tabpfn_kwargs.setdefault("device", "cpu")
        tabpfn_kwargs.setdefault("random_state", seed)
        model_path = (
            tabpfn_kwargs.get("model_path")
            or os.environ.get("TABPFN_CLASSIFIER_MODEL_PATH")
            or os.environ.get("TABPFN_MODEL_PATH")
        )
        if model_path:
            tabpfn_kwargs["model_path"] = model_path
        tabpfn_kwargs.setdefault("show_progress_bar", False)
        import inspect
        sig = inspect.signature(TabPFNClassifier.__init__)
        teacher = TabPFNClassifier(
            **{k: v for k, v in tabpfn_kwargs.items() if k in sig.parameters},
        )
        teacher.fit(X, y)
        p_soft = np.asarray(teacher.predict_proba(X), dtype=np.float64)
        if p_soft.ndim == 1:
            p_soft = np.column_stack([1.0 - p_soft, p_soft])
        if p_soft.shape[1] < n_classes:
            full = np.zeros((p_soft.shape[0], n_classes), dtype=np.float64)
            full[:, :p_soft.shape[1]] = p_soft
            p_soft = full

        y_distill = p_soft.argmax(axis=1).astype(np.int64)
        sample_weight = p_soft.max(axis=1).astype(np.float64)
        # Stabilise: drop near-zero weights so the student doesn't see
        # essentially-uniform rows that would shrink branches to noise.
        sample_weight = np.clip(sample_weight, 1.0 / max(n_classes, 2), None)

        # 2. Student: route to one of the existing rule sources, but with
        #    distilled labels.  We do not call the registered _fit
        #    directly because we need sample_weight support and a custom
        #    y; instead we replicate the minimal training path inline.
        student_extra: Dict[str, Any] = {
            "tabpfn_n": int(X.shape[0]),
            "tabpfn_confidence_mean": float(sample_weight.mean()),
            "distill_student": student_key,
        }

        if student_key in ("extratrees", "et"):
            from sklearn.ensemble import ExtraTreesClassifier
            from rule_network import extract_branches_from_sklearn_ensemble
            params = dict(
                n_estimators=self.kwargs.get("n_estimators", 16),
                max_leaf_nodes=self.kwargs.get("max_leaf_nodes", 32),
                random_state=seed,
                n_jobs=self.kwargs.get("n_jobs", 1),
            )
            student = ExtraTreesClassifier(**params).fit(
                X, y_distill, sample_weight=sample_weight,
            )
            branches_per_tree = extract_branches_from_sklearn_ensemble(student)
            # ExtraTrees' sklearn-tree path already encodes empirical cp
            # from y_distill at the parent node.  Re-refine against the
            # original y so cp reflects ground truth, not teacher labels.
            Xr, yr = self._refine_X(X, y, seed)
            n_ref = sum(
                _empirical_class_proportions(br, Xr, yr, n_classes)
                for br in branches_per_tree
            )
            student_extra["n_branches_refined"] = n_ref
            student_extra["refinement_mode"] = "vectorized_eval"
            return branches_per_tree, student, student_extra

        if student_key == "xgb":
            import xgboost as xgb
            params = dict(
                n_estimators=self.kwargs.get("n_estimators", 50),
                max_depth=self.kwargs.get("max_depth", 4),
                learning_rate=self.kwargs.get("learning_rate", 0.1),
                random_state=seed, verbosity=0,
                n_jobs=self.kwargs.get("n_jobs", 1),
            )
            if n_classes > 2:
                params.setdefault("objective", "multi:softprob")
                params.setdefault("num_class", n_classes)
            else:
                params.setdefault("objective", "binary:logistic")
            student = xgb.XGBClassifier(**params).fit(
                X, y_distill, sample_weight=sample_weight,
            )
            booster = student.get_booster()
            dump = booster.get_dump(dump_format="json")
            branches_per_tree, leaf_ids_per_branch_per_tree = [], []
            offset = 0
            for t, tree_json in enumerate(dump):
                node = json.loads(tree_json)
                br_t, leaf_ids_t = [], []
                _xgb_walk(
                    node, n_features=n_features, n_classes=n_classes,
                    tree_idx=t, branch_offset=offset, path=[], out=br_t,
                    leaf_ids_per_branch=leaf_ids_t,
                )
                branches_per_tree.append(br_t)
                leaf_ids_per_branch_per_tree.append(leaf_ids_t)
                offset += len(br_t)
            Xr, yr = self._refine_X(X, y, seed)
            leaf_idx_matrix = np.asarray(
                booster.predict(xgb.DMatrix(Xr), pred_leaf=True),
                dtype=np.int64,
            )
            if leaf_idx_matrix.ndim == 1:
                leaf_idx_matrix = leaf_idx_matrix.reshape(-1, 1)
            n_ref = _empirical_cp_via_leaf_indices(
                branches_per_tree, leaf_ids_per_branch_per_tree,
                leaf_idx_matrix, yr, n_classes,
            )
            student_extra["n_branches_refined"] = n_ref
            student_extra["refinement_mode"] = "pred_leaf"
            student_extra["n_boosters"] = len(dump)
            return branches_per_tree, student, student_extra

        # CatBoost
        from catboost import CatBoostClassifier
        params = dict(
            iterations=self.kwargs.get("iterations", 50),
            depth=self.kwargs.get("depth", 4),
            learning_rate=self.kwargs.get("learning_rate", 0.1),
            random_seed=seed, verbose=False,
            allow_writing_files=False,
            thread_count=self.kwargs.get("n_jobs", 1),
        )
        student = CatBoostClassifier(**params).fit(
            X, y_distill, sample_weight=sample_weight,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.json")
            student.save_model(path, format="json")
            with open(path, "r") as f:
                model_json = json.load(f)
        trees = (
            model_json.get("oblivious_trees")
            or model_json.get("trees")
            or []
        )
        branches_per_tree: List[List[Branch]] = []
        leaf_ids_per_branch_per_tree: List[List[List[int]]] = []
        offset = 0
        for t, tree_dict in enumerate(trees):
            br_t = _catboost_branches_from_oblivious(
                tree_dict, tree_idx=t, branch_offset=offset,
                n_features=n_features, n_classes=n_classes,
            )
            depth = len(tree_dict.get("splits", []))
            leaf_ids_t = [
                _catboost_leaf_ids_for_parent(int(b.parent_node_id), depth)
                for b in br_t
            ]
            branches_per_tree.append(br_t)
            leaf_ids_per_branch_per_tree.append(leaf_ids_t)
            offset += len(br_t)
        Xr, yr = self._refine_X(X, y, seed)
        leaf_idx_matrix = np.asarray(
            student.calc_leaf_indexes(Xr), dtype=np.int64,
        )
        if leaf_idx_matrix.ndim == 1:
            leaf_idx_matrix = leaf_idx_matrix.reshape(-1, 1)
        n_ref = _empirical_cp_via_leaf_indices(
            branches_per_tree, leaf_ids_per_branch_per_tree,
            leaf_idx_matrix, yr, n_classes,
        )
        student_extra["n_branches_refined"] = n_ref
        student_extra["refinement_mode"] = "calc_leaf_indexes"
        student_extra["n_oblivious_trees"] = len(trees)
        return branches_per_tree, student, student_extra


class RuleFitRuleSource(RuleSource):
    name = "rulefit"
    supports_multiclass = False  # imodels.RuleFit raises on multiclass

    def _fit(self, X, y, *, n_features, n_classes, seed):
        from imodels import RuleFitClassifier

        params = dict(
            max_rules=self.kwargs.get("max_rules", 30),
            random_state=seed,
        )
        clf = RuleFitClassifier(**params)
        feat_names = [f"feature_{i}" for i in range(n_features)]
        clf.fit(X, y, feature_names=feat_names)

        rules_df = clf.rules_
        # In recent imodels, .rules_ is a DataFrame with columns
        # ['rule', 'coef', 'support', ...]; older versions returned a list.
        if hasattr(rules_df, "iterrows"):
            iter_rules = (
                (str(row["rule"]), float(row.get("coef", 0.0)))
                for _, row in rules_df.iterrows()
            )
        else:
            iter_rules = ((str(r), 0.0) for r in rules_df)

        branches_per_tree: List[List[Branch]] = [[]]
        offset = 0
        skipped = 0
        for rule_str, coef in iter_rules:
            conditions = _parse_rulefit_rule(rule_str, feat_names)
            if conditions is None:
                skipped += 1
                continue
            # Binary: positive coef pushes towards class 1.
            p1 = 1.0 / (1.0 + np.exp(-coef))
            cp = np.array([1.0 - p1, p1], dtype=np.float64)
            last = conditions[-1]
            branches_per_tree[0].append(Branch(
                branch_id=f"b{offset}",
                tree_id=0,
                parent_node_id=offset,
                conditions=conditions,
                class_proportions=cp.tolist(),
                split_feature_idx=last.feature_idx,
                split_threshold=last.threshold,
                split_node_id=offset,
            ))
            offset += 1
        Xr, yr = self._refine_X(X, y, seed)
        n_refined = _empirical_class_proportions(
            branches_per_tree[0], Xr, yr, n_classes,
        )
        return branches_per_tree, clf, {
            "n_rules_total": offset + skipped,
            "n_rules_unparseable": skipped,
            "n_branches_refined": n_refined,
            "refinement_mode": "vectorized_eval",
            "refinement_samples": int(Xr.shape[0]),
        }


# --------------------------------------------------------------------------- #
# Registry + factory
# --------------------------------------------------------------------------- #


class _TabPFNDistillXGB(TabPFNDistillRuleSource):
    name = "tabpfn_distill_xgb"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("student", "xgb")
        super().__init__(**kwargs)


class _TabPFNDistillET(TabPFNDistillRuleSource):
    name = "tabpfn_distill_et"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("student", "extratrees")
        super().__init__(**kwargs)


class _TabPFNDistillCB(TabPFNDistillRuleSource):
    name = "tabpfn_distill_cb"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("student", "catboost")
        super().__init__(**kwargs)


RULE_SOURCE_REGISTRY: Dict[str, type] = {
    "extratrees":         ExtraTreesRuleSource,
    "xgb":                XGBoostRuleSource,
    "catboost":           CatBoostRuleSource,
    "figs":               FIGSRuleSource,
    "rulefit":            RuleFitRuleSource,
    "tabpfn_distill_xgb": _TabPFNDistillXGB,
    "tabpfn_distill_et":  _TabPFNDistillET,
    "tabpfn_distill_cb":  _TabPFNDistillCB,
}


RULE_SOURCE_LABELS: Dict[str, str] = {
    "extratrees":         "ExtraTrees",
    "xgb":                "XGBoost",
    "catboost":           "CatBoost",
    "figs":               "FIGS",
    "rulefit":            "RuleFit",
    "tabpfn_distill_xgb": "TabPFN→XGB",
    "tabpfn_distill_et":  "TabPFN→ExtraTrees",
    "tabpfn_distill_cb":  "TabPFN→CatBoost",
}


def build_rule_source(name: str, **kwargs: Any) -> RuleSource:
    """Instantiate a registered rule source by short key."""
    if name not in RULE_SOURCE_REGISTRY:
        raise KeyError(
            f"unknown rule source {name!r}; "
            f"choose from {sorted(RULE_SOURCE_REGISTRY)}"
        )
    return RULE_SOURCE_REGISTRY[name](**kwargs)
