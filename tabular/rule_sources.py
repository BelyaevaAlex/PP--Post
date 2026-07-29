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


def _default_tabpfn_device() -> str:
    requested = os.environ.get("TABPFN_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

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
# EBM terms — turn additive glass-box bins into explicit Branch objects
# --------------------------------------------------------------------------- #


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _sigmoid01(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(float(x), -50.0, 50.0))))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values - np.max(values)
    ex = np.exp(np.clip(values, -50.0, 50.0))
    s = ex.sum()
    return ex / s if s > 0 else np.ones_like(ex) / max(len(ex), 1)


def _ebm_score_to_class_proportions(score: Any, n_classes: int) -> List[float]:
    arr = np.asarray(score, dtype=np.float64).reshape(-1)
    if n_classes == 2 and arr.size == 1:
        p1 = _sigmoid01(float(arr[0]))
        return [1.0 - p1, p1]
    if arr.size >= n_classes:
        probs = _softmax(arr[:n_classes])
        return probs.tolist()
    out = np.ones(n_classes, dtype=np.float64) / max(n_classes, 1)
    if n_classes == 2 and arr.size:
        p1 = _sigmoid01(float(arr[0]))
        out = np.array([1.0 - p1, p1], dtype=np.float64)
    return out.tolist()


def _ebm_cuts_for_feature(clf: Any, feature_idx: int, level: int) -> Optional[np.ndarray]:
    bins = getattr(clf, "bins_", None)
    if bins is None or feature_idx >= len(bins):
        return None
    feature_bins = bins[feature_idx]
    if not feature_bins:
        return None
    raw = feature_bins[min(int(level), len(feature_bins) - 1)]
    # Continuous features expose numeric cut arrays.  Categorical bins are
    # dictionaries in interpret; mortality preprocessing is numeric, so skip
    # unsupported categorical encodings instead of inventing opaque rules.
    if isinstance(raw, dict):
        return None
    try:
        cuts = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    cuts = cuts[np.isfinite(cuts)]
    return np.sort(cuts)


def _ebm_conditions_for_bin(
    feature_idx: int,
    cuts: np.ndarray,
    bin_idx: int,
    n_bins: int,
    node_base: int,
) -> Optional[List[Condition]]:
    # interpret reserves bin 0 for missing and the final bin for unknown/other.
    if bin_idx <= 0 or bin_idx >= n_bins - 1:
        return None
    cuts = np.asarray(cuts, dtype=np.float64).reshape(-1)
    non_special_bins = n_bins - 2
    if non_special_bins <= 0:
        return None
    conditions: List[Condition] = []
    if cuts.size == 0:
        return conditions
    if bin_idx == 1:
        conditions.append(Condition(
            feature_idx=feature_idx,
            threshold=float(cuts[0]),
            direction="le",
            node_id=node_base,
        ))
    elif bin_idx == non_special_bins:
        conditions.append(Condition(
            feature_idx=feature_idx,
            threshold=float(cuts[-1]),
            direction="gt",
            node_id=node_base,
        ))
    else:
        lo = float(cuts[bin_idx - 2])
        hi = float(cuts[bin_idx - 1])
        conditions.append(Condition(
            feature_idx=feature_idx,
            threshold=lo,
            direction="gt",
            node_id=node_base,
        ))
        conditions.append(Condition(
            feature_idx=feature_idx,
            threshold=hi,
            direction="le",
            node_id=node_base + 1,
        ))
    return conditions


class EBMTermsRuleSource(RuleSource):
    """Expose EBM additive terms as explicit PPtheta-Post rule branches.

    Each non-special EBM bin (main effect) or interaction cell becomes one
    :class:`Branch` with interval conditions.  The EBM model remains the native
    predictor for ``source_native``; PPtheta-Post variants operate only over the
    extracted symbolic bins/cells.
    """

    name = "ebm_terms"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as e:
            raise ImportError(
                "interpret is not installed. Install with `pip install interpret` "
                "or activate the per-baseline venv."
            ) from e

        params = dict(
            max_bins=self.kwargs.get("max_bins", _env_int("EBM_MAX_BINS", 256)),
            max_interaction_bins=self.kwargs.get(
                "max_interaction_bins", _env_int("EBM_MAX_INTERACTION_BINS", 32)
            ),
            interactions=self.kwargs.get("interactions", _env_int("EBM_INTERACTIONS", 10)),
            outer_bags=self.kwargs.get("outer_bags", _env_int("EBM_OUTER_BAGS", 8)),
            random_state=seed,
        )
        clf = ExplainableBoostingClassifier(**params).fit(X, y)

        min_support = float(self.kwargs.get(
            "min_support", _env_float("EBM_TERMS_MIN_SUPPORT", 1.0)
        ))
        max_branches = int(self.kwargs.get(
            "max_branches", _env_int("EBM_TERMS_MAX_BRANCHES", 2048)
        ))

        candidates: List[Tuple[float, int, Tuple[int, ...], List[Condition], List[float], float]] = []
        term_features = list(getattr(clf, "term_features_", []))
        term_scores = list(getattr(clf, "term_scores_", []))
        bin_weights = list(getattr(clf, "bin_weights_", []))

        for term_idx, features_raw in enumerate(term_features):
            features = tuple(int(f) for f in features_raw)
            if not features or term_idx >= len(term_scores):
                continue
            scores = np.asarray(term_scores[term_idx])
            bin_shape = tuple(int(v) for v in scores.shape[:len(features)])
            if len(bin_shape) != len(features) or any(v <= 2 for v in bin_shape):
                continue
            weights = None
            if term_idx < len(bin_weights):
                try:
                    weights = np.asarray(bin_weights[term_idx], dtype=np.float64)
                except Exception:  # noqa: BLE001
                    weights = None
            level = max(0, len(features) - 1)
            cuts_by_feature = [
                _ebm_cuts_for_feature(clf, feat, level) for feat in features
            ]
            if any(cuts is None for cuts in cuts_by_feature):
                continue

            for bin_indices in np.ndindex(*bin_shape):
                all_conditions: List[Condition] = []
                skip = False
                for dim, (feat, bin_idx, n_bins, cuts) in enumerate(
                    zip(features, bin_indices, bin_shape, cuts_by_feature)
                ):
                    conds = _ebm_conditions_for_bin(
                        feat,
                        cuts,
                        int(bin_idx),
                        int(n_bins),
                        node_base=term_idx * 1_000_000 + dim * 10_000 + int(bin_idx) * 10,
                    )
                    if conds is None:
                        skip = True
                        break
                    all_conditions.extend(conds)
                if skip or not all_conditions:
                    continue

                support = 0.0
                if weights is not None and weights.shape[:len(features)] == bin_shape:
                    try:
                        support = float(weights[bin_indices])
                    except Exception:  # noqa: BLE001
                        support = 0.0
                if support < min_support:
                    continue

                cell_score = scores[bin_indices]
                cp = _ebm_score_to_class_proportions(cell_score, n_classes)
                score_mag = float(np.linalg.norm(np.asarray(cell_score, dtype=np.float64).reshape(-1)))
                priority = score_mag * np.log1p(max(support, 0.0))
                candidates.append((priority, term_idx, tuple(int(i) for i in bin_indices), all_conditions, cp, support))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        total_candidates = len(candidates)
        if max_branches > 0:
            candidates = candidates[:max_branches]
        candidates.sort(key=lambda item: (item[1], item[2]))

        branches_by_term: Dict[int, List[Branch]] = {}
        offset = 0
        for _, term_idx, bin_indices, conditions, cp, support in candidates:
            tree_id = int(term_idx)
            last = conditions[-1]
            branches_by_term.setdefault(tree_id, []).append(Branch(
                branch_id=f"b{offset}",
                tree_id=tree_id,
                parent_node_id=offset,
                conditions=list(conditions),
                class_proportions=cp,
                split_feature_idx=last.feature_idx,
                split_threshold=last.threshold,
                split_node_id=last.node_id,
            ))
            offset += 1

        branches_per_tree = [branches_by_term.get(i, []) for i in range(len(term_features))]
        Xr, yr = self._refine_X(X, y, seed)
        n_refined = sum(
            _empirical_class_proportions(br, Xr, yr, n_classes)
            for br in branches_per_tree
            if br
        )
        return branches_per_tree, clf, {
            "estimator": "ExplainableBoostingClassifier",
            "n_ebm_terms": len(term_features),
            "n_candidate_bins": int(total_candidates),
            "n_branches_selected": int(offset),
            "n_branches_refined": int(n_refined),
            "max_branches": int(max_branches),
            "min_support": float(min_support),
            "refinement_mode": "vectorized_eval",
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
        device = tabpfn_kwargs.get("device") or _default_tabpfn_device()
        if str(device).lower() == "cpu":
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
        tabpfn_kwargs.setdefault("device", device)
        tabpfn_kwargs.setdefault("random_state", seed)
        model_path = (
            tabpfn_kwargs.get("model_path")
            or os.environ.get("TABPFN_CLASSIFIER_MODEL_PATH")
            or os.environ.get("TABPFN_MODEL_PATH")
        )
        if model_path:
            tabpfn_kwargs["model_path"] = model_path
        tabpfn_kwargs.setdefault("show_progress_bar", False)
        tabpfn_kwargs.setdefault(
            "ignore_pretraining_limits",
            _env_bool("TABPFN_IGNORE_PRETRAINING_LIMITS", False),
        )
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

        distill_target = str(self.kwargs.get("distill_target", "hard_conf")).lower()
        if distill_target in {"soft_true", "soft", "mixed"}:
            true_weight = float(self.kwargs.get("true_label_weight", 0.50))
            true_weight = float(np.clip(true_weight, 0.0, 1.0))
            y_onehot = np.zeros((len(y), n_classes), dtype=np.float64)
            y_onehot[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
            p_mix = true_weight * y_onehot + (1.0 - true_weight) * p_soft[:, :n_classes]
            p_mix = np.clip(p_mix, 1e-6, 1.0)
            classes = np.arange(n_classes, dtype=np.int64)
            X_student = np.repeat(X, n_classes, axis=0)
            y_student = np.tile(classes, len(y)).astype(np.int64)
            sample_weight = p_mix.reshape(-1).astype(np.float64)
            sample_weight = sample_weight / np.maximum(sample_weight.mean(), 1e-12)
            tabpfn_confidence_mean = float(p_soft.max(axis=1).mean())
        else:
            y_student = p_soft.argmax(axis=1).astype(np.int64)
            sample_weight = p_soft.max(axis=1).astype(np.float64)
            # Stabilise: drop near-zero weights so the student doesn't see
            # essentially-uniform rows that would shrink branches to noise.
            sample_weight = np.clip(sample_weight, 1.0 / max(n_classes, 2), None)
            X_student = X
            tabpfn_confidence_mean = float(sample_weight.mean())

        # 2. Student: route to one of the existing rule sources, but with
        #    distilled labels.  We do not call the registered _fit
        #    directly because we need sample_weight support and a custom
        #    y; instead we replicate the minimal training path inline.
        student_extra: Dict[str, Any] = {
            "tabpfn_n": int(X.shape[0]),
            "tabpfn_confidence_mean": tabpfn_confidence_mean,
            "distill_student": student_key,
            "distill_target": distill_target,
            # Kept in-memory only: delta variants use the real TabPFN teacher
            # probabilities for auditable posterior distillation.
            "tabpfn_teacher_model": teacher,
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
                X_student, y_student, sample_weight=sample_weight,
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
                X_student, y_student, sample_weight=sample_weight,
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
            X_student, y_student, sample_weight=sample_weight,
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


class _TabPFNDistillXGBSoft(TabPFNDistillRuleSource):
    name = "tabpfn_distill_xgb_soft"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("student", "xgb")
        kwargs.setdefault("distill_target", "soft_true")
        kwargs.setdefault("true_label_weight", 0.50)
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




def _extract_ebm_term_branches(
    clf: Any,
    X_refine: np.ndarray,
    y_refine: np.ndarray,
    n_classes: int,
    *,
    max_branches: int,
    min_support: float,
) -> Tuple[List[List[Branch]], Dict[str, Any]]:
    """Extract EBM bins/cells as Branch objects and refine cp on real labels."""
    candidates: List[Tuple[float, int, Tuple[int, ...], List[Condition], List[float], float]] = []
    term_features = list(getattr(clf, "term_features_", []))
    term_scores = list(getattr(clf, "term_scores_", []))
    bin_weights = list(getattr(clf, "bin_weights_", []))

    for term_idx, features_raw in enumerate(term_features):
        features = tuple(int(f) for f in features_raw)
        if not features or term_idx >= len(term_scores):
            continue
        scores = np.asarray(term_scores[term_idx])
        bin_shape = tuple(int(v) for v in scores.shape[:len(features)])
        if len(bin_shape) != len(features) or any(v <= 2 for v in bin_shape):
            continue
        weights = None
        if term_idx < len(bin_weights):
            try:
                weights = np.asarray(bin_weights[term_idx], dtype=np.float64)
            except Exception:  # noqa: BLE001
                weights = None
        level = max(0, len(features) - 1)
        cuts_by_feature = [_ebm_cuts_for_feature(clf, feat, level) for feat in features]
        if any(cuts is None for cuts in cuts_by_feature):
            continue
        for bin_indices in np.ndindex(*bin_shape):
            all_conditions: List[Condition] = []
            skip = False
            for dim, (feat, bin_idx, n_bins, cuts) in enumerate(zip(features, bin_indices, bin_shape, cuts_by_feature)):
                conds = _ebm_conditions_for_bin(
                    feat, cuts, int(bin_idx), int(n_bins),
                    node_base=term_idx * 1_000_000 + dim * 10_000 + int(bin_idx) * 10,
                )
                if conds is None:
                    skip = True
                    break
                all_conditions.extend(conds)
            if skip or not all_conditions:
                continue
            support = 0.0
            if weights is not None and weights.shape[:len(features)] == bin_shape:
                try:
                    support = float(weights[bin_indices])
                except Exception:  # noqa: BLE001
                    support = 0.0
            if support < min_support:
                continue
            cell_score = scores[bin_indices]
            cp = _ebm_score_to_class_proportions(cell_score, n_classes)
            score_mag = float(np.linalg.norm(np.asarray(cell_score, dtype=np.float64).reshape(-1)))
            priority = score_mag * np.log1p(max(support, 0.0))
            candidates.append((priority, term_idx, tuple(int(i) for i in bin_indices), all_conditions, cp, support))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    total_candidates = len(candidates)
    if max_branches > 0:
        candidates = candidates[:max_branches]
    candidates.sort(key=lambda item: (item[1], item[2]))

    branches_by_term: Dict[int, List[Branch]] = {}
    offset = 0
    for _, term_idx, _bin_indices, conditions, cp, _support in candidates:
        last = conditions[-1]
        branches_by_term.setdefault(int(term_idx), []).append(Branch(
            branch_id=f"b{offset}",
            tree_id=int(term_idx),
            parent_node_id=offset,
            conditions=list(conditions),
            class_proportions=cp,
            split_feature_idx=last.feature_idx,
            split_threshold=last.threshold,
            split_node_id=last.node_id,
        ))
        offset += 1

    branches_per_tree = [branches_by_term.get(i, []) for i in range(len(term_features))]
    n_refined = sum(
        _empirical_class_proportions(br, X_refine, y_refine, n_classes)
        for br in branches_per_tree
        if br
    )
    return branches_per_tree, {
        "n_ebm_terms": len(term_features),
        "n_candidate_bins": int(total_candidates),
        "n_branches_selected": int(offset),
        "n_branches_refined": int(n_refined),
        "max_branches": int(max_branches),
        "min_support": float(min_support),
        "refinement_mode": "vectorized_eval",
        "refinement_samples": int(X_refine.shape[0]),
    }


class TabPFNDistillEBMTermsRuleSource(RuleSource):
    """Distil TabPFN into an EBM/GA2M-style student, then expose EBM terms.

    TabPFN is used only on the training fold. The deployed rule source is an
    ExplainableBoostingClassifier whose additive bins and interaction cells are
    converted into PPtheta-Post evidence branches and re-refined on true labels.
    """

    name = "tabpfn_distill_ebm_terms"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError("tabpfn is not installed for TabPFN-to-EBM distillation") from e
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as e:
            raise ImportError("interpret is not installed for EBM distillation") from e

        tabpfn_kwargs = dict(self.kwargs.get("tabpfn_kwargs", {}))
        tabpfn_kwargs.setdefault("device", _default_tabpfn_device())
        tabpfn_kwargs.setdefault("random_state", seed)
        tabpfn_kwargs.setdefault("show_progress_bar", False)
        tabpfn_kwargs.setdefault("ignore_pretraining_limits", _env_bool("TABPFN_IGNORE_PRETRAINING_LIMITS", False))
        model_path = os.environ.get("TABPFN_CLASSIFIER_MODEL_PATH") or os.environ.get("TABPFN_MODEL_PATH")
        if model_path:
            tabpfn_kwargs["model_path"] = model_path
        import inspect
        sig = inspect.signature(TabPFNClassifier.__init__)
        teacher = TabPFNClassifier(**{k: v for k, v in tabpfn_kwargs.items() if k in sig.parameters})
        teacher.fit(X, y)
        p_soft = np.asarray(teacher.predict_proba(X), dtype=np.float64)
        if p_soft.ndim == 1:
            p_soft = np.column_stack([1.0 - p_soft, p_soft])
        if p_soft.shape[1] < n_classes:
            full = np.zeros((p_soft.shape[0], n_classes), dtype=np.float64)
            full[:, :p_soft.shape[1]] = p_soft
            p_soft = full
        true_weight = float(np.clip(self.kwargs.get("true_label_weight", 0.35), 0.0, 1.0))
        y_onehot = np.zeros((len(y), n_classes), dtype=np.float64)
        y_onehot[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
        p_mix = true_weight * y_onehot + (1.0 - true_weight) * p_soft[:, :n_classes]
        y_student = p_mix.argmax(axis=1).astype(np.int64)
        sample_weight = np.clip(p_mix.max(axis=1), 1.0 / max(n_classes, 2), None)
        sample_weight = sample_weight / np.maximum(sample_weight.mean(), 1e-12)

        params = dict(
            max_bins=self.kwargs.get("max_bins", _env_int("EBM_MAX_BINS", 256)),
            max_interaction_bins=self.kwargs.get("max_interaction_bins", _env_int("EBM_MAX_INTERACTION_BINS", 32)),
            interactions=self.kwargs.get("interactions", _env_int("TABPFN_EBM_INTERACTIONS", _env_int("EBM_INTERACTIONS", 10))),
            outer_bags=self.kwargs.get("outer_bags", _env_int("EBM_OUTER_BAGS", 8)),
            random_state=seed,
        )
        clf = ExplainableBoostingClassifier(**params).fit(X, y_student, sample_weight=sample_weight)
        min_support = float(self.kwargs.get("min_support", _env_float("EBM_TERMS_MIN_SUPPORT", 1.0)))
        max_branches = int(self.kwargs.get("max_branches", _env_int("EBM_TERMS_MAX_BRANCHES", 2048)))
        Xr, yr = self._refine_X(X, y, seed)
        branches_per_tree, meta = _extract_ebm_term_branches(
            clf, Xr, yr, n_classes, max_branches=max_branches, min_support=min_support,
        )
        meta.update({
            "estimator": "TabPFNDistilledExplainableBoostingClassifier",
            "tabpfn_confidence_mean": float(p_soft.max(axis=1).mean()),
            "true_label_weight": float(true_weight),
            "tabpfn_teacher_model": teacher,
        })
        return branches_per_tree, clf, meta

class _ClinicalMonotoneNative:
    """Simple native scorer for clinical monotone rule families."""

    def __init__(self, branches: List[Branch], class_prior: np.ndarray, n_classes: int) -> None:
        self.branches = branches
        self.class_prior = np.asarray(class_prior, dtype=np.float64)
        self.n_classes = int(n_classes)
        self.classes_ = np.arange(self.n_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        prior = np.clip(self.class_prior, 1e-9, 1.0)
        prior = prior / np.maximum(prior.sum(), 1e-12)
        out = np.tile(prior, (X.shape[0], 1))
        if not self.branches:
            return out
        votes = np.zeros_like(out)
        counts = np.zeros(X.shape[0], dtype=np.float64)
        for branch in self.branches:
            mask = np.ones(X.shape[0], dtype=bool)
            for cond in branch.conditions:
                col = X[:, int(cond.feature_idx)]
                if cond.direction == "le":
                    mask &= col <= float(cond.threshold)
                else:
                    mask &= col > float(cond.threshold)
                if not mask.any():
                    break
            if not mask.any():
                continue
            cp = np.asarray(branch.class_proportions, dtype=np.float64)
            if cp.size != self.n_classes:
                cp = prior
            cp = np.clip(cp, 1e-9, 1.0)
            cp = cp / np.maximum(cp.sum(), 1e-12)
            votes[mask] += cp
            counts[mask] += 1.0
        active = counts > 0
        out[active] = votes[active] / counts[active, None]
        return out


class ClinicalMonotoneRuleSource(RuleSource):
    """Build compact monotone clinical threshold families from the train fold.

    The source is intentionally model-light: it chooses features whose values
    have a stable univariate direction with the mortality label, creates
    risk-direction threshold rules, and adds a small set of same-direction
    two-feature conjunctions. Class proportions are then re-estimated from
    the original labels, so downstream PPtheta-Post sees explicit clinical
    evidence objects without a black-box teacher at inference time.
    """

    name = "clinical_monotone"

    def _fit(self, X, y, *, n_features, n_classes, seed):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            prior = np.ones(n_classes, dtype=np.float64) / max(n_classes, 1)
            return [[]], _ClinicalMonotoneNative([], prior, n_classes), {
                "n_candidate_rules": 0,
                "n_branches_refined": 0,
            }

        max_features = int(self.kwargs.get(
            "max_features", _env_int("CLINICAL_MONOTONE_MAX_FEATURES", 32)
        ))
        max_interactions = int(self.kwargs.get(
            "max_interactions", _env_int("CLINICAL_MONOTONE_MAX_INTERACTIONS", 32)
        ))
        max_branches = int(self.kwargs.get(
            "max_branches", _env_int("CLINICAL_MONOTONE_MAX_BRANCHES", 256)
        ))
        min_support = float(self.kwargs.get(
            "min_support", _env_float("CLINICAL_MONOTONE_MIN_SUPPORT", 0.01)
        ))
        quantiles = self.kwargs.get("quantiles", None)
        if quantiles is None:
            quantiles = (0.50, 0.75, 0.90)

        if n_classes == 2:
            y_bin = (y == 1).astype(np.float64)
        else:
            y_bin = (y == int(np.bincount(y, minlength=n_classes).argmax())).astype(np.float64)
        y_center = y_bin - y_bin.mean()
        scores: List[Tuple[float, int, int]] = []
        for j in range(n_features):
            col = X[:, j]
            finite = np.isfinite(col)
            if finite.sum() < 5:
                continue
            xj = col[finite]
            if np.nanstd(xj) < 1e-9:
                continue
            yc = y_center[finite]
            corr = float(np.nan_to_num(np.corrcoef(xj, yc)[0, 1], nan=0.0))
            if abs(corr) <= 1e-8:
                continue
            scores.append((abs(corr), j, 1 if corr >= 0 else -1))
        scores.sort(reverse=True)
        selected = scores[:max(1, min(max_features, len(scores)))]

        branches: List[Branch] = []
        tree_id = 0
        candidate_count = 0

        def add_branch(conditions: List[Condition]) -> None:
            nonlocal candidate_count, tree_id
            candidate_count += 1
            if max_branches > 0 and len(branches) >= max_branches:
                return
            mask = np.ones(X.shape[0], dtype=bool)
            for cond in conditions:
                col = X[:, int(cond.feature_idx)]
                if cond.direction == "le":
                    mask &= col <= float(cond.threshold)
                else:
                    mask &= col > float(cond.threshold)
            support = float(mask.mean())
            if support < min_support or not mask.any():
                return
            counts = np.bincount(y[mask], minlength=n_classes).astype(np.float64) + 1e-3
            cp = counts / counts.sum()
            last = conditions[-1]
            branches.append(Branch(
                branch_id=f"b{len(branches)}",
                tree_id=tree_id,
                parent_node_id=len(branches),
                conditions=list(conditions),
                class_proportions=cp.tolist(),
                split_feature_idx=last.feature_idx,
                split_threshold=last.threshold,
                split_node_id=last.node_id,
            ))
            tree_id += 1

        for _, feat, sign in selected:
            qs = quantiles if sign >= 0 else tuple(1.0 - float(q) for q in quantiles)
            for q in qs:
                thr = float(np.nanquantile(X[:, feat], q))
                direction = "gt" if sign >= 0 else "le"
                add_branch([Condition(feat, thr, direction, node_id=len(branches))])

        pairs = []
        for a_idx, (_, fa, sa) in enumerate(selected):
            for _, fb, sb in selected[a_idx + 1:]:
                if len(pairs) >= max_interactions:
                    break
                pairs.append((fa, sa, fb, sb))
            if len(pairs) >= max_interactions:
                break
        for fa, sa, fb, sb in pairs:
            qa = 0.75 if sa >= 0 else 0.25
            qb = 0.75 if sb >= 0 else 0.25
            ta = float(np.nanquantile(X[:, fa], qa))
            tb = float(np.nanquantile(X[:, fb], qb))
            add_branch([
                Condition(fa, ta, "gt" if sa >= 0 else "le", node_id=len(branches) * 10),
                Condition(fb, tb, "gt" if sb >= 0 else "le", node_id=len(branches) * 10 + 1),
            ])

        branches_per_tree = [[b] for b in branches]
        Xr, yr = self._refine_X(X, y, seed)
        n_refined = sum(
            _empirical_class_proportions(br, Xr, yr, n_classes)
            for br in branches_per_tree
        )
        prior = np.bincount(y, minlength=n_classes).astype(np.float64) + 1e-3
        prior = prior / prior.sum()
        native = _ClinicalMonotoneNative(branches, prior, n_classes)
        return branches_per_tree, native, {
            "n_candidate_rules": int(candidate_count),
            "n_selected_features": int(len(selected)),
            "n_selected_interactions": int(len(pairs)),
            "n_branches_refined": int(n_refined),
            "max_branches": int(max_branches),
            "min_support": float(min_support),
            "refinement_mode": "vectorized_eval",
            "refinement_samples": int(Xr.shape[0]),
        }


RULE_SOURCE_REGISTRY: Dict[str, type] = {
    "extratrees":         ExtraTreesRuleSource,
    "xgb":                XGBoostRuleSource,
    "catboost":           CatBoostRuleSource,
    "figs":               FIGSRuleSource,
    "rulefit":            RuleFitRuleSource,
    "ebm_terms":          EBMTermsRuleSource,
    "clinical_monotone":  ClinicalMonotoneRuleSource,
    "tabpfn_distill_ebm_terms": TabPFNDistillEBMTermsRuleSource,
    "tabpfn_distill_xgb": _TabPFNDistillXGB,
    "tabpfn_distill_xgb_soft": _TabPFNDistillXGBSoft,
    "tabpfn_distill_et":  _TabPFNDistillET,
    "tabpfn_distill_cb":  _TabPFNDistillCB,
}


RULE_SOURCE_LABELS: Dict[str, str] = {
    "extratrees":         "ExtraTrees",
    "xgb":                "XGBoost",
    "catboost":           "CatBoost",
    "figs":               "FIGS",
    "rulefit":            "RuleFit",
    "ebm_terms":          "EBM-Terms",
    "clinical_monotone":  "Clinical-Monotone",
    "tabpfn_distill_ebm_terms": "TabPFN→EBM-Terms",
    "tabpfn_distill_xgb": "TabPFN→XGB",
    "tabpfn_distill_xgb_soft": "TabPFN→XGB-Soft",
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
