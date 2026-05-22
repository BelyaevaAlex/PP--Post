"""Temporal TabPFN-TS distillation helpers.

The public experiments in this module intentionally do *not* expose
``forecast/residual`` features as an interpretable PPtheta-Post input
space.  TabPFN-TS is used as a black-box teacher to produce soft labels;
tree students are then trained on ordinary L2/L3 temporal features, and
PPtheta-Post extracts symbolic branches from those students.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from branch_schema import Branch
from rule_network import extract_branches_from_sklearn_ensemble
from tabular.rule_sources import (
    _catboost_branches_from_oblivious,
    _catboost_leaf_ids_for_parent,
    _empirical_class_proportions,
    _empirical_cp_via_leaf_indices,
    _xgb_walk,
)

from .tabpfn_ts_teacher import TabPFNTSFeatureTeacher


def _ensure_proba_shape(proba: np.ndarray, n_classes: int, classes=None) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    if classes is None:
        full[:, : min(n_classes, proba.shape[1])] = proba[:, :n_classes]
    else:
        for col, cls in enumerate(classes):
            if 0 <= int(cls) < n_classes:
                full[:, int(cls)] = proba[:, col]
    row_sum = full.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0
    if np.any(bad):
        full[bad, :] = 1.0 / max(n_classes, 1)
        row_sum = full.sum(axis=1, keepdims=True)
    return full / row_sum


@dataclass
class TabPFNTSClassifierTeacher:
    """Black-box temporal teacher built from TabPFN-TS representation.

    ``TabPFNTSFeatureTeacher`` creates forecasting-derived representation
    features.  A classifier head then produces ``p(y | X_ts)`` soft labels
    used by distillation students or by the standalone black-box baseline.
    """

    n_classes: int = 2
    seed: int = 42
    ts_backend: str = "tabpfn_ts"
    ts_max_rows: int = 4096
    ts_model_path: Optional[str] = None
    ts_device: str = "cpu"
    ts_n_estimators: int = 8
    ts_num_workers: int = 1
    head: str = "tabpfn"
    classifier_model_path: Optional[str] = None
    classifier_device: str = "cpu"
    classifier_n_estimators: int = 8

    feature_teacher_: Optional[TabPFNTSFeatureTeacher] = field(default=None, init=False)
    scaler_: Optional[StandardScaler] = field(default=None, init=False)
    classifier_: Optional[Any] = field(default=None, init=False)
    head_used_: Optional[str] = field(default=None, init=False)

    def fit(
        self, X_ts: np.ndarray, mask: np.ndarray, y: np.ndarray,
    ) -> "TabPFNTSClassifierTeacher":
        self.feature_teacher_ = TabPFNTSFeatureTeacher(
            backend=self.ts_backend,
            seed=self.seed,
            max_regression_rows=self.ts_max_rows,
            tabpfn_estimators=self.ts_n_estimators,
            tabpfn_ts_model_path=self.ts_model_path,
            tabpfn_ts_device=self.ts_device,
            tabpfn_ts_num_workers=self.ts_num_workers,
        ).fit(X_ts, mask)
        X_feat = self.feature_teacher_.transform(X_ts, mask)
        self.scaler_ = StandardScaler().fit(X_feat)
        X_scaled = self.scaler_.transform(X_feat)
        self.classifier_ = self._fit_head(X_scaled, y)
        return self

    def transform(self, X_ts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.feature_teacher_ is None or self.scaler_ is None:
            raise RuntimeError("TabPFNTSClassifierTeacher must be fit first")
        return self.scaler_.transform(self.feature_teacher_.transform(X_ts, mask))

    def predict_proba(self, X_ts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("TabPFNTSClassifierTeacher must be fit first")
        X = self.transform(X_ts, mask)
        proba = self.classifier_.predict_proba(X)
        return _ensure_proba_shape(
            proba, self.n_classes, getattr(self.classifier_, "classes_", None),
        )

    @property
    def ts_backend_used(self) -> str:
        if self.feature_teacher_ is None:
            return self.ts_backend
        return self.feature_teacher_.backend_used or self.ts_backend

    def _fit_head(self, X: np.ndarray, y: np.ndarray):
        head = self.head.lower()
        self.head_used_ = head
        if head == "tabpfn":
            if str(self.classifier_device).lower() == "cpu":
                os.environ.setdefault("TABPFN_EXCLUDE_DEVICES", "mps")
            os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
            from tabpfn import TabPFNClassifier

            params = dict(
                device=self.classifier_device,
                n_estimators=self.classifier_n_estimators,
                random_state=self.seed,
                show_progress_bar=False,
            )
            model_path = (
                self.classifier_model_path
                or os.environ.get("TABPFN_CLASSIFIER_MODEL_PATH")
                or os.environ.get("TABPFN_MODEL_PATH")
            )
            if model_path:
                params["model_path"] = model_path
            sig = inspect.signature(TabPFNClassifier.__init__)
            clf = TabPFNClassifier(
                **{k: v for k, v in params.items() if k in sig.parameters}
            )
            return clf.fit(X, y)
        if head in ("xgb", "xgboost"):
            import xgboost as xgb

            objective = "binary:logistic" if self.n_classes == 2 else "multi:softprob"
            params = dict(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.seed,
                n_jobs=1,
                objective=objective,
                verbosity=0,
            )
            if self.n_classes > 2:
                params["num_class"] = self.n_classes
            return xgb.XGBClassifier(**params).fit(X, y)
        if head in ("extratrees", "et"):
            return ExtraTreesClassifier(
                n_estimators=128,
                max_leaf_nodes=32,
                random_state=self.seed,
                n_jobs=1,
            ).fit(X, y)
        if head in ("logreg", "lr"):
            return LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=self.seed,
            ).fit(X, y)
        raise ValueError(
            f"unknown TabPFN-TS classifier head {self.head!r}; "
            "choose tabpfn / xgb / extratrees / logreg"
        )


@dataclass
class FittedTemporalDistillStudent:
    student: str
    branches_per_tree: List[List[Branch]]
    native_model: Any
    extra: Dict[str, Any] = field(default_factory=dict)


def student_label(student: str) -> str:
    labels = {
        "xgb": "XGB",
        "xgboost": "XGB",
        "et": "ET",
        "extratrees": "ET",
        "cb": "CB",
        "catboost": "CB",
    }
    return labels.get(student.lower(), student)


def _global_class_proportions(y_true: np.ndarray, n_classes: int) -> List[float]:
    counts = np.bincount(y_true.astype(np.int64), minlength=n_classes).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return [1.0 / max(n_classes, 1)] * n_classes
    return (counts / total).tolist()


def _ensure_branch_class_width(
    branches_per_tree: List[List[Branch]],
    y_true: np.ndarray,
    n_classes: int,
) -> int:
    """Make extracted branch priors compatible with PPtheta-Post heads."""
    fallback = _global_class_proportions(y_true, n_classes)
    n_fixed = 0
    for branches in branches_per_tree:
        for br in branches:
            cp = br.class_proportions
            if cp is None or len(cp) != n_classes:
                br.class_proportions = list(fallback)
                n_fixed += 1
                continue
            arr = np.asarray(cp, dtype=np.float64)
            total = float(arr.sum())
            if not np.isfinite(total) or total <= 0:
                br.class_proportions = list(fallback)
                n_fixed += 1
            else:
                br.class_proportions = (arr / total).tolist()
    return n_fixed


def fit_distilled_rule_student(
    X: np.ndarray,
    y_true: np.ndarray,
    p_soft: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    student: str,
    n_estimators: int = 32,
    max_leaf_nodes: int = 32,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> FittedTemporalDistillStudent:
    """Fit XGB/ET/CB rule student from teacher soft labels."""
    key = student.lower()
    if key == "xgboost":
        key = "xgb"
    if key == "extratrees":
        key = "et"
    if key == "catboost":
        key = "cb"
    if key not in ("xgb", "et", "cb"):
        raise ValueError("student must be one of xgb / et / cb")

    p_soft = _ensure_proba_shape(p_soft, n_classes)
    y_distill = p_soft.argmax(axis=1).astype(np.int64)
    sample_weight = np.clip(
        p_soft.max(axis=1).astype(np.float64),
        1.0 / max(n_classes, 2),
        None,
    )

    extra: Dict[str, Any] = {
        "teacher_confidence_mean": float(sample_weight.mean()),
        "distill_student": key,
    }

    if key == "et":
        student_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_leaf_nodes=max_leaf_nodes,
            random_state=seed,
            n_jobs=1,
        ).fit(X, y_distill, sample_weight=sample_weight)
        branches_per_tree = extract_branches_from_sklearn_ensemble(student_model)
        n_ref = sum(
            _empirical_class_proportions(br, X, y_true, n_classes)
            for br in branches_per_tree
        )
        n_fixed = _ensure_branch_class_width(branches_per_tree, y_true, n_classes)
        extra["n_branches_refined"] = n_ref
        extra["n_branch_priors_fixed"] = n_fixed
        extra["refinement_mode"] = "vectorized_eval"
        return FittedTemporalDistillStudent(key, branches_per_tree, student_model, extra)

    if key == "xgb":
        import xgboost as xgb

        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        if n_classes > 2:
            params.update(objective="multi:softprob", num_class=n_classes)
        else:
            params.update(objective="binary:logistic")
        student_model = xgb.XGBClassifier(**params).fit(
            X, y_distill, sample_weight=sample_weight,
        )
        booster = student_model.get_booster()
        dump = booster.get_dump(dump_format="json")
        branches_per_tree: List[List[Branch]] = []
        leaf_ids_per_branch_per_tree: List[List[List[int]]] = []
        offset = 0
        for t, tree_json in enumerate(dump):
            br_t: List[Branch] = []
            leaf_ids_t: List[List[int]] = []
            _xgb_walk(
                json.loads(tree_json),
                n_features=X.shape[1],
                n_classes=n_classes,
                tree_idx=t,
                branch_offset=offset,
                path=[],
                out=br_t,
                leaf_ids_per_branch=leaf_ids_t,
            )
            branches_per_tree.append(br_t)
            leaf_ids_per_branch_per_tree.append(leaf_ids_t)
            offset += len(br_t)
        leaf_idx_matrix = np.asarray(
            booster.predict(xgb.DMatrix(X), pred_leaf=True), dtype=np.int64,
        )
        if leaf_idx_matrix.ndim == 1:
            leaf_idx_matrix = leaf_idx_matrix.reshape(-1, 1)
        n_ref = _empirical_cp_via_leaf_indices(
            branches_per_tree, leaf_ids_per_branch_per_tree,
            leaf_idx_matrix, y_true, n_classes,
        )
        extra.update(
            n_boosters=len(dump),
            n_branches_refined=n_ref,
            n_branch_priors_fixed=_ensure_branch_class_width(
                branches_per_tree, y_true, n_classes,
            ),
            refinement_mode="pred_leaf",
        )
        return FittedTemporalDistillStudent(key, branches_per_tree, student_model, extra)

    from catboost import CatBoostClassifier

    params = dict(
        iterations=n_estimators,
        depth=max_depth,
        learning_rate=learning_rate,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    student_model = CatBoostClassifier(**params).fit(
        X, y_distill, sample_weight=sample_weight,
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.json")
        student_model.save_model(path, format="json")
        with open(path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

    trees = model_json.get("oblivious_trees") or model_json.get("trees") or []
    branches_per_tree = []
    leaf_ids_per_branch_per_tree = []
    offset = 0
    for t, tree_dict in enumerate(trees):
        br_t = _catboost_branches_from_oblivious(
            tree_dict,
            tree_idx=t,
            branch_offset=offset,
            n_features=X.shape[1],
            n_classes=n_classes,
        )
        depth = len(tree_dict.get("splits", []))
        leaf_ids_t = [
            _catboost_leaf_ids_for_parent(int(b.parent_node_id), depth)
            for b in br_t
        ]
        branches_per_tree.append(br_t)
        leaf_ids_per_branch_per_tree.append(leaf_ids_t)
        offset += len(br_t)
    leaf_idx_matrix = np.asarray(
        student_model.calc_leaf_indexes(X), dtype=np.int64,
    )
    if leaf_idx_matrix.ndim == 1:
        leaf_idx_matrix = leaf_idx_matrix.reshape(-1, 1)
    n_ref = _empirical_cp_via_leaf_indices(
        branches_per_tree, leaf_ids_per_branch_per_tree,
        leaf_idx_matrix, y_true, n_classes,
    )
    extra.update(
        n_oblivious_trees=len(trees),
        n_branches_refined=n_ref,
        n_branch_priors_fixed=_ensure_branch_class_width(
            branches_per_tree, y_true, n_classes,
        ),
        refinement_mode="calc_leaf_indexes",
    )
    return FittedTemporalDistillStudent(key, branches_per_tree, student_model, extra)
