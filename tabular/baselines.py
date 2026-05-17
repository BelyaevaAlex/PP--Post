"""Standalone tabular baselines for the PPtheta-Post benchmark.

These are competitors that do NOT feed the PPtheta-Post inference pipeline
— they are run end-to-end on (X, y) and their predict_proba is reported
side-by-side with the PL-* / PPtheta-Post variants.

Categories:
    * **EBM** — :class:`EBMBaseline` (interpret.glassbox); the current SOTA
      glass-box model on tabular data.
    * **FIGS** — :class:`FIGSBaseline` (imodels); fast interpretable greedy
      sums of trees, also exposed as a rule source in
      :mod:`tabular.rule_sources`.
    * **RuleFit** — :class:`RuleFitBaseline` (imodels); rule-ensemble +
      linear model; binary classification only (imodels.RuleFit raises on
      multiclass and we propagate that as ``NotImplementedError`` so the
      driver surfaces a clean ``[skip]``).
    * **TabPFN** — :class:`TabPFNBaseline` (tabpfn); prior-fitted
      transformer for small tabular datasets.  v1 enforces ``n ≤ 1000``
      and ``d ≤ 100``; v2 lifts those caps but downloads a ~200 MB HF
      checkpoint on first use.  The adapter raises ``NotImplementedError``
      with the offending shape so the driver can skip cleanly.

All adapters share the contract::

    base = make_tabular_baseline(name, **kwargs)
    base.fit(X_train, y_train, n_classes=K, seed=42)
    proba = base.predict_proba(X_test)      # (N, K)

This is intentionally narrower than :class:`tabular.rule_sources.RuleSource`
— baselines don't emit ``Branch`` objects and the driver consumes only
their ``predict_proba``.

Imports are lazy: each adapter calls ``import …`` inside ``fit`` and
raises a clear ``ImportError`` instructing how to install the package.
If pip install fails outright on a specific machine (numpy/torch version
clashes, missing system libs), the per-baseline venv fallback in
:mod:`tabular.baselines_vendored` is the next step — see that module for
the bootstrap convention copied from :mod:`temporal.baselines_vendored`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


@dataclass
class FittedTabularBaseline:
    name: str
    model: Any
    fit_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


class TabularBaselineBase(ABC):
    name: str = "abstract"
    supports_multiclass: bool = True

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._fitted: FittedTabularBaseline | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_classes: int,
        seed: int,
    ) -> "TabularBaselineBase":
        if n_classes > 2 and not self.supports_multiclass:
            raise NotImplementedError(
                f"{self.name!r} does not support multiclass classification "
                f"(n_classes={n_classes})"
            )
        t0 = time.time()
        model, extra = self._fit(X, y, n_classes=n_classes, seed=seed)
        self._fitted = FittedTabularBaseline(
            name=self.name,
            model=model,
            fit_seconds=time.time() - t0,
            extra=extra,
        )
        return self

    @abstractmethod
    def _fit(
        self, X, y, *, n_classes, seed,
    ) -> tuple[Any, Dict[str, Any]]:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError(f"{self.name!r} baseline must be .fit() first")
        proba = self._fitted.model.predict_proba(X)
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    @property
    def fit_seconds(self) -> float:
        return self._fitted.fit_seconds if self._fitted else 0.0


# --------------------------------------------------------------------------- #
# EBM (interpret.glassbox)
# --------------------------------------------------------------------------- #


class EBMBaseline(TabularBaselineBase):
    name = "ebm"

    def _fit(self, X, y, *, n_classes, seed):
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as e:
            raise ImportError(
                "interpret is not installed.  Install with `pip install interpret` "
                "or activate the per-baseline venv (see tabular/baselines_vendored.py)."
            ) from e
        params = dict(
            max_bins=self.kwargs.get("max_bins", 256),
            max_interaction_bins=self.kwargs.get("max_interaction_bins", 32),
            interactions=self.kwargs.get("interactions", 10),
            outer_bags=self.kwargs.get("outer_bags", 8),
            random_state=seed,
        )
        clf = ExplainableBoostingClassifier(**params).fit(X, y)
        return clf, {"n_features_in": int(getattr(clf, "n_features_in_", X.shape[1]))}


# --------------------------------------------------------------------------- #
# FIGS / RuleFit (imodels)
# --------------------------------------------------------------------------- #


class FIGSBaseline(TabularBaselineBase):
    name = "figs"

    def _fit(self, X, y, *, n_classes, seed):
        try:
            from imodels import FIGSClassifier
        except ImportError as e:
            raise ImportError(
                "imodels is not installed.  `pip install imodels`."
            ) from e
        params = dict(
            max_rules=self.kwargs.get("max_rules", 30),
            random_state=seed,
        )
        if "max_trees" in self.kwargs:
            params["max_trees"] = self.kwargs["max_trees"]
        clf = FIGSClassifier(**params).fit(X, y)
        return clf, {"n_trees": len(clf.trees_)}


class RuleFitBaseline(TabularBaselineBase):
    name = "rulefit"
    supports_multiclass = False

    def _fit(self, X, y, *, n_classes, seed):
        try:
            from imodels import RuleFitClassifier
        except ImportError as e:
            raise ImportError(
                "imodels is not installed.  `pip install imodels`."
            ) from e
        params = dict(
            max_rules=self.kwargs.get("max_rules", 30),
            random_state=seed,
        )
        clf = RuleFitClassifier(**params)
        feat_names = [f"feature_{i}" for i in range(X.shape[1])]
        clf.fit(X, y, feature_names=feat_names)
        n_rules = (
            len(clf.rules_) if hasattr(clf, "rules_") and hasattr(clf.rules_, "__len__")
            else 0
        )
        return clf, {"n_rules": int(n_rules)}


# --------------------------------------------------------------------------- #
# TabPFN
# --------------------------------------------------------------------------- #


_TABPFN_V1_MAX_N = 1000
_TABPFN_V1_MAX_D = 100


class TabPFNBaseline(TabularBaselineBase):
    name = "tabpfn"

    def _fit(self, X, y, *, n_classes, seed):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                "tabpfn is not installed.  `pip install tabpfn` "
                "(downloads a ~200 MB checkpoint on first use)."
            ) from e

        n, d = X.shape
        version = getattr(__import__("tabpfn"), "__version__", "0")
        # Best-effort: TabPFN v1 enforces n<=1000, d<=100 hard.  v2 lifts
        # these but still degrades on large inputs.
        if version.startswith("1.") and (n > _TABPFN_V1_MAX_N or d > _TABPFN_V1_MAX_D):
            raise NotImplementedError(
                f"TabPFN v1 cap exceeded (n={n}>{_TABPFN_V1_MAX_N} "
                f"or d={d}>{_TABPFN_V1_MAX_D}); upgrade to v2 or subsample"
            )

        params = dict(
            device=self.kwargs.get("device", "cpu"),
            random_state=seed,
        )
        # TabPFN v2 takes (n_estimators, ignore_pretraining_limits) extras.
        if "n_estimators" in self.kwargs:
            params["n_estimators"] = self.kwargs["n_estimators"]
        if "ignore_pretraining_limits" in self.kwargs:
            params["ignore_pretraining_limits"] = self.kwargs["ignore_pretraining_limits"]

        # Different TabPFN versions accept different kwargs; filter safely.
        import inspect
        sig = inspect.signature(TabPFNClassifier.__init__)
        accepted = {k: v for k, v in params.items() if k in sig.parameters}
        clf = TabPFNClassifier(**accepted)
        clf.fit(X, y)
        return clf, {"tabpfn_version": version, "n": int(n), "d": int(d)}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


TABULAR_BASELINE_REGISTRY: Dict[str, type] = {
    "ebm":     EBMBaseline,
    "figs":    FIGSBaseline,
    "rulefit": RuleFitBaseline,
    "tabpfn":  TabPFNBaseline,
}


TABULAR_BASELINE_LABELS: Dict[str, str] = {
    "ebm":     "EBM",
    "figs":    "FIGS-std",
    "rulefit": "RuleFit-std",
    "tabpfn":  "TabPFN",
}


def make_tabular_baseline(name: str, **kwargs: Any) -> TabularBaselineBase:
    if name not in TABULAR_BASELINE_REGISTRY:
        raise KeyError(
            f"unknown tabular baseline {name!r}; "
            f"choose from {sorted(TABULAR_BASELINE_REGISTRY)}"
        )
    return TABULAR_BASELINE_REGISTRY[name](**kwargs)
