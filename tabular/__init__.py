"""Tabular extension: alternative rule sources and standalone baselines.

This package adds two parallel tracks for the tabular benchmark of
PPtheta-Post:

* :mod:`tabular.rule_sources` — drop-in replacements for the default
  ExtraTrees rule source (XGBoost, CatBoost, FIGS, RuleFit).  Each
  source produces :class:`branch_schema.Branch` lists in the format
  consumed by :meth:`rule_network.RuleNetwork.build_model_from_branches`,
  so the rest of the PPθ-Post pipeline (PL-fast / PL-full / PL-wmean /
  e2e-NoisyOr / theta-learn / …) runs unchanged on top of any source.

* :mod:`tabular.baselines` — standalone competitors that do NOT feed
  the PPθ-Post pipeline (EBM, FIGS, RuleFit, TabPFN).  They expose a
  uniform ``fit(X, y) / predict_proba(X)`` contract, like
  ``temporal.baselines`` does for IMTS SOTA.

The vendoring fallback (per-baseline venv, git submodule) lives in
:mod:`tabular.baselines_vendored` and is activated lazily only when a
pip-installed import fails.
"""
