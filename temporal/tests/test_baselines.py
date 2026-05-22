"""Smoke-suite for the local baseline track in :mod:`temporal.baselines`.

After the migration to the *only-vendored* SOTA strategy, this module
hosts local baselines with no upstream worth vendoring plus the
standalone TabPFN-TS black-box row: ``lr`` / ``xgb`` / ``tabpfn_ts`` /
``transformer``.  The seven SOTA baselines (GRU-D, SAnD, mTAN, SeFT,
Raindrop, CAMELOT, InterpGN) are covered by
:mod:`temporal.tests.test_baselines_vendored` and
:mod:`temporal.tests.test_baselines_vendored_tf`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, GRANDPARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from temporal.baselines import (  # noqa: E402
    BASELINE_REGISTRY,
    DEFAULT_BASELINES,
    make_baseline,
)
from temporal.datasets import load_synthetic_p12  # noqa: E402


def _tiny_dataset():
    return load_synthetic_p12(
        n_samples=24, T=8, missing_ratio=0.4, seed=11,
    )


def test_registry_covers_local_baselines_only():
    expected = {"lr", "xgb", "tabpfn_ts", "transformer"}
    assert set(BASELINE_REGISTRY) == expected
    assert set(DEFAULT_BASELINES) == expected


def test_make_baseline_unknown_raises():
    try:
        make_baseline("does_not_exist", n_classes=2)
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown baseline")


def _check_fit_predict(name: str, fast_kwargs):
    X_ts, mask, y, var_names, _ = _tiny_dataset()
    baseline = make_baseline(name, n_classes=2, seed=0, **fast_kwargs)
    baseline.fit(X_ts, mask, y)
    proba = baseline.predict_proba(X_ts, mask)
    assert proba.shape == (X_ts.shape[0], 2), (
        f"{name}: expected ({X_ts.shape[0]}, 2), got {proba.shape}"
    )
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(proba >= -1e-6) and np.all(proba <= 1.0 + 1e-6)


def test_lr_stats_fits_and_predicts():
    _check_fit_predict("lr", {})


def test_xgb_stats_fits_and_predicts():
    _check_fit_predict("xgb", {"n_windows": 2})


def test_tabpfn_ts_baseline_fits_and_predicts():
    _check_fit_predict(
        "tabpfn_ts",
        {"ts_backend": "extratrees", "ts_max_rows": 128, "head": "logreg"},
    )


def test_transformer_fits_and_predicts():
    _check_fit_predict(
        "transformer",
        {"d_model": 16, "n_heads": 2, "n_layers": 1, "epochs": 5,
         "batch_size": 8},
    )


if __name__ == "__main__":
    test_registry_covers_local_baselines_only()
    test_make_baseline_unknown_raises()
    test_lr_stats_fits_and_predicts()
    test_xgb_stats_fits_and_predicts()
    test_tabpfn_ts_baseline_fits_and_predicts()
    test_transformer_fits_and_predicts()
    print("test_baselines: OK")
