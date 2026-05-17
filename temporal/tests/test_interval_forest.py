"""Smoke tests for L3 interval forest backbone."""

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

from temporal.datasets import load_synthetic_pam
from temporal.interval_forest import (
    INTERVAL_STATS,
    IntervalFeatureExtractor,
    fit_interval_forest,
    interval_feature_meta_to_human,
)


def test_extractor_metadata_is_consistent():
    X_ts, mask, _, var_names, _ = load_synthetic_pam(n_per_class=2, T=20, seed=2)
    extractor = IntervalFeatureExtractor(
        var_names=var_names, T=20, n_intervals=5, seed=7,
    )
    feats = extractor.transform(X_ts, mask)
    assert feats.shape == (X_ts.shape[0], extractor.n_features)
    n_intervals_total = len(extractor.intervals)
    expected_n_features = len(var_names) * n_intervals_total * len(INTERVAL_STATS)
    assert extractor.n_features == expected_n_features
    assert all(0 <= m.feature_idx < extractor.n_features
               for m in extractor.feature_meta)


def test_fit_interval_forest_runs():
    X_ts, mask, y, var_names, _ = load_synthetic_pam(n_per_class=4, T=20, seed=3)
    forest, extractor, X_feat = fit_interval_forest(
        X_ts, mask, y, var_names=var_names, n_intervals=4, seed=3,
    )
    assert forest is not None
    assert X_feat.shape == (X_ts.shape[0], extractor.n_features)
    pred = forest.predict(X_feat)
    assert pred.shape == y.shape


def test_interval_feature_meta_to_human():
    X_ts, mask, _, var_names, _ = load_synthetic_pam(n_per_class=1, T=10, seed=4)
    extractor = IntervalFeatureExtractor(
        var_names=var_names, T=10, n_intervals=3, seed=4,
    )
    sample_meta = extractor.feature_meta[0]
    text = interval_feature_meta_to_human(sample_meta)
    assert sample_meta.variable_name in text
    assert str(sample_meta.interval_start) in text


if __name__ == "__main__":
    test_extractor_metadata_is_consistent()
    test_fit_interval_forest_runs()
    test_interval_feature_meta_to_human()
    print("test_interval_forest: OK")
