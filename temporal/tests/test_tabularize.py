"""Smoke tests for L1 / L2 tabularization."""

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

from temporal.datasets import load_synthetic_p12, load_synthetic_pam
from temporal.tabularize import (
    SUMMARY_STATS,
    multi_window_feature_names,
    multi_window_flatten,
    summary_feature_names,
    summary_flatten,
)


def test_summary_flatten_shape():
    X_ts, mask, y, var_names, _ = load_synthetic_p12(n_samples=20, T=24, seed=0)
    feats = summary_flatten(X_ts, mask)
    assert feats.shape == (20, len(var_names) * len(SUMMARY_STATS))
    assert np.all(np.isfinite(feats))


def test_summary_flatten_handles_all_missing():
    X_ts = np.zeros((1, 5, 2), dtype=np.float32)
    mask = np.zeros((1, 5, 2), dtype=np.uint8)
    feats = summary_flatten(X_ts, mask)
    assert feats.shape == (1, 2 * len(SUMMARY_STATS))
    assert np.allclose(feats, 0.0)


def test_summary_feature_names_match_columns():
    X_ts, mask, y, var_names, _ = load_synthetic_pam(n_per_class=2, T=12, seed=0)
    feats = summary_flatten(X_ts, mask)
    names = summary_feature_names(var_names)
    assert len(names) == feats.shape[1]


def test_multi_window_flatten_shape():
    X_ts, mask, y, var_names, _ = load_synthetic_p12(n_samples=12, T=24, seed=1)
    n_windows = 4
    feats = multi_window_flatten(X_ts, mask, n_windows=n_windows)
    expected = X_ts.shape[2] * (n_windows * len(SUMMARY_STATS) + 2)
    assert feats.shape == (12, expected)
    assert np.all(np.isfinite(feats))


def test_multi_window_feature_names_match_columns():
    X_ts, mask, _, var_names, _ = load_synthetic_pam(n_per_class=2, T=12, seed=1)
    n_windows = 3
    feats = multi_window_flatten(X_ts, mask, n_windows=n_windows)
    names = multi_window_feature_names(var_names, n_windows=n_windows, T=12)
    assert len(names) == feats.shape[1]


if __name__ == "__main__":
    test_summary_flatten_shape()
    test_summary_flatten_handles_all_missing()
    test_summary_feature_names_match_columns()
    test_multi_window_flatten_shape()
    test_multi_window_feature_names_match_columns()
    print("test_tabularize: OK")
