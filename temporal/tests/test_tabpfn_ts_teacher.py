"""Smoke tests for TabPFN-TS temporal distillation helpers."""

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

from temporal.baselines import TabPFNTSBaseline
from temporal.compare_temporal import run_tabpfn_ts_distill
from temporal.datasets import load_synthetic_p12
from temporal.tabpfn_ts_teacher import TEACHER_FEATURES, TabPFNTSFeatureTeacher


def test_teacher_features_are_finite_and_named():
    X_ts, mask, _y, var_names, _ = load_synthetic_p12(
        n_samples=24, T=16, missing_ratio=0.25, seed=41,
    )
    teacher = TabPFNTSFeatureTeacher(
        backend="extratrees", seed=41, max_regression_rows=128,
    ).fit(X_ts, mask)
    feats = teacher.transform(X_ts, mask)
    assert teacher.backend_used == "extratrees"
    assert feats.shape == (X_ts.shape[0], X_ts.shape[2] * len(TEACHER_FEATURES))
    assert len(teacher.feature_names(var_names)) == feats.shape[1]
    assert np.all(np.isfinite(feats))


def test_l2_tabpfn_ts_distill_et_runs():
    X_ts, mask, y, _var_names, _ = load_synthetic_p12(
        n_samples=36, T=12, missing_ratio=0.2, seed=42,
    )
    results = run_tabpfn_ts_distill(
        X_ts[:24], mask[:24], y[:24],
        X_ts[24:], mask[24:], y[24:],
        var_names=[f"v{i}" for i in range(X_ts.shape[2])],
        n_classes=int(np.unique(y).size),
        seed=42,
        epochs=2,
        level="L2",
        student="et",
        n_windows=3,
        n_intervals=3,
        teacher_backend="extratrees",
        teacher_max_rows=128,
        teacher_head="logreg",
    )
    assert any(name.startswith("L2-TabPFNTS-DistillET") for name in results)
    assert all(np.isfinite(r.accuracy) for r in results.values())


def test_l3_tabpfn_ts_distill_et_runs():
    X_ts, mask, y, var_names, _ = load_synthetic_p12(
        n_samples=36, T=12, missing_ratio=0.2, seed=43,
    )
    results = run_tabpfn_ts_distill(
        X_ts[:24], mask[:24], y[:24],
        X_ts[24:], mask[24:], y[24:],
        var_names=var_names,
        n_classes=int(np.unique(y).size),
        seed=43,
        epochs=2,
        level="L3",
        student="et",
        n_windows=3,
        n_intervals=3,
        teacher_backend="extratrees",
        teacher_max_rows=128,
        teacher_head="logreg",
    )
    assert any(name.startswith("L3-TabPFNTS-DistillET") for name in results)
    assert all(np.isfinite(r.f1_weighted) for r in results.values())


def test_tabpfn_ts_black_box_baseline_runs():
    X_ts, mask, y, _var_names, _ = load_synthetic_p12(
        n_samples=36, T=12, missing_ratio=0.2, seed=44,
    )
    baseline = TabPFNTSBaseline(
        n_classes=int(np.unique(y).size),
        seed=44,
        ts_backend="extratrees",
        ts_max_rows=128,
        head="logreg",
    ).fit(X_ts[:24], mask[:24], y[:24])
    proba = baseline.predict_proba(X_ts[24:], mask[24:])
    assert proba.shape == (12, int(np.unique(y).size))
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)


if __name__ == "__main__":
    test_teacher_features_are_finite_and_named()
    test_l2_tabpfn_ts_distill_et_runs()
    test_l3_tabpfn_ts_distill_et_runs()
    test_tabpfn_ts_black_box_baseline_runs()
    print("test_tabpfn_ts_teacher: OK")
