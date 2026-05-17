"""Sanity test: build_model_from_branches must reproduce build_model_from_ensemble.

The refactor of :class:`RuleNetwork` split the ensemble parser
(:func:`extract_branches_from_sklearn_ensemble`) from the weight-builder
(:meth:`build_model_from_branches`).  This test pins the equivalence
between the two entry points so that future rule sources (XGB, CatBoost,
FIGS, RuleFit) can route through ``build_model_from_branches`` with
confidence that the sklearn path is unaffected.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import ExtraTreesClassifier

from rule_network import (
    RuleNetwork,
    extract_branches_from_sklearn_ensemble,
)


def _fit_ensemble(loader, seed=0, n_estimators=8, max_leaf_nodes=16):
    data = loader()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
        n_jobs=1,
    ).fit(X, y)
    return et, X, y


def _build_via_ensemble(et):
    m = RuleNetwork(device=torch.device("cpu"))
    m.build_model_from_ensemble(et)
    return m


def _build_via_branches(et):
    m = RuleNetwork(device=torch.device("cpu"))
    branches = extract_branches_from_sklearn_ensemble(et)
    m.build_model_from_branches(
        branches,
        in_features=int(et.n_features_in_),
        out_features=int(et.n_classes_),
    )
    return m


def _assert_models_equivalent(m1: RuleNetwork, m2: RuleNetwork, X: np.ndarray):
    assert m1.hidden_neurons == m2.hidden_neurons, (
        f"hidden_neurons differ: {m1.hidden_neurons} vs {m2.hidden_neurons}"
    )
    assert m1.in_features == m2.in_features
    assert m1.out_features == m2.out_features
    assert len(m1.branches) == len(m2.branches)

    for b1, b2 in zip(m1.branches, m2.branches):
        assert b1.branch_id == b2.branch_id
        assert b1.tree_id == b2.tree_id
        assert b1.parent_node_id == b2.parent_node_id
        assert b1.split_feature_idx == b2.split_feature_idx
        assert b1.split_node_id == b2.split_node_id
        if b1.split_threshold is None:
            assert b2.split_threshold is None
        else:
            assert abs(b1.split_threshold - b2.split_threshold) < 1e-9
        assert len(b1.conditions) == len(b2.conditions)
        for c1, c2 in zip(b1.conditions, b2.conditions):
            assert c1.feature_idx == c2.feature_idx
            assert c1.direction == c2.direction
            assert c1.node_id == c2.node_id
            assert abs(c1.threshold - c2.threshold) < 1e-9
        assert np.allclose(
            np.asarray(b1.class_proportions),
            np.asarray(b2.class_proportions),
            atol=1e-12,
        )

    assert torch.allclose(m1.w1.data, m2.w1.data, atol=1e-12), (
        f"w1 max abs diff = {(m1.w1.data - m2.w1.data).abs().max().item()}"
    )
    assert torch.allclose(m1.w2.data, m2.w2.data, atol=1e-12), (
        f"w2 max abs diff = {(m1.w2.data - m2.w2.data).abs().max().item()}"
    )
    assert (m1.m1 == m2.m1).all()

    # Forward parity in eval mode (BN stats are init-identical because both
    # models have just been built; we compare the linear pipeline only).
    m1.eval()
    m2.eval()
    xb = torch.from_numpy(X[:32].astype(np.float32))
    with torch.no_grad():
        # Bypass BN stats divergence (they default to identity at init).
        out1 = m1(xb)
        out2 = m2(xb)
    assert torch.allclose(out1, out2, atol=1e-6), (
        f"forward max abs diff = {(out1 - out2).abs().max().item()}"
    )


def test_equivalence_iris():
    et, X, _ = _fit_ensemble(load_iris)
    _assert_models_equivalent(_build_via_ensemble(et), _build_via_branches(et), X)


def test_equivalence_wine():
    et, X, _ = _fit_ensemble(load_wine, n_estimators=12, max_leaf_nodes=24)
    _assert_models_equivalent(_build_via_ensemble(et), _build_via_branches(et), X)


def test_branches_per_tree_extractor_idempotent():
    """Running the extractor twice must produce identical Branch objects."""
    et, _, _ = _fit_ensemble(load_iris)
    a = extract_branches_from_sklearn_ensemble(et)
    b = extract_branches_from_sklearn_ensemble(et)
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        assert len(ta) == len(tb)
        for ba, bb in zip(ta, tb):
            assert ba.to_dict() == bb.to_dict()


if __name__ == "__main__":
    test_equivalence_iris()
    test_equivalence_wine()
    test_branches_per_tree_extractor_idempotent()
    print("OK: build_model_from_branches ≡ build_model_from_ensemble")
