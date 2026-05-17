"""Smoke tests for L4 PPThetaPostTemporal + temporal aggregation."""

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
from temporal.pp_theta_post_temporal import (
    PPThetaPostTemporal,
    TemporalAttentionAggregator,
    TemporalRuleNetwork,  # legacy alias, must still resolve
    temporal_aggregate,
    VALID_AGGREGATIONS,
)
from temporal.temporal_inference import (
    DEFAULT_TEMPORAL_VARIANTS,
    TemporalProbLogClassifier,
    aggregate_z_over_time,
)
from problog_inference import build_theta_matrix


def test_temporal_aggregate_modes_shapes():
    rng = np.random.default_rng(0)
    z = rng.uniform(size=(3, 5, 4))
    for mode in ("mean", "max", "exists", "forall", "last"):
        out = temporal_aggregate(z, mode=mode)
        assert out.shape == (3, 4)
        assert np.all(np.isfinite(out))


def test_temporal_aggregate_k_of_t():
    z = np.full((2, 6, 3), 0.5)
    out = temporal_aggregate(z, mode="k_of_t", k=0.5)
    assert out.shape == (2, 3)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_temporal_aggregate_attention():
    z = np.ones((1, 4, 2)) * 0.6
    weights = np.array([1.0, 0.0, 0.0, 0.0])
    out = temporal_aggregate(
        z, mode="attention", attention_weights=weights,
    )
    assert out.shape == (1, 2)
    assert np.allclose(out, 0.6)


def test_temporal_rule_network_fits_and_predicts():
    X_ts, mask, y, var_names, _ = load_synthetic_p12(
        n_samples=64, T=12, missing_ratio=0.5, seed=11,
    )
    tbn = PPThetaPostTemporal(
        var_names=var_names, n_classes=2, seed=11, epochs=20,
    ).fit(X_ts, mask, y)
    z_per_time = tbn.predict_branch_probs_per_time(X_ts, mask)
    assert z_per_time.shape == (X_ts.shape[0], X_ts.shape[1], len(tbn.branches))
    z_agg = tbn.predict_branch_probs(X_ts, mask, aggregation="mean")
    assert z_agg.shape == (X_ts.shape[0], len(tbn.branches))


def test_legacy_alias_resolves():
    """``TemporalRuleNetwork`` must still resolve to ``PPThetaPostTemporal``
    while the deprecation period is in effect."""
    assert TemporalRuleNetwork is PPThetaPostTemporal


def test_top_k_time_filter_reduces_noisy_or_saturation():
    """When ``top_k_time`` zeros all but the strongest few timesteps,
    noisy-or aggregation should produce strictly smaller activation."""
    rng = np.random.default_rng(0)
    z = rng.uniform(0.05, 0.4, size=(2, 50, 4))
    full = temporal_aggregate(z, mode="exists")
    filtered = temporal_aggregate(z, mode="exists", top_k_time=0.1)
    assert np.all(filtered <= full + 1e-9)
    # filtered should retain some signal — not collapse to zero.
    assert filtered.max() > 0.0


def test_temporal_classifier_works_for_default_variants():
    X_ts, mask, y, var_names, _ = load_synthetic_pam(
        n_per_class=8, T=12, missing_ratio=0.4, seed=13,
    )
    n_classes = int(np.unique(y).size)
    tbn = PPThetaPostTemporal(
        var_names=var_names, n_classes=n_classes, seed=13, epochs=20,
    ).fit(X_ts, mask, y)
    z_per_time = tbn.predict_branch_probs_per_time(X_ts, mask)
    theta = build_theta_matrix(tbn.branches, n_classes)

    attn_cache = {}
    for variant in DEFAULT_TEMPORAL_VARIANTS:
        if variant["temporal_mode"] == "attention":
            attn_mode = variant.get("attention_mode", "shared")
            if attn_mode not in attn_cache:
                tbn.fit_attention(
                    X_ts, mask, y, theta=theta,
                    mode=attn_mode, epochs=30, lr=0.05,
                )
                attn_cache[attn_mode] = tbn.attention.weights()
            attn_w = attn_cache[attn_mode]
        else:
            attn_w = None
        clf = TemporalProbLogClassifier(
            branches=tbn.branches,
            n_classes=n_classes,
            head=variant.get("head", "weighted_mean"),
            temporal_mode=variant["temporal_mode"],
            k=variant.get("k"),
            top_k_time=variant.get("top_k_time"),
            theta=theta,
        )
        proba = clf.predict_proba(z_per_time, attention_weights=attn_w)
        assert proba.shape == (X_ts.shape[0], n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_aggregation_modes_are_registered():
    expected = {"mean", "max", "exists", "forall", "k_of_t", "last", "attention"}
    assert expected == set(VALID_AGGREGATIONS)


def test_temporal_attention_aggregator_fits():
    rng = np.random.default_rng(42)
    z = rng.uniform(size=(20, 6, 5))
    theta = rng.uniform(size=(5, 3))
    theta = theta / theta.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, size=20)
    attn = TemporalAttentionAggregator(T=6)
    weights = attn.fit(z, theta, y, epochs=30, lr=0.05)
    assert weights.shape == (6,)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)


def test_per_branch_attention_returns_2d_weights():
    """Per-branch attention must learn an independent ``α_b ∈ R^T`` for
    every branch; the resulting weights tensor is ``[T, B]`` and each
    column sums to 1.
    """
    rng = np.random.default_rng(7)
    T, B, K = 6, 5, 3
    z = rng.uniform(size=(20, T, B))
    theta = rng.uniform(size=(B, K))
    theta = theta / theta.sum(axis=1, keepdims=True)
    y = rng.integers(0, K, size=20)
    attn = TemporalAttentionAggregator(T=T, n_branches=B, mode="per_branch")
    weights = attn.fit(z, theta, y, epochs=30, lr=0.05)
    assert weights.shape == (T, B)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-5)
    out = temporal_aggregate(z, mode="attention", attention_weights=weights)
    assert out.shape == (20, B)


def test_multi_head_attention_returns_2d_weights():
    rng = np.random.default_rng(8)
    T, B, K = 6, 5, 3
    z = rng.uniform(size=(20, T, B))
    theta = rng.uniform(size=(B, K))
    theta = theta / theta.sum(axis=1, keepdims=True)
    y = rng.integers(0, K, size=20)
    attn = TemporalAttentionAggregator(
        T=T, n_branches=B, mode="multi_head", n_heads=3,
    )
    weights = attn.fit(z, theta, y, epochs=30, lr=0.05)
    assert weights.shape == (T, B)


if __name__ == "__main__":
    test_temporal_aggregate_modes_shapes()
    test_temporal_aggregate_k_of_t()
    test_temporal_aggregate_attention()
    test_temporal_rule_network_fits_and_predicts()
    test_legacy_alias_resolves()
    test_top_k_time_filter_reduces_noisy_or_saturation()
    test_temporal_classifier_works_for_default_variants()
    test_aggregation_modes_are_registered()
    test_temporal_attention_aggregator_fits()
    test_per_branch_attention_returns_2d_weights()
    test_multi_head_attention_returns_2d_weights()
    print("test_pp_theta_post_temporal: OK")
