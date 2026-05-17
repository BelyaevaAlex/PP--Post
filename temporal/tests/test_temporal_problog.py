"""Smoke tests for L3 / L4 ProbLog export."""

from __future__ import annotations

import os
import re
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, GRANDPARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from branch_schema import Branch, Condition
from temporal.datasets import load_synthetic_p12
from temporal.interval_forest import (
    IntervalFeatureExtractor, fit_interval_forest,
)
from temporal.temporal_problog import (
    feature_meta_to_atom,
    export_temporal_branches_to_problog,
    export_temporal_problog_program,
    temporal_aggregation_rule,
    temporal_latent_facts,
)


def _build_dummy_branch():
    cond = Condition(feature_idx=0, threshold=12.5, direction="le", node_id=2)
    branch = Branch(
        branch_id="b0", tree_id=0, parent_node_id=2,
        conditions=[cond],
        class_proportions=[0.7, 0.3],
        split_feature_idx=0,
        split_threshold=12.5,
        split_node_id=2,
    )
    return branch


def test_feature_meta_to_atom_uses_temporal_functor():
    X_ts, mask, y, var_names, _ = load_synthetic_p12(n_samples=8, T=12, seed=5)
    forest, extractor, X_feat = fit_interval_forest(
        X_ts, mask, y, var_names=var_names, n_intervals=3, seed=5,
    )
    branch = _build_dummy_branch()
    branch.conditions[0].feature_idx = 0  # first interval feature
    atom = feature_meta_to_atom(branch, branch.conditions[0],
                                 extractor.feature_meta, x_symbol="X")
    meta0 = extractor.feature_meta[0]
    expected_functor = f"le_{meta0.stat}"
    assert atom.startswith(expected_functor)
    assert meta0.variable_name in atom


def test_export_temporal_branches_to_problog_no_meta():
    branch = _build_dummy_branch()
    out_path = os.path.join(THIS_DIR, "_tmp_kb.pl")
    try:
        export_temporal_branches_to_problog([branch], None, output_path=out_path)
        assert os.path.exists(out_path)
        text = open(out_path).read()
        assert "branch_struct(b0, X)" in text
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_temporal_problog_program_with_meta():
    X_ts, mask, y, var_names, _ = load_synthetic_p12(n_samples=8, T=10, seed=6)
    forest, extractor, X_feat = fit_interval_forest(
        X_ts, mask, y, var_names=var_names, n_intervals=3, seed=6,
    )
    branch = _build_dummy_branch()
    branch.conditions[0].feature_idx = 0
    program = export_temporal_problog_program(
        branches=[branch],
        branch_probs_single=np.array([0.6]),
        interval_feature_row=X_feat[0],
        feature_meta=extractor.feature_meta,
        x_id=0,
        n_classes=2,
    )
    assert "z(b0,0)" in program
    assert "evidence(" in program


def test_temporal_latent_facts_shape():
    branch = _build_dummy_branch()
    z_per_time = np.linspace(0.1, 0.9, 4).reshape(4, 1)
    facts = temporal_latent_facts([branch], z_per_time, x_id=3)
    assert len(facts) == 4
    assert all(re.match(r"\d\.\d+::z\(b0,3,\d+\)\.", line) for line in facts)


def test_temporal_aggregation_rule_modes():
    branch = _build_dummy_branch()
    assert "z_overall" in temporal_aggregation_rule(branch, "exists")
    assert temporal_aggregation_rule(branch, "mean").startswith("%")


if __name__ == "__main__":
    test_feature_meta_to_atom_uses_temporal_functor()
    test_export_temporal_branches_to_problog_no_meta()
    test_temporal_problog_program_with_meta()
    test_temporal_latent_facts_shape()
    test_temporal_aggregation_rule_modes()
    print("test_temporal_problog: OK")
