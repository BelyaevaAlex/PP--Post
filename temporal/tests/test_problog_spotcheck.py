"""Smoke test: L3 ProbLog program ≡ analytical posterior on a tiny but
*complete* model (no branch truncation)."""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, GRANDPARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from temporal.problog_spotcheck import spotcheck_l3


def test_l3_problog_engine_matches_analytical_posterior():
    """The full L3 ProbLog program of a tiny forest must agree with
    :func:`_compute_analytical_posterior` to within ``1e-6``."""
    results = spotcheck_l3(
        n_samples=18, n_intervals=2, T=6,
        n_check=2, seed=42, epochs=10,
        n_estimators=2, max_leaf_nodes=4,
        atol=1e-3,
    )
    assert results, "spotcheck must return at least one sample"
    for r in results:
        assert r["max_abs_err"] < 1e-6, (
            f"engine vs analytical diverge for x_id={r['x_id']} "
            f"by {r['max_abs_err']:.4e}"
        )


if __name__ == "__main__":
    test_l3_problog_engine_matches_analytical_posterior()
    print("test_problog_spotcheck: OK")
