"""L3 — Time-Series-Forest-style backbone for PPθ-Post.

For each variable and each *random* (or grid) interval ``[t_a, t_b)`` we
compute three statistics (mean, std, slope) and treat the result as a
single tabular feature.  An ``ExtraTreesClassifier`` is then fitted on
these interval features.  The resulting tree topology (parent-of-leaf
branches) is **identical** to the static case, but every condition
``feature_idx ≤ threshold`` now corresponds to a *temporal* clause such
as ``mean(HR over hours 0–12) ≤ 95``.

The feature metadata table produced by :class:`IntervalFeatureExtractor`
is consumed by :mod:`temporal_problog` to emit temporal ProbLog atoms
(e.g. ``gt_mean(b0, hr, 0, 12, 95.0, X)``) without modifying the
underlying ``Branch`` / ``Condition`` schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


INTERVAL_STATS = ("mean", "std", "slope")


@dataclass
class IntervalFeatureMeta:
    """Metadata for a single interval-based feature."""

    feature_idx: int
    variable_idx: int
    variable_name: str
    interval_start: int
    interval_end: int
    stat: str  # one of INTERVAL_STATS

    def to_dict(self) -> dict:
        return {
            "feature_idx": int(self.feature_idx),
            "variable_idx": int(self.variable_idx),
            "variable_name": str(self.variable_name),
            "interval_start": int(self.interval_start),
            "interval_end": int(self.interval_end),
            "stat": str(self.stat),
        }


@dataclass
class IntervalFeatureExtractor:
    """Generate interval-based features for a multivariate time-series."""

    var_names: Sequence[str]
    T: int
    n_intervals: int = 12
    min_interval_len: int = 4
    max_interval_len: Optional[int] = None
    stats: Sequence[str] = field(default_factory=lambda: list(INTERVAL_STATS))
    seed: int = 42
    intervals: List[Tuple[int, int]] = field(default_factory=list)
    feature_meta: List[IntervalFeatureMeta] = field(default_factory=list)

    def __post_init__(self) -> None:
        for s in self.stats:
            if s not in INTERVAL_STATS:
                raise ValueError(f"unsupported stat {s!r}")
        rng = np.random.default_rng(self.seed)
        max_len = self.max_interval_len if self.max_interval_len else self.T
        intervals: List[Tuple[int, int]] = []
        for _ in range(self.n_intervals):
            length = int(rng.integers(self.min_interval_len, max_len + 1))
            start = int(rng.integers(0, self.T - length + 1))
            intervals.append((start, start + length))
        # add a "full-window" interval as a stable anchor
        intervals.append((0, self.T))
        self.intervals = intervals

        meta: List[IntervalFeatureMeta] = []
        idx = 0
        for v_idx, v_name in enumerate(self.var_names):
            for (a, b) in self.intervals:
                for s in self.stats:
                    meta.append(IntervalFeatureMeta(
                        feature_idx=idx,
                        variable_idx=v_idx,
                        variable_name=str(v_name),
                        interval_start=a,
                        interval_end=b,
                        stat=s,
                    ))
                    idx += 1
        self.feature_meta = meta

    @property
    def n_features(self) -> int:
        return len(self.feature_meta)

    def transform(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Convert ``[N, T, V]`` time-series into ``[N, n_features]``.

        Missing entries are skipped via ``mask``.  Intervals with no
        observations contribute zeros (the trees can still recover this
        information from the *count* / *mask* implicitly via splits on
        other features).
        """
        if X_ts.ndim != 3 or mask.ndim != 3 or X_ts.shape != mask.shape:
            raise ValueError("X_ts and mask must both have shape [N, T, V]")
        N = X_ts.shape[0]
        out = np.zeros((N, self.n_features), dtype=np.float32)

        n_stats = len(self.stats)
        n_intervals_per_var = len(self.intervals) * n_stats

        for i in range(N):
            for v_idx in range(len(self.var_names)):
                base = v_idx * n_intervals_per_var
                for w_idx, (a, b) in enumerate(self.intervals):
                    obs = mask[i, a:b, v_idx].astype(bool)
                    if not obs.any():
                        for s_idx in range(n_stats):
                            out[i, base + w_idx * n_stats + s_idx] = 0.0
                        continue
                    indices = np.nonzero(obs)[0] + a
                    values = X_ts[i, indices, v_idx].astype(np.float64)
                    n = values.shape[0]
                    for s_idx, s in enumerate(self.stats):
                        col = base + w_idx * n_stats + s_idx
                        if s == "mean":
                            out[i, col] = float(values.mean())
                        elif s == "std":
                            out[i, col] = float(values.std()) if n > 1 else 0.0
                        elif s == "slope":
                            if n < 2 or indices.std() == 0:
                                out[i, col] = 0.0
                            else:
                                a_mat = np.vstack([
                                    indices.astype(np.float64),
                                    np.ones(n, dtype=np.float64),
                                ]).T
                                slope, _ = np.linalg.lstsq(
                                    a_mat, values, rcond=None
                                )[0]
                                out[i, col] = float(slope)
        return out


def fit_interval_forest(
    X_ts: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    var_names: Sequence[str],
    n_intervals: int = 12,
    n_estimators: Optional[int] = None,
    max_leaf_nodes: Optional[int] = None,
    seed: int = 42,
) -> Tuple[ExtraTreesClassifier, IntervalFeatureExtractor, np.ndarray]:
    """Fit an ExtraTrees ensemble over interval features.

    The defaults follow the NSToolkit-compatible rule-network budget:

    * ``n_estimators = n_classes + floor(log2(n_features))``
    * ``max_leaf_nodes = 2 ** (floor(log2(n_features)) + 4)``

    Returns
    -------
    (forest, extractor, X_features)
        ``forest`` is the fitted ExtraTreesClassifier; ``extractor``
        carries the interval feature metadata (used downstream for
        temporal ProbLog export); ``X_features`` is the materialised
        feature matrix on which ``forest`` was trained.
    """
    if X_ts.ndim != 3 or mask.ndim != 3 or X_ts.shape != mask.shape:
        raise ValueError("X_ts and mask must both have shape [N, T, V]")
    N, T, V = X_ts.shape
    extractor = IntervalFeatureExtractor(
        var_names=list(var_names),
        T=T,
        n_intervals=n_intervals,
        seed=seed,
    )
    X_feat = extractor.transform(X_ts, mask)

    n_classes = int(np.unique(y).size)
    log2_d = int(np.floor(np.log2(max(extractor.n_features, 2))))
    if n_estimators is None:
        n_estimators = max(2, n_classes + log2_d)
    if max_leaf_nodes is None:
        max_leaf_nodes = 2 ** (log2_d + 4)

    forest = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X_feat, y)
    return forest, extractor, X_feat


def interval_feature_meta_to_human(meta: IntervalFeatureMeta) -> str:
    """Render a feature meta entry as ``stat(var, [t_a:t_b])``."""
    return (
        f"{meta.stat}({meta.variable_name}, "
        f"[{meta.interval_start}:{meta.interval_end}])"
    )
