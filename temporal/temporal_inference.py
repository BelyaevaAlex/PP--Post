"""L4 — Temporal ProbLog inference variants.

This module provides a thin classifier wrapper that:

1. Receives the per-timestep latent tensor ``z_per_time`` of shape
   ``[N, T, B]`` produced by :class:`temporal.TemporalRuleNetwork`.
2. Aggregates the time dimension into a per-sample matrix
   ``z [N, B]`` via :func:`aggregate_z_over_time`.
3. Delegates the rest of the pipeline (θ matrix construction, posterior
   correction, noisy-or / weighted-mean / etc.) to the existing PPθ-Post
   inference building blocks in :mod:`problog_inference`.

The result is that **every** static PPθ-Post inference variant lifts to
the temporal setting for free; only the way ``z`` is produced changes.
The wrapper exposes a small registry of standard temporal modes used in
the comparison driver: ``PL-tMean``, ``PL-tMax``, ``PL-tNoisyOr``,
``PL-kOfT``, ``PL-tAttn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from branch_schema import Branch
from problog_inference import (
    aggregate_noisy_or,
    aggregate_weighted_mean,
    build_theta_matrix,
)

from .pp_theta_post_temporal import (
    AggregationMode,
    VALID_AGGREGATIONS,
    temporal_aggregate,
)


def aggregate_z_over_time(
    z_per_time: np.ndarray,
    mode: AggregationMode = "mean",
    k: Optional[float] = None,
    attention_weights: Optional[np.ndarray] = None,
    top_k_time: Optional[float] = None,
) -> np.ndarray:
    """Public facade for :func:`pp_theta_post_temporal.temporal_aggregate`."""
    return temporal_aggregate(
        z_per_time,
        mode=mode,
        k=k,
        attention_weights=attention_weights,
        top_k_time=top_k_time,
    )


@dataclass
class TemporalProbLogClassifier:
    """Classifier that combines a temporal aggregator with a
    PPθ-Post-compatible class head.

    Parameters
    ----------
    branches : list of Branch
        Branches extracted from the per-timestep RuleNetwork.
    n_classes : int
    head : str, optional
        Aggregation across branches once the temporal aggregation has
        produced a per-sample latent.  Either ``"noisy_or"`` (default,
        ``PL-fast``-equivalent semantics) or ``"weighted_mean"``
        (``PL-wmean``-equivalent semantics).
    temporal_mode : str
        One of :data:`VALID_AGGREGATIONS`.  Defaults to ``"mean"``.
    k : float, optional
        Activation threshold for ``temporal_mode == "k_of_t"``.
    min_theta : float
        Minimum θ value to avoid log(0).
    """

    branches: List[Branch]
    n_classes: int
    head: str = "noisy_or"
    temporal_mode: AggregationMode = "mean"
    k: Optional[float] = None
    top_k_time: Optional[float] = None
    min_theta: float = 1e-6
    theta: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.head not in ("noisy_or", "weighted_mean"):
            raise ValueError(
                f"head must be 'noisy_or' or 'weighted_mean', got {self.head!r}"
            )
        if self.temporal_mode not in VALID_AGGREGATIONS:
            raise ValueError(
                f"unsupported temporal_mode {self.temporal_mode!r}"
            )
        if self.theta is None:
            self.theta = build_theta_matrix(
                self.branches, self.n_classes, min_theta=self.min_theta,
            )

    def predict_proba(
        self,
        z_per_time: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        z = aggregate_z_over_time(
            z_per_time,
            mode=self.temporal_mode,
            k=self.k,
            attention_weights=attention_weights,
            top_k_time=self.top_k_time,
        )
        if z.ndim == 1:
            z = z[np.newaxis, :]
        if self.head == "noisy_or":
            return aggregate_noisy_or(z, self.theta)
        return aggregate_weighted_mean(z, self.theta)

    def predict(
        self,
        z_per_time: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        proba = self.predict_proba(
            z_per_time, attention_weights=attention_weights,
        )
        return np.argmax(proba, axis=1)


# ─────────────────────────────────────────────────────────────────────────
# Registry of canonical L4 inference variants
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_TEMPORAL_VARIANTS: Sequence[dict] = (
    {"name": "PL-tMean",         "temporal_mode": "mean",     "head": "weighted_mean"},
    {"name": "PL-tMax",          "temporal_mode": "max",      "head": "weighted_mean"},
    {"name": "PL-tNoisyOr",      "temporal_mode": "exists",   "head": "noisy_or"},
    # PL-tNoisyOr-topk: zero out low-activation timesteps before taking
    # noisy-or — directly addresses saturation when T is large.
    {"name": "PL-tNoisyOr-top10", "temporal_mode": "exists",   "head": "noisy_or",
     "top_k_time": 0.1},
    {"name": "PL-tNoisyOr-top25", "temporal_mode": "exists",   "head": "noisy_or",
     "top_k_time": 0.25},
    {"name": "PL-tForall",       "temporal_mode": "forall",   "head": "weighted_mean"},
    {"name": "PL-kOfT-25",       "temporal_mode": "k_of_t",   "k": 0.25,
     "head": "weighted_mean"},
    {"name": "PL-kOfT-50",       "temporal_mode": "k_of_t",   "k": 0.5,
     "head": "weighted_mean"},
    {"name": "PL-tLast",         "temporal_mode": "last",     "head": "weighted_mean"},
    {"name": "PL-tAttn",         "temporal_mode": "attention",
     "head": "weighted_mean"},
    # Per-branch attention variant (multi-head) — uses learnable
    # per-branch temporal pooling, fitted by ``fit_attention(mode='per_branch')``.
    {"name": "PL-tAttnPB",       "temporal_mode": "attention",
     "head": "weighted_mean", "attention_mode": "per_branch"},
    {"name": "PL-tAttnMH",       "temporal_mode": "attention",
     "head": "weighted_mean", "attention_mode": "multi_head"},
)
