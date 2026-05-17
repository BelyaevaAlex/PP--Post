"""Temporal extensions for PPθ-Post.

Four levels of temporal capability are provided, in increasing order of
architectural impact:

L1 — flat summary statistics per variable (interpretable shallow baseline).
L2 — multi-window summary statistics (temporal-flavoured shallow baseline).
L3 — interval-based backbone with temporal Branch metadata; conditions read
     ``mean of HR over hours 0–12 > 95``.  Compatible with all 9 PPθ-Post
     inference variants out of the box.
L4 — per-timestep latent ``z(b, X, t)`` with temporal aggregations
     (mean, max, noisy-or-over-time, k-of-T, attention).  ProbLog export uses
     temporal atoms ``z(B, X, T)`` and aggregation rules.

The public API exposes a unified loader interface so each dataset can be
materialised in any of the four temporal modes.
"""

from .tabularize import (
    summary_flatten,
    multi_window_flatten,
    summary_feature_names,
    multi_window_feature_names,
)
from .interval_forest import (
    IntervalFeatureExtractor,
    fit_interval_forest,
    interval_feature_meta_to_human,
)
from .temporal_problog import (
    feature_meta_to_atom,
    export_temporal_branches_to_problog,
    export_temporal_problog_program,
)
from .pp_theta_post_temporal import (
    PPThetaPostTemporal,
    TemporalAttentionAggregator,
    temporal_aggregate,
    AggregationMode,
    # legacy alias kept for backwards compatibility; use PPThetaPostTemporal
    TemporalRuleNetwork,
)
from .temporal_inference import (
    TemporalProbLogClassifier,
    aggregate_z_over_time,
    DEFAULT_TEMPORAL_VARIANTS,
)
from .datasets import (
    load_synthetic_p12,
    load_synthetic_pam,
    load_synthetic_mimic3_mortality,
    load_temporal_dataset,
)

__all__ = [
    # L1 / L2
    "summary_flatten",
    "multi_window_flatten",
    "summary_feature_names",
    "multi_window_feature_names",
    # L3
    "IntervalFeatureExtractor",
    "fit_interval_forest",
    "interval_feature_meta_to_human",
    "feature_meta_to_atom",
    "export_temporal_branches_to_problog",
    "export_temporal_problog_program",
    # L4
    "PPThetaPostTemporal",
    "TemporalRuleNetwork",  # legacy alias
    "TemporalAttentionAggregator",
    "temporal_aggregate",
    "AggregationMode",
    "TemporalProbLogClassifier",
    "aggregate_z_over_time",
    "DEFAULT_TEMPORAL_VARIANTS",
    # datasets
    "load_synthetic_p12",
    "load_synthetic_pam",
    "load_synthetic_mimic3_mortality",
    "load_temporal_dataset",
]
