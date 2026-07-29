"""Synthetic and demo time-series datasets compatible with the four PPθ-Post
temporal levels.

Conventions
-----------
A dataset is a tuple ``(X_ts, mask, y, var_names, dataset_name)`` where:

* ``X_ts``  has shape ``[n_samples, T, n_vars]`` (np.float32).
* ``mask``  has shape ``[n_samples, T, n_vars]`` (uint8) with 1 for observed
  values and 0 for missing; missing entries in ``X_ts`` are filled with
  ``np.nan`` so that L1/L2 statistics can ignore them safely.
* ``y``     has shape ``[n_samples]`` with integer class labels.
* ``var_names`` is a list of length ``n_vars`` with human-readable names.

The synthetic generators are deliberately small so the full L1→L4 comparison
runs on a laptop in a few minutes while still exposing the structural
features of each benchmark (high missingness in P12-like, multiclass in
PAM-like, moderate missingness in MIMIC-3-demo-like).

These are *prototype* datasets for pipeline validation, NOT a substitute
for the real benchmarks — replacing this module with real PhysioNet /
PAMAP2 / MIMIC loaders does not require any other code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


TemporalDataset = Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str]


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────

def _apply_missingness(
    rng: np.random.Generator,
    x: np.ndarray,
    missing_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop ``missing_ratio`` fraction of observations independently per
    (sample, time, variable) cell.  Returns ``(x_with_nan, mask)``.
    """
    if missing_ratio <= 0.0:
        return x.astype(np.float32), np.ones_like(x, dtype=np.uint8)

    keep = rng.random(size=x.shape) >= missing_ratio
    mask = keep.astype(np.uint8)
    x_out = x.astype(np.float32).copy()
    x_out[~keep] = np.nan
    return x_out, mask


def _ar1_series(
    rng: np.random.Generator,
    T: int,
    mean: float,
    std: float,
    rho: float = 0.85,
) -> np.ndarray:
    """Generate a stationary AR(1) series of length T with given marginal
    mean / std.
    """
    eps = rng.normal(0, std * np.sqrt(1.0 - rho * rho), size=T)
    s = np.empty(T, dtype=np.float32)
    s[0] = mean + rng.normal(0, std)
    for t in range(1, T):
        s[t] = mean + rho * (s[t - 1] - mean) + eps[t]
    return s


# ─────────────────────────────────────────────────────────────────────────
# Synthetic P12 — ICU mortality (binary, high missingness)
# ─────────────────────────────────────────────────────────────────────────

P12_VARS = [
    "HR", "MAP", "SBP", "DBP", "Temp", "Resp",
    "Glucose", "Lactate", "Creatinine", "BUN", "WBC", "Bilirubin",
]


def load_synthetic_p12(
    n_samples: int = 600,
    T: int = 48,
    missing_ratio: float = 0.85,
    pos_ratio: float = 0.14,
    seed: int = 42,
) -> TemporalDataset:
    """Synthetic P12-like ICU mortality benchmark.

    Class structure:
        * ``y = 1`` (mortality):  mean HR shifted up, lactate trend up,
          MAP shifted down, WBC elevated.  Mimics deteriorating ICU course.
        * ``y = 0``:               near-normal AR(1) trajectories.

    Parameters mirror the published P12 description (12k patients, T=48,
    36 vars, 88% missing) but at a fraction of the scale for laptop runs.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_samples * pos_ratio))
    n_neg = n_samples - n_pos
    y = np.concatenate([np.zeros(n_neg, dtype=np.int64),
                         np.ones(n_pos, dtype=np.int64)])

    n_vars = len(P12_VARS)
    X = np.zeros((n_samples, T, n_vars), dtype=np.float32)

    base_means = np.array(
        [80, 75, 120, 70, 36.8, 16,
         110, 1.5, 1.0, 18, 9, 0.8],
        dtype=np.float32,
    )
    base_stds = np.array(
        [8, 8, 10, 8, 0.4, 2,
         15, 0.5, 0.3, 5, 2, 0.2],
        dtype=np.float32,
    )

    pos_mean_shift = np.array(
        [+15, -12, -15, -8, +0.6, +4,
         +25, +1.5, +0.5, +6, +3, +0.4],
        dtype=np.float32,
    )

    for i in range(n_samples):
        is_pos = y[i] == 1
        for v in range(n_vars):
            mean = base_means[v]
            if is_pos:
                # gradual trend: positive cases drift toward shifted mean
                trend = np.linspace(0.0, pos_mean_shift[v], T)
                series = _ar1_series(rng, T, mean, base_stds[v]) + trend
            else:
                series = _ar1_series(rng, T, mean, base_stds[v])
            X[i, :, v] = series

    X, mask = _apply_missingness(rng, X, missing_ratio)
    perm = rng.permutation(n_samples)
    return (X[perm], mask[perm], y[perm], list(P12_VARS), "synthetic_p12")


# ─────────────────────────────────────────────────────────────────────────
# Synthetic PAM — wearable activity recognition (multiclass, moderate missingness)
# ─────────────────────────────────────────────────────────────────────────

PAM_VARS = [
    "hand_acc_x", "hand_acc_y", "hand_acc_z",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "hand_gyro", "chest_gyro", "ankle_gyro",
    "hand_mag", "chest_mag", "ankle_mag",
    "heart_rate", "skin_temp",
]
PAM_ACTIVITIES = [
    "lying", "sitting", "standing", "walking",
    "ascending_stairs", "descending_stairs", "cycling", "running",
]


def load_synthetic_pam(
    n_per_class: int = 60,
    T: int = 100,
    missing_ratio: float = 0.55,
    seed: int = 42,
) -> TemporalDataset:
    """Synthetic PAM-like activity-recognition benchmark.

    Each activity class is associated with a distinct pattern in the
    accelerometer / gyroscope / heart-rate channels (different mean,
    amplitude and dominant frequency).  This produces a multiclass
    problem (8 activities) that is genuinely *temporal* — frequency
    content matters more than marginal statistics — yet is still
    learnable from windowed summaries.
    """
    rng = np.random.default_rng(seed)
    n_vars = len(PAM_VARS)
    n_classes = len(PAM_ACTIVITIES)
    n_samples = n_classes * n_per_class

    # per-class signature: (mean[n_vars], amp[n_vars], freq[n_vars], hr_mean)
    class_signatures: List[Dict[str, np.ndarray]] = []
    for k in range(n_classes):
        mean = rng.normal(0.0, 0.3, size=n_vars).astype(np.float32)
        mean[15] = 60 + 8 * k          # heart_rate ramps with intensity
        mean[16] = 33 + 0.15 * k        # skin temp slight rise
        amp = np.abs(rng.normal(0.5 + 0.15 * k, 0.2, size=n_vars)).astype(np.float32)
        freq = np.abs(rng.normal(0.5 + 0.4 * k, 0.15, size=n_vars)).astype(np.float32)
        class_signatures.append({"mean": mean, "amp": amp, "freq": freq})

    X = np.zeros((n_samples, T, n_vars), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    t_axis = np.linspace(0, 2 * np.pi, T, dtype=np.float32)

    idx = 0
    for k in range(n_classes):
        sig = class_signatures[k]
        for _ in range(n_per_class):
            phase = rng.uniform(0, 2 * np.pi, size=n_vars).astype(np.float32)
            jitter = rng.normal(0, 0.1, size=(T, n_vars)).astype(np.float32)
            for v in range(n_vars):
                X[idx, :, v] = (
                    sig["mean"][v]
                    + sig["amp"][v] * np.sin(sig["freq"][v] * t_axis + phase[v])
                )
            X[idx] = X[idx] + jitter
            y[idx] = k
            idx += 1

    X, mask = _apply_missingness(rng, X, missing_ratio)
    perm = rng.permutation(n_samples)
    return (X[perm], mask[perm], y[perm], list(PAM_VARS), "synthetic_pam")


# ─────────────────────────────────────────────────────────────────────────
# Synthetic MIMIC-III demo — in-hospital mortality benchmark (binary)
# ─────────────────────────────────────────────────────────────────────────

MIMIC3_VARS = [
    "Capillary_refill_rate", "Diastolic_BP", "Fraction_inspired_oxygen",
    "Glascow_coma_scale_eye", "Glascow_coma_scale_motor",
    "Glascow_coma_scale_verbal", "Glascow_coma_scale_total",
    "Glucose", "Heart_Rate", "Height", "Mean_BP",
    "Oxygen_saturation", "Respiratory_rate", "Systolic_BP",
    "Temperature", "Weight", "pH",
]


def load_synthetic_mimic3_mortality(
    n_samples: int = 800,
    T: int = 48,
    missing_ratio: float = 0.7,
    pos_ratio: float = 0.13,
    seed: int = 42,
) -> TemporalDataset:
    """Synthetic mimic3-benchmarks-style in-hospital mortality dataset.

    Uses the canonical 17-variable feature set from Harutyunyan et al.
    2019 over the first 48 hours of ICU stay.  Positive (deceased) cases
    show drifting GCS, falling MAP and rising heart rate.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_samples * pos_ratio))
    n_neg = n_samples - n_pos
    y = np.concatenate([np.zeros(n_neg, dtype=np.int64),
                         np.ones(n_pos, dtype=np.int64)])

    n_vars = len(MIMIC3_VARS)
    X = np.zeros((n_samples, T, n_vars), dtype=np.float32)

    means = np.array(
        [1.5, 65, 0.4, 4, 5, 4, 13,
         120, 85, 165, 75, 98, 18, 125, 36.8, 75, 7.4],
        dtype=np.float32,
    )
    stds = np.array(
        [0.5, 8, 0.1, 1, 1, 1, 1,
         15, 10, 10, 8, 1.5, 2.5, 12, 0.4, 6, 0.05],
        dtype=np.float32,
    )
    pos_shift = np.array(
        [+0.8, -10, +0.15, -1.5, -1.5, -1.5, -3,
         +20, +18, 0, -12, -3, +5, -10, +0.5, 0, -0.06],
        dtype=np.float32,
    )

    for i in range(n_samples):
        is_pos = y[i] == 1
        for v in range(n_vars):
            mean = means[v]
            if is_pos:
                trend = np.linspace(0.0, pos_shift[v], T)
                series = _ar1_series(rng, T, mean, stds[v]) + trend
            else:
                series = _ar1_series(rng, T, mean, stds[v])
            X[i, :, v] = series

    X, mask = _apply_missingness(rng, X, missing_ratio)
    perm = rng.permutation(n_samples)
    return (X[perm], mask[perm], y[perm], list(MIMIC3_VARS),
            "synthetic_mimic3_mortality")




# ─────────────────────────────────────────────────────────────────────────
# Processed real ICU mortality caches
# ─────────────────────────────────────────────────────────────────────────

PROCESSED_MORTALITY_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "mortality"
)


def _processed_mortality_dir() -> Path:
    override = os.environ.get("MORTALITY_PROCESSED_DIR")
    return Path(override).expanduser() if override else PROCESSED_MORTALITY_DIR


def load_processed_mortality(name: str, path: Optional[str] = None) -> TemporalDataset:
    """Load a preprocessed real ICU mortality cache.

    Caches are produced by::

        python -m temporal.mortality_preprocess --datasets mimic3 mimic4 eicu

    The NPZ must contain ``X_ts``, ``mask``, ``y`` and ``var_names``.  Real
    source databases are never mixed: each key points to one independent cache.
    """
    cache_path = Path(path).expanduser() if path is not None else (
        _processed_mortality_dir() / f"{name}_mortality_48h_temporal.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing processed mortality cache for {name!r}: {cache_path}. "
            "Build it with `python -m temporal.mortality_preprocess "
            f"--datasets {name}`."
        )
    arr = np.load(cache_path, allow_pickle=True)
    required = {"X_ts", "mask", "y", "var_names"}
    missing = sorted(required - set(arr.files))
    if missing:
        raise ValueError(f"{cache_path} is missing arrays: {missing}")
    X_ts = np.asarray(arr["X_ts"], dtype=np.float32)
    mask = np.asarray(arr["mask"], dtype=np.uint8)
    y = np.asarray(arr["y"], dtype=np.int64)
    var_names = [str(v) for v in arr["var_names"].tolist()]
    if "dataset_name" in arr.files:
        dataset_name = str(np.asarray(arr["dataset_name"]).item())
    else:
        dataset_name = f"{name}_hospital_mortality_48h"
    return X_ts, mask, y, var_names, dataset_name


def load_mimic3_mortality(**kwargs) -> TemporalDataset:
    return load_processed_mortality("mimic3", **kwargs)


def load_mimic4_mortality(**kwargs) -> TemporalDataset:
    return load_processed_mortality("mimic4", **kwargs)


def load_eicu_mortality(**kwargs) -> TemporalDataset:
    return load_processed_mortality("eicu", **kwargs)


# ─────────────────────────────────────────────────────────────────────────
# unified loader
# ─────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Callable[..., TemporalDataset]] = {
    "p12":              load_synthetic_p12,
    "pam":              load_synthetic_pam,
    "mimic3":           load_synthetic_mimic3_mortality,
    "mimic3_mortality": load_mimic3_mortality,
    "mimic4_mortality": load_mimic4_mortality,
    "eicu_mortality":   load_eicu_mortality,
}


def load_temporal_dataset(name: str, **kwargs) -> TemporalDataset:
    """Load a dataset by short name.

    Supported synthetic names: ``p12``, ``pam``, ``mimic3``.
    Processed real mortality caches: ``mimic3_mortality``,
    ``mimic4_mortality`` and ``eicu_mortality``.
    """
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown temporal dataset {name!r}; "
            f"available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[key](**kwargs)
