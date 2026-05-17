"""L1 / L2 — convert irregular multivariate time-series into a flat
feature vector compatible with the existing PPθ-Post static pipeline.

Two modes are exposed:

* ``summary_flatten``:        per-variable summary statistics  (L1)
* ``multi_window_flatten``:   per-(variable, window) summary statistics + deltas
                              between adjacent windows                (L2)

Both modes accept tensors with shape ``[n_samples, T, n_vars]`` and an
explicit ``mask`` tensor.  Missing entries (``mask == 0``) are ignored
when computing statistics.  When all entries of a variable are missing
for a sample, the corresponding cells are filled with ``0`` so that
downstream sklearn estimators can run; an extra ``count`` / ``frac_obs``
feature lets the trees recover this information.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


SUMMARY_STATS = ("mean", "std", "min", "max", "first", "last", "slope",
                 "count", "frac_obs")


def _slope(values: np.ndarray, indices: np.ndarray) -> float:
    """Least-squares slope of values vs. indices (1-D arrays).

    Returns ``0`` when fewer than 2 points are available or when all
    indices coincide.
    """
    n = values.shape[0]
    if n < 2:
        return 0.0
    if indices.std() == 0:
        return 0.0
    a = np.vstack([indices.astype(np.float64), np.ones(n, dtype=np.float64)]).T
    slope, _ = np.linalg.lstsq(a, values.astype(np.float64), rcond=None)[0]
    return float(slope)


def _series_stats(values: np.ndarray, indices: np.ndarray, T: int) -> List[float]:
    """Compute the 9 summary statistics for one (sample, variable) pair.

    ``values`` are observed values only; ``indices`` are the corresponding
    timestep indices (0..T-1).  ``T`` is the total length used for
    ``frac_obs``.  Empty input returns all-zeros.
    """
    n = values.shape[0]
    if n == 0:
        return [0.0] * len(SUMMARY_STATS)
    mean = float(values.mean())
    std = float(values.std()) if n > 1 else 0.0
    vmin = float(values.min())
    vmax = float(values.max())
    first = float(values[0])
    last = float(values[-1])
    slope = _slope(values, indices)
    count = float(n)
    frac_obs = float(n) / float(max(T, 1))
    return [mean, std, vmin, vmax, first, last, slope, count, frac_obs]


def summary_flatten(
    X_ts: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute L1 summary features.

    Parameters
    ----------
    X_ts : np.ndarray, shape [N, T, V]
        Time-series tensor.  Missing entries may be NaN; they are
        identified via ``mask``.
    mask : np.ndarray, shape [N, T, V]
        Binary mask (1 = observed, 0 = missing).

    Returns
    -------
    np.ndarray of shape [N, V * len(SUMMARY_STATS)] (float32).
        Feature order is ``[var_0_mean, var_0_std, ..., var_0_frac_obs,
        var_1_mean, ...]``.
    """
    if X_ts.ndim != 3 or mask.ndim != 3 or X_ts.shape != mask.shape:
        raise ValueError("X_ts and mask must both have shape [N, T, V]")

    N, T, V = X_ts.shape
    n_stats = len(SUMMARY_STATS)
    out = np.zeros((N, V * n_stats), dtype=np.float32)
    for i in range(N):
        for v in range(V):
            obs = mask[i, :, v].astype(bool)
            if not obs.any():
                continue
            indices = np.nonzero(obs)[0]
            values = X_ts[i, indices, v]
            stats = _series_stats(values, indices, T)
            out[i, v * n_stats : (v + 1) * n_stats] = stats
    return out


def summary_feature_names(var_names: Sequence[str]) -> List[str]:
    """Return ``[var_stat]`` names matching the columns of
    :func:`summary_flatten`."""
    names = []
    for v in var_names:
        for s in SUMMARY_STATS:
            names.append(f"{v}__{s}")
    return names


def _default_windows(T: int, n_windows: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, T, n_windows + 1).astype(int)
    windows = []
    for i in range(n_windows):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = a + 1
        windows.append((a, min(b, T)))
    return windows


def multi_window_flatten(
    X_ts: np.ndarray,
    mask: np.ndarray,
    n_windows: int = 4,
    windows: Sequence[Tuple[int, int]] | None = None,
) -> np.ndarray:
    """Compute L2 multi-window features.

    For each variable and each contiguous window ``[t_a, t_b)`` we compute
    the same 9 summary statistics as in L1, then append two cross-window
    *delta* features per variable: ``mean[last] − mean[first]`` and
    ``slope[last] − slope[first]``.  This gives the trees temporal
    structure without any architectural change.

    Parameters
    ----------
    X_ts, mask : as in :func:`summary_flatten`.
    n_windows : int
        Number of equal-width contiguous windows (default 4).  Ignored
        if ``windows`` is given.
    windows : optional list of ``(t_a, t_b)`` tuples
        Explicit window definitions.

    Returns
    -------
    np.ndarray of shape [N, V * (n_windows * 9 + 2)] (float32).
    """
    if X_ts.ndim != 3 or mask.ndim != 3 or X_ts.shape != mask.shape:
        raise ValueError("X_ts and mask must both have shape [N, T, V]")

    N, T, V = X_ts.shape
    win = list(windows) if windows is not None else _default_windows(T, n_windows)
    n_win = len(win)
    n_stats = len(SUMMARY_STATS)

    out = np.zeros((N, V * (n_win * n_stats + 2)), dtype=np.float32)

    for i in range(N):
        for v in range(V):
            base = v * (n_win * n_stats + 2)
            per_window_means: List[float] = []
            per_window_slopes: List[float] = []
            for w_idx, (a, b) in enumerate(win):
                obs = mask[i, a:b, v].astype(bool)
                if obs.any():
                    indices = np.nonzero(obs)[0] + a
                    values = X_ts[i, indices, v]
                    stats = _series_stats(values, indices, T)
                else:
                    stats = [0.0] * n_stats
                per_window_means.append(stats[0])
                per_window_slopes.append(stats[6])
                offset = base + w_idx * n_stats
                out[i, offset : offset + n_stats] = stats
            # delta features (last − first window)
            out[i, base + n_win * n_stats] = (
                per_window_means[-1] - per_window_means[0]
            )
            out[i, base + n_win * n_stats + 1] = (
                per_window_slopes[-1] - per_window_slopes[0]
            )
    return out


def multi_window_feature_names(
    var_names: Sequence[str],
    n_windows: int = 4,
    windows: Sequence[Tuple[int, int]] | None = None,
    T: int | None = None,
) -> List[str]:
    """Return human-readable feature names for :func:`multi_window_flatten`."""
    if windows is None:
        if T is None:
            raise ValueError("Provide either explicit windows or total T")
        windows = _default_windows(T, n_windows)
    names: List[str] = []
    for v in var_names:
        for (a, b) in windows:
            for s in SUMMARY_STATS:
                names.append(f"{v}__w{a}_{b}__{s}")
        names.append(f"{v}__delta_mean_first_last")
        names.append(f"{v}__delta_slope_first_last")
    return names
