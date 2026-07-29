"""Internal TabPFN-TS representation extractor.

The public temporal experiments use TabPFN-TS as a black-box teacher or
standalone baseline.  This helper builds the forecasting representation
consumed by that black-box classifier; the exposed interpretable rows
distill soft labels into XGB / ExtraTrees / CatBoost students trained on
ordinary L2/L3 temporal features.

The preferred ``tabpfn_ts`` backend uses the installed
``tabpfn_time_series.TabPFNTSPipeline`` in LOCAL mode.  It expects the
TabPFN-TS checkpoint downloaded by ``download_tabpfn_ts_weights.py`` or a
``TABPFN_TS_MODEL_PATH`` override.  A local ExtraTrees backend is also
exposed so CI and smoke tests can run without gated checkpoints or
network access.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


TEACHER_FEATURES: Tuple[str, ...] = (
    "forecast_next_norm",
    "forecast_delta_norm",
    "residual_abs_mean",
    "residual_std",
    "residual_bias",
    "obs_frac",
)

TABPFN_TS_CHECKPOINT_FILENAME = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"


def default_tabpfn_checkpoint_dir() -> Path:
    """Default local TabPFN cache directory used by downloader and teachers."""
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("tabpfn"))
    except Exception:
        return Path.home() / ".cache" / "tabpfn"


def default_tabpfn_ts_checkpoint_path() -> Path:
    """Default local checkpoint path used by the download script and teacher."""
    return default_tabpfn_checkpoint_dir() / TABPFN_TS_CHECKPOINT_FILENAME


def _validate_ts(X_ts: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    if X_ts.ndim != 3 or mask.ndim != 3 or X_ts.shape != mask.shape:
        raise ValueError("X_ts and mask must both have shape [N, T, V]")
    return X_ts.shape


def _safe_observed_values(X_ts: np.ndarray, mask: np.ndarray, v: int) -> np.ndarray:
    obs = mask[:, :, v].astype(bool) & np.isfinite(X_ts[:, :, v])
    vals = X_ts[:, :, v][obs]
    return vals.astype(np.float64)


@dataclass
class TabPFNTSFeatureTeacher:
    """Forecasting representation builder for TabPFN-TS distillation.

    Parameters
    ----------
    backend:
        ``"tabpfn_ts"`` uses
        :class:`tabpfn_time_series.TabPFNTSPipeline` in LOCAL mode;
        ``"extratrees"`` is a deterministic local backend; ``"auto"``
        tries ``tabpfn_ts`` first and falls back to ExtraTrees with a
        warning.  ``"tabpfn"`` is accepted as a backwards-compatible
        alias for ``"tabpfn_ts"``.
    max_regression_rows:
        Maximum number of transition rows used by the ExtraTrees smoke
        backend.  The native TabPFN-TS backend forecasts all
        sample-variable series.
    """

    backend: str = "auto"
    seed: int = 42
    max_regression_rows: int = 4096
    tabpfn_estimators: int = 8
    tabpfn_ts_model_path: Optional[str] = None
    tabpfn_ts_device: str = "cpu"
    tabpfn_ts_max_context_length: int = 32768
    tabpfn_ts_num_workers: int = 1
    tabpfn_ts_feature_mode: str = "safe"
    local_estimators: int = 128
    min_transition_rows: int = 8
    verbose: bool = False

    backend_used: Optional[str] = field(default=None, init=False)
    n_variables_: Optional[int] = field(default=None, init=False)
    T_: Optional[int] = field(default=None, init=False)
    var_mean_: Optional[np.ndarray] = field(default=None, init=False)
    var_scale_: Optional[np.ndarray] = field(default=None, init=False)
    pipeline_: Optional[object] = field(default=None, init=False)
    regressor_: Optional[object] = field(default=None, init=False)
    feature_names_: List[str] = field(default_factory=list, init=False)

    _VALID_BACKENDS = ("auto", "tabpfn_ts", "tabpfn", "extratrees")

    def __post_init__(self) -> None:
        if self.backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {self._VALID_BACKENDS}, got {self.backend!r}"
            )

    def fit(self, X_ts: np.ndarray, mask: np.ndarray) -> "TabPFNTSFeatureTeacher":
        N, T, V = _validate_ts(X_ts, mask)
        self.T_ = T
        self.n_variables_ = V
        self._fit_variable_scalers(X_ts, mask)
        self.feature_names_ = self.feature_names(
            [f"var_{i}" for i in range(V)]
        )

        if self.backend in ("auto", "tabpfn_ts", "tabpfn"):
            try:
                self.pipeline_ = self._make_tabpfn_ts_pipeline()
                self.backend_used = self._native_backend_label()
                return self
            except Exception as exc:
                if self.backend in ("tabpfn_ts", "tabpfn"):
                    raise RuntimeError(
                        "TabPFN temporal feature teacher failed; install/configure "
                        "`tabpfn-time-series` and download the gated TS "
                        "checkpoint with "
                        "`python download_tabpfn_ts_weights.py --kind ts`, "
                        "or set TABPFN_TS_MODEL_PATH to an existing checkpoint."
                    ) from exc
                warnings.warn(
                    "TabPFN temporal feature teacher fell back to ExtraTrees: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                )

        return self._fit_local_backend(X_ts, mask)

    def _fit_local_backend(
        self, X_ts: np.ndarray, mask: np.ndarray,
    ) -> "TabPFNTSFeatureTeacher":
        rows, targets, _keys = self._transition_table(X_ts, mask)
        self.pipeline_ = None
        if rows.shape[0] < self.min_transition_rows:
            self.backend_used = "constant"
            self.regressor_ = None
            return self

        rng = np.random.default_rng(self.seed)
        if rows.shape[0] > self.max_regression_rows:
            keep = rng.choice(
                rows.shape[0], size=self.max_regression_rows, replace=False,
            )
            rows = rows[keep]
            targets = targets[keep]

        self.regressor_ = ExtraTreesRegressor(
            n_estimators=self.local_estimators,
            random_state=self.seed,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        self.regressor_.fit(rows.astype(np.float32), targets.astype(np.float32))
        self.backend_used = "extratrees"
        return self

    def transform(self, X_ts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._check_fitted()
        N, T, V = _validate_ts(X_ts, mask)
        if T != self.T_ or V != self.n_variables_:
            raise ValueError(
                f"input shape [T={T}, V={V}] does not match fitted "
                f"[T={self.T_}, V={self.n_variables_}]"
            )
        if self.backend_used and self.backend_used.startswith("tabpfn_ts"):
            try:
                return self._transform_native_tabpfn_ts(X_ts, mask)
            except Exception as exc:
                if self.backend in ("tabpfn_ts", "tabpfn"):
                    raise RuntimeError(
                        "Native TabPFN-TS prediction failed. For ICU-style sparse "
                        "series use TABPFN_TS_FEATURE_MODE=safe (default in this "
                        "project) or --ts-teacher-backend auto for a fold-level "
                        "ExtraTrees fallback."
                    ) from exc
                warnings.warn(
                    "TabPFN temporal feature teacher fell back to ExtraTrees "
                    "during prediction: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                )
                self._fit_local_backend(X_ts, mask)

        n_feat = len(TEACHER_FEATURES)
        out = np.zeros((N, V, n_feat), dtype=np.float32)
        out[:, :, TEACHER_FEATURES.index("obs_frac")] = self._obs_fraction(mask)

        last_rows, last_keys, last_values = self._last_observation_rows(X_ts, mask)
        if last_rows.shape[0] > 0:
            pred_last = self._predict_rows(last_rows)
            for pred, (i, v), last_value in zip(pred_last, last_keys, last_values):
                out[i, v, 0] = float(pred)
                out[i, v, 1] = float(pred - last_value)

        trans_rows, trans_targets, trans_keys = self._transition_table(X_ts, mask)
        if trans_rows.shape[0] > 0:
            pred_trans = self._predict_rows(trans_rows)
            residuals: dict[Tuple[int, int], List[float]] = {}
            for pred, target, key in zip(pred_trans, trans_targets, trans_keys):
                residuals.setdefault(key, []).append(float(target - pred))
            for (i, v), vals in residuals.items():
                arr = np.asarray(vals, dtype=np.float64)
                out[i, v, 2] = float(np.abs(arr).mean())
                out[i, v, 3] = float(arr.std()) if arr.size > 1 else 0.0
                out[i, v, 4] = float(arr.mean())

        return out.reshape(N, V * n_feat)

    def fit_transform(self, X_ts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.fit(X_ts, mask).transform(X_ts, mask)

    def feature_names(self, var_names: Sequence[str]) -> List[str]:
        names: List[str] = []
        for v in var_names:
            for f in TEACHER_FEATURES:
                names.append(f"{v}__tabpfn_ts_teacher__{f}")
        return names

    def _fit_variable_scalers(self, X_ts: np.ndarray, mask: np.ndarray) -> None:
        _, _, V = X_ts.shape
        mean = np.zeros(V, dtype=np.float64)
        scale = np.ones(V, dtype=np.float64)
        for v in range(V):
            vals = _safe_observed_values(X_ts, mask, v)
            if vals.size > 0:
                mean[v] = float(vals.mean())
                std = float(vals.std())
                scale[v] = std if std > 1e-8 else 1.0
        self.var_mean_ = mean
        self.var_scale_ = scale

    def _normalise(self, values: np.ndarray, v: int) -> np.ndarray:
        return (values.astype(np.float64) - self.var_mean_[v]) / self.var_scale_[v]

    def _transition_table(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        N, T, V = X_ts.shape
        rows: List[List[float]] = []
        targets: List[float] = []
        keys: List[Tuple[int, int]] = []
        for i in range(N):
            for v in range(V):
                obs = mask[i, :, v].astype(bool) & np.isfinite(X_ts[i, :, v])
                idx = np.flatnonzero(obs)
                if idx.size < 2:
                    continue
                vals = self._normalise(X_ts[i, idx, v], v)
                for pos in range(idx.size - 1):
                    rows.append(self._row_features(v, idx, vals, pos, T))
                    targets.append(float(vals[pos + 1]))
                    keys.append((i, v))
        if not rows:
            return (
                np.zeros((0, self._row_width()), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                [],
            )
        return (
            np.asarray(rows, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            keys,
        )

    def _last_observation_rows(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[float]]:
        N, T, V = X_ts.shape
        rows: List[List[float]] = []
        keys: List[Tuple[int, int]] = []
        last_values: List[float] = []
        for i in range(N):
            for v in range(V):
                obs = mask[i, :, v].astype(bool) & np.isfinite(X_ts[i, :, v])
                idx = np.flatnonzero(obs)
                if idx.size == 0:
                    continue
                vals = self._normalise(X_ts[i, idx, v], v)
                rows.append(self._row_features(v, idx, vals, idx.size - 1, T))
                keys.append((i, v))
                last_values.append(float(vals[-1]))
        if not rows:
            return (
                np.zeros((0, self._row_width()), dtype=np.float32),
                [],
                [],
            )
        return np.asarray(rows, dtype=np.float32), keys, last_values

    def _row_features(
        self,
        variable_idx: int,
        obs_idx: np.ndarray,
        obs_values_norm: np.ndarray,
        pos: int,
        T: int,
    ) -> List[float]:
        v_den = max(int(self.n_variables_ or 1) - 1, 1)
        t_den = max(T - 1, 1)
        value = float(obs_values_norm[pos])
        prev_delta = (
            float(obs_values_norm[pos] - obs_values_norm[pos - 1])
            if pos > 0 else 0.0
        )
        prefix = obs_values_norm[: pos + 1]
        if pos + 1 < obs_idx.size:
            gap = int(obs_idx[pos + 1] - obs_idx[pos])
        elif pos > 0:
            gap = int(np.median(np.diff(obs_idx)))
        else:
            gap = 1
        return [
            float(variable_idx) / float(v_den),
            float(obs_idx[pos]) / float(t_den),
            float(max(gap, 1)) / float(max(T, 1)),
            value,
            prev_delta,
            float(prefix.mean()),
            float(prefix.std()) if prefix.size > 1 else 0.0,
            float(prefix.size) / float(max(T, 1)),
        ]

    @staticmethod
    def _row_width() -> int:
        return 8

    def _obs_fraction(self, mask: np.ndarray) -> np.ndarray:
        return mask.astype(bool).mean(axis=1).astype(np.float32)

    def _make_tabpfn_ts_pipeline(self):
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        if str(self.tabpfn_ts_device).lower() == "cpu":
            os.environ.setdefault("TABPFN_EXCLUDE_DEVICES", "mps")
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        model_path = self._resolve_tabpfn_ts_model_path()
        kwargs = dict(
            max_context_length=self.tabpfn_ts_max_context_length,
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_model_config={
                "model_path": str(model_path),
                "device": self.tabpfn_ts_device,
                "n_estimators": self.tabpfn_estimators,
                "random_state": self.seed,
                "show_progress_bar": False,
            },
        )
        temporal_features = self._tabpfn_ts_temporal_features()
        if temporal_features is not None:
            kwargs["temporal_features"] = temporal_features
        pipeline = TabPFNTSPipeline(**kwargs)
        worker = getattr(getattr(pipeline, "predictor", None), "_worker", None)
        if worker is not None and hasattr(worker, "num_workers"):
            worker.num_workers = max(1, int(self.tabpfn_ts_num_workers))
        return pipeline

    def _tabpfn_ts_feature_mode(self) -> str:
        raw = os.environ.get(
            "TABPFN_TS_FEATURE_MODE", self.tabpfn_ts_feature_mode,
        )
        mode = str(raw).strip().lower().replace("-", "_")
        aliases = {
            "default": "default",
            "safe": "safe",
            "no_auto": "safe",
            "no_auto_seasonal": "safe",
            "auto_no_detrend": "auto_no_detrend",
        }
        if mode not in aliases:
            warnings.warn(
                f"Unknown TABPFN_TS_FEATURE_MODE={raw!r}; using 'safe'.",
                RuntimeWarning,
            )
            return "safe"
        return aliases[mode]

    def _native_backend_label(self) -> str:
        mode = self._tabpfn_ts_feature_mode()
        return "tabpfn_ts" if mode == "default" else f"tabpfn_ts_{mode}"

    def _tabpfn_ts_temporal_features(self):
        mode = self._tabpfn_ts_feature_mode()
        if mode == "default":
            return None
        from tabpfn_time_series.features import (
            AutoSeasonalFeature,
            CalendarFeature,
            RunningIndexFeature,
        )

        features = [RunningIndexFeature(), CalendarFeature()]
        if mode == "auto_no_detrend":
            features.append(AutoSeasonalFeature(config={"do_detrend": False}))
        return features

    def _resolve_tabpfn_ts_model_path(self) -> Path:
        raw = (
            self.tabpfn_ts_model_path
            or os.environ.get("TABPFN_TS_MODEL_PATH")
            or os.environ.get("TABPFN_MODEL_PATH")
        )
        path = Path(raw).expanduser() if raw else default_tabpfn_ts_checkpoint_path()
        if not path.exists():
            raise FileNotFoundError(
                "TabPFN-TS checkpoint not found at "
                f"{path}. Download it with "
                "`python download_tabpfn_ts_weights.py --kind ts` after accepting "
                "https://huggingface.co/Prior-Labs/tabpfn_3, or set "
                "TABPFN_TS_MODEL_PATH."
            )
        return path

    def _transform_native_tabpfn_ts(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        N, _T, V = X_ts.shape
        n_feat = len(TEACHER_FEATURES)
        out = np.zeros((N, V, n_feat), dtype=np.float32)
        out[:, :, TEACHER_FEATURES.index("obs_frac")] = self._obs_fraction(mask)

        context_df, future_df, keys, last_values = self._forecast_frames(
            X_ts, mask, holdout_last=False,
        )
        next_pred = self._predict_tabpfn_ts(context_df, future_df)
        for key, item_id, timestamp in keys:
            row = self._lookup_prediction(next_pred, item_id, timestamp)
            if row is None:
                continue
            i, v = key
            target, _q10, _q50, _q90 = self._prediction_values(row)
            out[i, v, 0] = target
            out[i, v, 1] = target - last_values[(item_id, timestamp)]

        bt_context, bt_future, bt_keys, _bt_last_values = self._forecast_frames(
            X_ts, mask, holdout_last=True,
        )
        bt_pred = self._predict_tabpfn_ts(bt_context, bt_future)
        for key, item_id, timestamp in bt_keys:
            row = self._lookup_prediction(bt_pred, item_id, timestamp)
            if row is None:
                continue
            i, v = key
            target, q10, _q50, q90 = self._prediction_values(row)
            actual = self._actual_value_for_item(X_ts, mask, i, v, timestamp)
            if actual is None:
                continue
            residual = actual - target
            out[i, v, 2] = abs(residual)
            out[i, v, 3] = max(q90 - q10, 0.0) / 2.563
            out[i, v, 4] = residual

        return out.reshape(N, V * n_feat)

    def _forecast_frames(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
        holdout_last: bool,
    ):
        import pandas as pd

        N, T, V = X_ts.shape
        rows = []
        future_rows = []
        keys = []
        last_values: dict[Tuple[str, pd.Timestamp], float] = {}
        origin = pd.Timestamp("2000-01-01")

        for i in range(N):
            for v in range(V):
                obs = mask[i, :, v].astype(bool) & np.isfinite(X_ts[i, :, v])
                idx = np.flatnonzero(obs)
                if idx.size == 0:
                    continue
                if holdout_last and idx.size < 2:
                    continue

                context_idx = idx[:-1] if holdout_last else idx
                future_t = int(idx[-1] if holdout_last else idx[-1] + 1)
                values = self._normalise(X_ts[i, context_idx, v], v)
                item_id = f"s{i}__v{v}"

                for t, value in zip(context_idx, values):
                    rows.append({
                        "item_id": item_id,
                        "timestamp": origin + pd.to_timedelta(int(t), unit="h"),
                        "target": float(value),
                    })

                timestamp = origin + pd.to_timedelta(future_t, unit="h")
                future_rows.append({"item_id": item_id, "timestamp": timestamp})
                keys.append(((i, v), item_id, timestamp))
                last_values[(item_id, timestamp)] = float(values[-1])

        return (
            pd.DataFrame(rows),
            pd.DataFrame(future_rows),
            keys,
            last_values,
        )

    def _predict_tabpfn_ts(self, context_df, future_df):
        if self.pipeline_ is None or context_df.empty or future_df.empty:
            return None
        return self.pipeline_.predict_df(
            context_df,
            future_df=future_df,
            quantiles=[0.1, 0.5, 0.9],
        )

    @staticmethod
    def _lookup_prediction(pred_df, item_id: str, timestamp):
        if pred_df is None:
            return None
        try:
            return pred_df.loc[(item_id, timestamp)]
        except KeyError:
            return None

    @staticmethod
    def _prediction_values(row) -> Tuple[float, float, float, float]:
        def get_col(name, default):
            if name in row:
                return row[name]
            if str(name) in row:
                return row[str(name)]
            return default

        target = float(row["target"])
        q10 = float(get_col(0.1, target))
        q50 = float(get_col(0.5, target))
        q90 = float(get_col(0.9, target))
        return target, q10, q50, q90

    def _actual_value_for_item(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
        i: int,
        v: int,
        timestamp,
    ) -> Optional[float]:
        import pandas as pd

        origin = pd.Timestamp("2000-01-01")
        t = int((timestamp - origin) / pd.Timedelta(hours=1))
        if t < 0 or t >= X_ts.shape[1] or not mask[i, t, v]:
            return None
        return float(self._normalise(np.asarray([X_ts[i, t, v]]), v)[0])

    def _predict_rows(self, rows: np.ndarray) -> np.ndarray:
        if self.regressor_ is None:
            return np.zeros(rows.shape[0], dtype=np.float32)
        pred = self.regressor_.predict(rows.astype(np.float32))
        return np.asarray(pred, dtype=np.float32).reshape(-1)

    def _check_fitted(self) -> None:
        if self.var_mean_ is None or self.var_scale_ is None:
            raise RuntimeError("TabPFNTSFeatureTeacher has not been fitted")
