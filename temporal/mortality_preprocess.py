"""Build real ICU mortality datasets for PPtheta-Post.

The raw PhysioNet databases in the workspace are relational EHR dumps, not
ready-made ML matrices.  This module creates a consistent binary mortality
benchmark for each source independently:

* MIMIC-III: first ICU stay per subject, target = HOSPITAL_EXPIRE_FLAG.
* MIMIC-IV:  first ICU stay per subject, target = hospital_expire_flag.
* eICU:      first unit stay per unique patient, target = actualhospitalmortality.

For every dataset we emit two cache files:

* ``<dataset>_mortality_48h_temporal.npz`` with ``X_ts``, ``mask``, ``y`` and
  ``var_names`` for ``temporal.compare_temporal``.
* ``<dataset>_mortality_48h_tabular.npz`` with L1 summary features ``X`` and
  ``y`` for ``compare_datasets.py``.

No rows from the raw clinical tables are printed by this script; logs are only
aggregate counts and output paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .tabularize import summary_feature_names, summary_flatten


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "mortality"
DEFAULT_EVENT_CACHE_DIR = PROJECT_DIR / "data" / "processed" / "mortality_event_cache"
EVENT_CACHE_VERSION = 2
EVENT_CACHE_COLUMNS: Tuple[str, ...] = ("sample_idx", "hour", "var_idx", "value")

# Keep the first benchmark intentionally shared and compact: core vitals plus
# high-signal labs that are available in all three databases.
VAR_NAMES: Tuple[str, ...] = (
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "respiratory_rate",
    "temperature",
    "spo2",
    "glucose",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "hematocrit",
    "hemoglobin",
    "platelet",
    "wbc",
    "lactate",
    "bilirubin_total",
)
VAR_INDEX: Dict[str, int] = {name: i for i, name in enumerate(VAR_NAMES)}

# MIMIC-III/IV itemids.  MIMIC-III mixes CareVue and MetaVision itemids;
# MIMIC-IV mostly uses the MetaVision ids.
MIMIC_CHART_ITEMIDS: Mapping[str, Tuple[int, ...]] = {
    "heart_rate": (211, 220045),
    "systolic_bp": (51, 442, 455, 6701, 220179, 220050),
    "diastolic_bp": (8368, 8440, 8441, 8555, 220180, 220051),
    "mean_bp": (456, 52, 6702, 443, 220052, 220181),
    "respiratory_rate": (618, 615, 220210, 224690),
    "temperature": (676, 678, 223761, 223762),
    "spo2": (646, 220277),
    "glucose": (807, 811, 1529, 3744, 3745, 220621, 225664, 226537),
}
TEMP_F_ITEMIDS = {678, 223761}

MIMIC_LAB_ITEMIDS: Mapping[str, Tuple[int, ...]] = {
    "glucose": (50809, 50931),
    "creatinine": (50912,),
    "bun": (51006,),
    "sodium": (50983,),
    "potassium": (50971,),
    "chloride": (50902,),
    "bicarbonate": (50882,),
    "hematocrit": (51221,),
    "hemoglobin": (51222,),
    "platelet": (51265,),
    "wbc": (51300, 51301),
    "lactate": (50813,),
    "bilirubin_total": (50885,),
}

EICU_VITAL_COLUMNS: Mapping[str, str] = {
    "heartrate": "heart_rate",
    "systemicsystolic": "systolic_bp",
    "systemicdiastolic": "diastolic_bp",
    "systemicmean": "mean_bp",
    "respiration": "respiratory_rate",
    "temperature": "temperature",
    "sao2": "spo2",
}

EICU_LAB_ALIASES: Mapping[str, str] = {
    "bedsideglucose": "glucose",
    "glucose": "glucose",
    "creatinine": "creatinine",
    "bun": "bun",
    "bloodureanitrogen": "bun",
    "sodium": "sodium",
    "potassium": "potassium",
    "chloride": "chloride",
    "bicarbonate": "bicarbonate",
    "hco3": "bicarbonate",
    "hematocrit": "hematocrit",
    "hct": "hematocrit",
    "hemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "platelets": "platelet",
    "platelet": "platelet",
    "plateletsx1000": "platelet",
    "wbc": "wbc",
    "wbcx1000": "wbc",
    "whitebloodcellcount": "wbc",
    "lactate": "lactate",
    "totbilirubin": "bilirubin_total",
    "totalbilirubin": "bilirubin_total",
    "bilirubintotal": "bilirubin_total",
}

# Conservative physiologic plausibility ranges. Values outside these bounds are
# treated as data-entry/unit artifacts and excluded before hourly aggregation.
PLAUSIBLE_RANGES: Mapping[str, Tuple[float, float]] = {
    "heart_rate": (20.0, 250.0),
    "systolic_bp": (40.0, 300.0),
    "diastolic_bp": (20.0, 200.0),
    "mean_bp": (20.0, 250.0),
    "respiratory_rate": (2.0, 80.0),
    "temperature": (25.0, 45.0),
    "spo2": (40.0, 100.0),
    "glucose": (10.0, 1000.0),
    "creatinine": (0.1, 80.0),
    "bun": (1.0, 300.0),
    "sodium": (80.0, 200.0),
    "potassium": (1.0, 12.0),
    "chloride": (50.0, 160.0),
    "bicarbonate": (2.0, 80.0),
    "hematocrit": (5.0, 80.0),
    "hemoglobin": (1.0, 30.0),
    "platelet": (1.0, 3000.0),
    "wbc": (0.1, 500.0),
    "lactate": (0.1, 50.0),
    "bilirubin_total": (0.0, 100.0),
}


@dataclass
class CacheResult:
    dataset_key: str
    dataset_name: str
    n_samples: int
    n_positive: int
    temporal_path: Path
    tabular_path: Path
    meta_path: Path


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _normalise_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _as_binary_mortality(series: pd.Series) -> pd.Series:
    """Convert common mortality encodings to nullable int 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").round().astype("Int64")
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out[s.isin(["1", "true", "t", "yes", "y", "expired", "dead", "deceased"])] = 1
    out[s.isin(["0", "false", "f", "no", "n", "alive", "survived"])] = 0
    return out


def _clean_values_for_var(var_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    cleaned = values.astype(np.float32, copy=True)
    for name, idx in VAR_INDEX.items():
        lo, hi = PLAUSIBLE_RANGES[name]
        mask = var_idx == idx
        if mask.any():
            bad = mask & ((cleaned < lo) | (cleaned > hi))
            cleaned[bad] = np.nan
    return cleaned


def _maybe_convert_eicu_temperature(var_name: str, values: np.ndarray) -> np.ndarray:
    if var_name != "temperature":
        return values
    out = values.astype(np.float32, copy=True)
    fahrenheit = np.isfinite(out) & (out > 70.0) & (out < 120.0)
    out[fahrenheit] = (out[fahrenheit] - 32.0) * (5.0 / 9.0)
    return out


def _stratified_limit(df: pd.DataFrame, y_col: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    if not max_samples or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    classes = sorted(df[y_col].dropna().unique().tolist())
    per_class = max(1, max_samples // max(1, len(classes)))
    for cls in classes:
        cls_df = df[df[y_col] == cls]
        take = min(len(cls_df), per_class)
        if take == len(cls_df):
            parts.append(cls_df)
        else:
            idx = rng.choice(cls_df.index.to_numpy(), size=take, replace=False)
            parts.append(cls_df.loc[idx])
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed)
    if len(out) > max_samples:
        out = out.iloc[:max_samples]
    return out.reset_index(drop=True)


def _init_accumulators(n_samples: int, hours: int) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_samples, hours, len(VAR_NAMES)), dtype=np.float32)
    counts = np.zeros((n_samples, hours, len(VAR_NAMES)), dtype=np.uint16)
    return sums, counts


def _finalize_tensor(sums: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = counts > 0
    X = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=X, where=mask)
    return X, mask.astype(np.uint8)


def _event_mask(
    hours_limit: int,
    sample_idx: np.ndarray,
    hour_idx: np.ndarray,
    var_idx: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return (
        (sample_idx >= 0)
        & (hour_idx >= 0)
        & (hour_idx < hours_limit)
        & (var_idx >= 0)
        & np.isfinite(values)
    )


def _add_events(
    sums: np.ndarray,
    counts: np.ndarray,
    sample_idx: np.ndarray,
    hour_idx: np.ndarray,
    var_idx: np.ndarray,
    values: np.ndarray,
) -> int:
    ok = _event_mask(sums.shape[1], sample_idx, hour_idx, var_idx, values)
    if not np.any(ok):
        return 0
    si = sample_idx[ok].astype(np.int64, copy=False)
    hi = hour_idx[ok].astype(np.int64, copy=False)
    vi = var_idx[ok].astype(np.int64, copy=False)
    vals = values[ok].astype(np.float32, copy=False)
    np.add.at(sums, (si, hi, vi), vals)
    np.add.at(counts, (si, hi, vi), 1)
    return int(vals.size)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pyarrow is required for mortality event-cache support. "
            "Install pyarrow or pass --no-event-cache."
        ) from exc
    return pa, pq


def _event_cache_scope(max_samples: Optional[int], seed: int) -> str:
    if max_samples and max_samples > 0:
        return f"sample{int(max_samples)}_seed{int(seed)}"
    return "full"


def _event_cache_base(
    event_cache_dir: Path,
    dataset_key: str,
    hours: int,
    max_samples: Optional[int],
    seed: int,
) -> Path:
    scope = _event_cache_scope(max_samples, seed)
    return event_cache_dir / dataset_key / f"v{EVENT_CACHE_VERSION}_{hours}h_{scope}"


def _event_cache_paths(base: Path, stream_names: Sequence[str]) -> Dict[str, Path]:
    return {name: base / f"{name}.parquet" for name in stream_names}


def _event_cache_meta_path(base: Path) -> Path:
    return base / "meta.json"


def _event_cache_ready(base: Path, stream_names: Sequence[str]) -> bool:
    meta_path = _event_cache_meta_path(base)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if int(meta.get("event_cache_version", -1)) != EVENT_CACHE_VERSION:
        return False
    paths = _event_cache_paths(base, stream_names)
    return all(path.exists() for path in paths.values())


def _write_event_cache_meta(
    base: Path,
    dataset_key: str,
    stream_counts: Mapping[str, int],
    n_samples: int,
    n_positive: int,
    hours: int,
    max_samples: Optional[int],
    seed: int,
    raw_root: Path,
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    meta = {
        "event_cache_version": EVENT_CACHE_VERSION,
        "dataset_key": dataset_key,
        "hours": int(hours),
        "scope": _event_cache_scope(max_samples, seed),
        "max_samples": None if not max_samples or max_samples <= 0 else int(max_samples),
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_positive": int(n_positive),
        "raw_root": str(raw_root),
        "var_names": list(VAR_NAMES),
        "plausible_ranges": {k: list(v) for k, v in PLAUSIBLE_RANGES.items()},
        "columns": list(EVENT_CACHE_COLUMNS),
        "stream_counts": {str(k): int(v) for k, v in stream_counts.items()},
    }
    _event_cache_meta_path(base).write_text(json.dumps(meta, indent=2, sort_keys=True))


class _EventParquetWriter:
    def __init__(self, path: Path):
        pa, pq = _require_pyarrow()
        self.pa = pa
        self.pq = pq
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema(
            [
                ("sample_idx", pa.int32()),
                ("hour", pa.int16()),
                ("var_idx", pa.int16()),
                ("value", pa.float32()),
            ]
        )
        self.writer = None
        self.rows = 0

    def write(
        self,
        sample_idx: np.ndarray,
        hour_idx: np.ndarray,
        var_idx: np.ndarray,
        values: np.ndarray,
    ) -> None:
        if sample_idx.size == 0:
            return
        table = self.pa.table(
            {
                "sample_idx": sample_idx.astype(np.int32, copy=False),
                "hour": hour_idx.astype(np.int16, copy=False),
                "var_idx": var_idx.astype(np.int16, copy=False),
                "value": values.astype(np.float32, copy=False),
            },
            schema=self.schema,
        )
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=False,
            )
        self.writer.write_table(table)
        self.rows += int(sample_idx.size)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            return
        empty = self.pa.table(
            {
                "sample_idx": np.asarray([], dtype=np.int32),
                "hour": np.asarray([], dtype=np.int16),
                "var_idx": np.asarray([], dtype=np.int16),
                "value": np.asarray([], dtype=np.float32),
            },
            schema=self.schema,
        )
        self.pq.write_table(empty, self.path, compression="zstd", use_dictionary=False)

    def __enter__(self) -> "_EventParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _add_events_and_cache(
    sums: np.ndarray,
    counts: np.ndarray,
    sample_idx: np.ndarray,
    hour_idx: np.ndarray,
    var_idx: np.ndarray,
    values: np.ndarray,
    writer: Optional[_EventParquetWriter] = None,
) -> int:
    ok = _event_mask(sums.shape[1], sample_idx, hour_idx, var_idx, values)
    if not np.any(ok):
        return 0
    si = sample_idx[ok].astype(np.int64, copy=False)
    hi = hour_idx[ok].astype(np.int64, copy=False)
    vi = var_idx[ok].astype(np.int64, copy=False)
    vals = values[ok].astype(np.float32, copy=False)
    np.add.at(sums, (si, hi, vi), vals)
    np.add.at(counts, (si, hi, vi), 1)
    if writer is not None:
        writer.write(
            si.astype(np.int32, copy=False),
            hi.astype(np.int16, copy=False),
            vi.astype(np.int16, copy=False),
            vals,
        )
    return int(vals.size)


def _read_event_cache_stream(path: Path, sums: np.ndarray, counts: np.ndarray, batch_size: int) -> int:
    pa, pq = _require_pyarrow()
    total = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(EVENT_CACHE_COLUMNS)):
        sample_idx = batch.column(0).to_numpy(zero_copy_only=False)
        hour_idx = batch.column(1).to_numpy(zero_copy_only=False)
        var_idx = batch.column(2).to_numpy(zero_copy_only=False)
        values = batch.column(3).to_numpy(zero_copy_only=False)
        total += _add_events(sums, counts, sample_idx, hour_idx, var_idx, values)
    return total


def _load_event_cache(
    base: Path,
    stream_names: Sequence[str],
    sums: np.ndarray,
    counts: np.ndarray,
    batch_size: int,
) -> Dict[str, int]:
    print(f"reading mortality event-cache: {base}", flush=True)
    out: Dict[str, int] = {}
    for name, path in _event_cache_paths(base, stream_names).items():
        out[name] = _read_event_cache_stream(path, sums, counts, batch_size)
    return out

def _cache_paths(dataset_key: str, output_dir: Path, hours: int) -> Tuple[Path, Path, Path]:
    stem = f"{dataset_key}_mortality_{hours}h"
    return (
        output_dir / f"{stem}_temporal.npz",
        output_dir / f"{stem}_tabular.npz",
        output_dir / f"{stem}_meta.json",
    )


def _existing_result(dataset_key: str, output_dir: Path, hours: int) -> Optional[CacheResult]:
    temporal_path, tabular_path, meta_path = _cache_paths(dataset_key, output_dir, hours)
    if not (temporal_path.exists() and tabular_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    return CacheResult(
        dataset_key=dataset_key,
        dataset_name=str(meta.get("dataset_name", f"{dataset_key}_hospital_mortality_{hours}h")),
        n_samples=int(meta.get("n_samples", 0)),
        n_positive=int(meta.get("n_positive", 0)),
        temporal_path=temporal_path,
        tabular_path=tabular_path,
        meta_path=meta_path,
    )


def _save_outputs(
    dataset_key: str,
    dataset_name: str,
    X_ts: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    hours: int,
    meta: dict,
) -> CacheResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_path, tabular_path, meta_path = _cache_paths(dataset_key, output_dir, hours)

    np.savez_compressed(
        temporal_path,
        X_ts=X_ts.astype(np.float32),
        mask=mask.astype(np.uint8),
        y=y.astype(np.int64),
        var_names=np.asarray(VAR_NAMES, dtype=str),
        dataset_name=np.asarray(dataset_name, dtype=str),
    )
    X_tab = summary_flatten(X_ts, mask)
    np.savez_compressed(
        tabular_path,
        X=X_tab.astype(np.float32),
        y=y.astype(np.int64),
        feature_names=np.asarray(summary_feature_names(VAR_NAMES), dtype=str),
        class_names=np.asarray(["survived", "died"], dtype=str),
        dataset_name=np.asarray(dataset_name, dtype=str),
    )
    meta_out = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "hours": hours,
        "n_samples": int(y.shape[0]),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()) if y.size else math.nan,
        "var_names": list(VAR_NAMES),
        "plausible_ranges": {k: list(v) for k, v in PLAUSIBLE_RANGES.items()},
        **meta,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, sort_keys=True))
    return CacheResult(
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        n_samples=int(y.shape[0]),
        n_positive=int(y.sum()),
        temporal_path=temporal_path,
        tabular_path=tabular_path,
        meta_path=meta_path,
    )


def _mimic_item_maps() -> Tuple[Dict[int, int], Dict[int, int]]:
    chart_map: Dict[int, int] = {}
    for name, ids in MIMIC_CHART_ITEMIDS.items():
        for itemid in ids:
            chart_map[int(itemid)] = VAR_INDEX[name]
    lab_map: Dict[int, int] = {}
    for name, ids in MIMIC_LAB_ITEMIDS.items():
        for itemid in ids:
            lab_map[int(itemid)] = VAR_INDEX[name]
    return chart_map, lab_map


def _hours_from_time(
    time_values: pd.Series,
    key_values: pd.Series,
    key_to_intime_ns: Mapping[int, int],
) -> np.ndarray:
    chart_ns = pd.to_datetime(
        time_values, errors="coerce", format="%Y-%m-%d %H:%M:%S"
    ).astype("int64").to_numpy()
    intime_ns = key_values.map(key_to_intime_ns).to_numpy(dtype="float64", na_value=np.nan)
    # pandas NaT is the minimum int64; mark it invalid through NaN hour.
    invalid = chart_ns == np.iinfo(np.int64).min
    hour = np.floor((chart_ns.astype("float64") - intime_ns) / (3600.0 * 1e9))
    hour[invalid | ~np.isfinite(hour)] = -1
    return hour.astype(np.int32, copy=False)


def _process_mimic_chart_events(
    path: Path,
    id_col: str,
    time_col: str,
    item_col: str,
    value_col: str,
    key_to_idx: Mapping[int, int],
    key_to_intime_ns: Mapping[int, int],
    item_to_var: Mapping[int, int],
    sums: np.ndarray,
    counts: np.ndarray,
    chunksize: int,
    value_is_temp_f: Optional[set[int]] = None,
    event_writer: Optional[_EventParquetWriter] = None,
) -> int:
    usecols = [id_col, time_col, item_col, value_col]
    total = 0
    wanted = set(item_to_var)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk[item_col].isin(wanted)]
        if chunk.empty:
            continue
        idx = chunk[id_col].map(key_to_idx).to_numpy(dtype="float64", na_value=np.nan)
        keep = np.isfinite(idx)
        if not keep.any():
            continue
        chunk = chunk.loc[keep]
        idx = idx[keep].astype(np.int32, copy=False)
        hours = _hours_from_time(chunk[time_col], chunk[id_col], key_to_intime_ns)
        values = pd.to_numeric(chunk[value_col], errors="coerce").to_numpy(dtype=np.float32)
        itemids = chunk[item_col].astype(int).to_numpy()
        if value_is_temp_f:
            temp_f = np.isin(itemids, list(value_is_temp_f)) & np.isfinite(values)
            values[temp_f] = (values[temp_f] - 32.0) * (5.0 / 9.0)
        var_idx = np.asarray([item_to_var[int(item)] for item in itemids], dtype=np.int32)
        values = _clean_values_for_var(var_idx, values)
        total += _add_events_and_cache(
            sums, counts, idx, hours, var_idx, values, event_writer
        )
    return total


def _process_mimic_labs(
    path: Path,
    hadm_col: str,
    time_col: str,
    item_col: str,
    value_col: str,
    hadm_to_idx: Mapping[int, int],
    hadm_to_intime_ns: Mapping[int, int],
    item_to_var: Mapping[int, int],
    sums: np.ndarray,
    counts: np.ndarray,
    chunksize: int,
    event_writer: Optional[_EventParquetWriter] = None,
) -> int:
    return _process_mimic_chart_events(
        path=path,
        id_col=hadm_col,
        time_col=time_col,
        item_col=item_col,
        value_col=value_col,
        key_to_idx=hadm_to_idx,
        key_to_intime_ns=hadm_to_intime_ns,
        item_to_var=item_to_var,
        sums=sums,
        counts=counts,
        chunksize=chunksize,
        value_is_temp_f=None,
        event_writer=event_writer,
    )


def build_mimic3(
    raw_root: Path,
    output_dir: Path,
    hours: int,
    chunksize: int,
    max_samples: Optional[int],
    seed: int,
    event_cache_dir: Optional[Path],
    rebuild_event_cache: bool,
) -> CacheResult:
    root = raw_root / "MIMIC-III"
    admissions = _read_csv(root / "ADMISSIONS.csv.gz", usecols=["SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG"])
    icu = _read_csv(root / "ICUSTAYS.csv", usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"])
    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="coerce")
    stays = icu.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    stays = stays.dropna(subset=["INTIME", "ICUSTAY_ID", "HADM_ID", "SUBJECT_ID", "HOSPITAL_EXPIRE_FLAG"])
    stays = stays.sort_values(["SUBJECT_ID", "INTIME", "ICUSTAY_ID"]).groupby("SUBJECT_ID", as_index=False).first()
    stays["y"] = _as_binary_mortality(stays["HOSPITAL_EXPIRE_FLAG"])
    stays = stays.dropna(subset=["y"]).copy()
    stays["y"] = stays["y"].astype(int)
    stays = _stratified_limit(stays, "y", max_samples, seed)
    stays["sample_idx"] = np.arange(len(stays), dtype=np.int32)

    sums, counts = _init_accumulators(len(stays), hours)
    chart_map, lab_map = _mimic_item_maps()
    stay_to_idx = dict(zip(stays["ICUSTAY_ID"].astype(int), stays["sample_idx"].astype(int)))
    stay_to_intime_ns = dict(zip(stays["ICUSTAY_ID"].astype(int), stays["INTIME"].astype("int64")))
    hadm_to_idx = dict(zip(stays["HADM_ID"].astype(int), stays["sample_idx"].astype(int)))
    hadm_to_intime_ns = dict(zip(stays["HADM_ID"].astype(int), stays["INTIME"].astype("int64")))

    print(f"selected stays: n={len(stays)} positives={int(stays['y'].sum())}", flush=True)
    stream_names = ("chart", "lab")
    event_base = (
        _event_cache_base(event_cache_dir, "mimic3", hours, max_samples, seed)
        if event_cache_dir is not None
        else None
    )
    event_cache_used = False
    if event_base is not None and _event_cache_ready(event_base, stream_names) and not rebuild_event_cache:
        stream_counts = _load_event_cache(event_base, stream_names, sums, counts, chunksize)
        chart_events = stream_counts["chart"]
        lab_events = stream_counts["lab"]
        event_cache_used = True
    else:
        paths = _event_cache_paths(event_base, stream_names) if event_base is not None else {}
        chart_writer = _EventParquetWriter(paths["chart"]) if event_base is not None else None
        lab_writer = _EventParquetWriter(paths["lab"]) if event_base is not None else None
        try:
            print("reading MIMIC-III CHARTEVENTS...", flush=True)
            chart_events = _process_mimic_chart_events(
                root / "CHARTEVENTS.csv.gz",
                id_col="ICUSTAY_ID",
                time_col="CHARTTIME",
                item_col="ITEMID",
                value_col="VALUENUM",
                key_to_idx=stay_to_idx,
                key_to_intime_ns=stay_to_intime_ns,
                item_to_var=chart_map,
                sums=sums,
                counts=counts,
                chunksize=chunksize,
                value_is_temp_f=TEMP_F_ITEMIDS,
                event_writer=chart_writer,
            )
            print("reading MIMIC-III LABEVENTS...", flush=True)
            lab_events = _process_mimic_labs(
                root / "LABEVENTS.csv",
                hadm_col="HADM_ID",
                time_col="CHARTTIME",
                item_col="ITEMID",
                value_col="VALUENUM",
                hadm_to_idx=hadm_to_idx,
                hadm_to_intime_ns=hadm_to_intime_ns,
                item_to_var=lab_map,
                sums=sums,
                counts=counts,
                chunksize=chunksize,
                event_writer=lab_writer,
            )
        finally:
            if chart_writer is not None:
                chart_writer.close()
            if lab_writer is not None:
                lab_writer.close()
        if event_base is not None:
            _write_event_cache_meta(
                event_base,
                "mimic3",
                {"chart": chart_events, "lab": lab_events},
                len(stays),
                int(stays["y"].sum()),
                hours,
                max_samples,
                seed,
                root,
            )
            print(f"wrote mortality event-cache: {event_base}", flush=True)
    X_ts, mask = _finalize_tensor(sums, counts)
    return _save_outputs(
        "mimic3",
        "mimic3_hospital_mortality_48h",
        X_ts,
        mask,
        stays["y"].to_numpy(dtype=np.int64),
        output_dir,
        hours,
        {
            "raw_root": str(root),
            "chart_events_seen": chart_events,
            "lab_events_seen": lab_events,
            "event_cache_path": str(event_base) if event_base is not None else None,
            "event_cache_used": event_cache_used,
        },
    )


def build_mimic4(
    raw_root: Path,
    output_dir: Path,
    hours: int,
    chunksize: int,
    max_samples: Optional[int],
    seed: int,
    event_cache_dir: Optional[Path],
    rebuild_event_cache: bool,
) -> CacheResult:
    root = raw_root / "mimic-4" / "physionet.org" / "files" / "mimiciv" / "3.1"
    admissions = _read_csv(root / "hosp" / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "hospital_expire_flag"])
    icu = _read_csv(root / "icu" / "icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"])
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    stays = icu.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    stays = stays.dropna(subset=["intime", "stay_id", "hadm_id", "subject_id", "hospital_expire_flag"])
    stays = stays.sort_values(["subject_id", "intime", "stay_id"]).groupby("subject_id", as_index=False).first()
    stays["y"] = _as_binary_mortality(stays["hospital_expire_flag"])
    stays = stays.dropna(subset=["y"]).copy()
    stays["y"] = stays["y"].astype(int)
    stays = _stratified_limit(stays, "y", max_samples, seed)
    stays["sample_idx"] = np.arange(len(stays), dtype=np.int32)

    sums, counts = _init_accumulators(len(stays), hours)
    chart_map, lab_map = _mimic_item_maps()
    stay_to_idx = dict(zip(stays["stay_id"].astype(int), stays["sample_idx"].astype(int)))
    stay_to_intime_ns = dict(zip(stays["stay_id"].astype(int), stays["intime"].astype("int64")))
    hadm_to_idx = dict(zip(stays["hadm_id"].astype(int), stays["sample_idx"].astype(int)))
    hadm_to_intime_ns = dict(zip(stays["hadm_id"].astype(int), stays["intime"].astype("int64")))

    print(f"selected stays: n={len(stays)} positives={int(stays['y'].sum())}", flush=True)
    stream_names = ("chart", "lab")
    event_base = (
        _event_cache_base(event_cache_dir, "mimic4", hours, max_samples, seed)
        if event_cache_dir is not None
        else None
    )
    event_cache_used = False
    if event_base is not None and _event_cache_ready(event_base, stream_names) and not rebuild_event_cache:
        stream_counts = _load_event_cache(event_base, stream_names, sums, counts, chunksize)
        chart_events = stream_counts["chart"]
        lab_events = stream_counts["lab"]
        event_cache_used = True
    else:
        paths = _event_cache_paths(event_base, stream_names) if event_base is not None else {}
        chart_writer = _EventParquetWriter(paths["chart"]) if event_base is not None else None
        lab_writer = _EventParquetWriter(paths["lab"]) if event_base is not None else None
        try:
            print("reading MIMIC-IV icu/chartevents...", flush=True)
            chart_events = _process_mimic_chart_events(
                root / "icu" / "chartevents.csv.gz",
                id_col="stay_id",
                time_col="charttime",
                item_col="itemid",
                value_col="valuenum",
                key_to_idx=stay_to_idx,
                key_to_intime_ns=stay_to_intime_ns,
                item_to_var=chart_map,
                sums=sums,
                counts=counts,
                chunksize=chunksize,
                value_is_temp_f=TEMP_F_ITEMIDS,
                event_writer=chart_writer,
            )
            print("reading MIMIC-IV hosp/labevents...", flush=True)
            lab_events = _process_mimic_labs(
                root / "hosp" / "labevents.csv.gz",
                hadm_col="hadm_id",
                time_col="charttime",
                item_col="itemid",
                value_col="valuenum",
                hadm_to_idx=hadm_to_idx,
                hadm_to_intime_ns=hadm_to_intime_ns,
                item_to_var=lab_map,
                sums=sums,
                counts=counts,
                chunksize=chunksize,
                event_writer=lab_writer,
            )
        finally:
            if chart_writer is not None:
                chart_writer.close()
            if lab_writer is not None:
                lab_writer.close()
        if event_base is not None:
            _write_event_cache_meta(
                event_base,
                "mimic4",
                {"chart": chart_events, "lab": lab_events},
                len(stays),
                int(stays["y"].sum()),
                hours,
                max_samples,
                seed,
                root,
            )
            print(f"wrote mortality event-cache: {event_base}", flush=True)
    X_ts, mask = _finalize_tensor(sums, counts)
    return _save_outputs(
        "mimic4",
        "mimic4_hospital_mortality_48h",
        X_ts,
        mask,
        stays["y"].to_numpy(dtype=np.int64),
        output_dir,
        hours,
        {
            "raw_root": str(root),
            "chart_events_seen": chart_events,
            "lab_events_seen": lab_events,
            "event_cache_path": str(event_base) if event_base is not None else None,
            "event_cache_used": event_cache_used,
        },
    )


def _process_eicu_vitals(
    path: Path,
    stay_to_idx: Mapping[int, int],
    sums: np.ndarray,
    counts: np.ndarray,
    chunksize: int,
    event_writer: Optional[_EventParquetWriter] = None,
) -> int:
    usecols = ["patientunitstayid", "observationoffset", *EICU_VITAL_COLUMNS.keys()]
    total = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        idx = chunk["patientunitstayid"].map(stay_to_idx).to_numpy(dtype="float64", na_value=np.nan)
        keep = np.isfinite(idx)
        if not keep.any():
            continue
        chunk = chunk.loc[keep]
        idx = idx[keep].astype(np.int32, copy=False)
        offsets = pd.to_numeric(chunk["observationoffset"], errors="coerce").to_numpy(dtype=np.float64)
        hours = np.floor(offsets / 60.0).astype(np.int32, copy=False)
        for col, var_name in EICU_VITAL_COLUMNS.items():
            values = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=np.float32)
            values = _maybe_convert_eicu_temperature(var_name, values)
            var_idx = np.full(values.shape, VAR_INDEX[var_name], dtype=np.int32)
            values = _clean_values_for_var(var_idx, values)
            total += _add_events_and_cache(
                sums, counts, idx, hours, var_idx, values, event_writer
            )
    return total


def _process_eicu_labs(
    path: Path,
    stay_to_idx: Mapping[int, int],
    sums: np.ndarray,
    counts: np.ndarray,
    chunksize: int,
    event_writer: Optional[_EventParquetWriter] = None,
) -> int:
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    total = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        idx = chunk["patientunitstayid"].map(stay_to_idx).to_numpy(dtype="float64", na_value=np.nan)
        keep = np.isfinite(idx)
        if not keep.any():
            continue
        chunk = chunk.loc[keep].copy()
        idx = idx[keep].astype(np.int32, copy=False)
        norm = chunk["labname"].map(_normalise_name)
        var_names = norm.map(EICU_LAB_ALIASES)
        keep_lab = var_names.notna().to_numpy()
        if not keep_lab.any():
            continue
        idx = idx[keep_lab]
        chunk = chunk.loc[keep_lab]
        var_names = var_names.loc[keep_lab]
        offsets = pd.to_numeric(chunk["labresultoffset"], errors="coerce").to_numpy(dtype=np.float64)
        hours = np.floor(offsets / 60.0).astype(np.int32, copy=False)
        values = pd.to_numeric(chunk["labresult"], errors="coerce").to_numpy(dtype=np.float32)
        var_idx = np.asarray([VAR_INDEX[str(v)] for v in var_names], dtype=np.int32)
        values = _clean_values_for_var(var_idx, values)
        total += _add_events_and_cache(
            sums, counts, idx, hours, var_idx, values, event_writer
        )
    return total


def build_eicu(
    raw_root: Path,
    output_dir: Path,
    hours: int,
    chunksize: int,
    max_samples: Optional[int],
    seed: int,
    event_cache_dir: Optional[Path],
    rebuild_event_cache: bool,
) -> CacheResult:
    root = raw_root / "eICU" / "physionet.org" / "files" / "eicu-crd" / "2.0"
    patient = _read_csv(
        root / "patient.csv.gz",
        usecols=["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "unitvisitnumber"],
        low_memory=False,
    )
    apache = _read_csv(root / "apachePatientResult.csv.gz", usecols=["patientunitstayid", "actualhospitalmortality"], low_memory=False)
    stays = patient.merge(apache, on="patientunitstayid", how="inner")
    stays["y"] = _as_binary_mortality(stays["actualhospitalmortality"])
    stays = stays.dropna(subset=["patientunitstayid", "uniquepid", "y"]).copy()
    stays["y"] = stays["y"].astype(int)
    stays["unitvisitnumber"] = pd.to_numeric(stays["unitvisitnumber"], errors="coerce").fillna(0)
    stays = stays.sort_values(["uniquepid", "unitvisitnumber", "patientunitstayid"]).groupby("uniquepid", as_index=False).first()
    stays = _stratified_limit(stays, "y", max_samples, seed)
    stays["sample_idx"] = np.arange(len(stays), dtype=np.int32)

    sums, counts = _init_accumulators(len(stays), hours)
    stay_to_idx = dict(zip(stays["patientunitstayid"].astype(int), stays["sample_idx"].astype(int)))
    print(f"selected stays: n={len(stays)} positives={int(stays['y'].sum())}", flush=True)
    stream_names = ("vitals", "lab")
    event_base = (
        _event_cache_base(event_cache_dir, "eicu", hours, max_samples, seed)
        if event_cache_dir is not None
        else None
    )
    event_cache_used = False
    if event_base is not None and _event_cache_ready(event_base, stream_names) and not rebuild_event_cache:
        stream_counts = _load_event_cache(event_base, stream_names, sums, counts, chunksize)
        vital_events = stream_counts["vitals"]
        lab_events = stream_counts["lab"]
        event_cache_used = True
    else:
        paths = _event_cache_paths(event_base, stream_names) if event_base is not None else {}
        vital_writer = _EventParquetWriter(paths["vitals"]) if event_base is not None else None
        lab_writer = _EventParquetWriter(paths["lab"]) if event_base is not None else None
        try:
            print("reading eICU vitalPeriodic...", flush=True)
            vital_events = _process_eicu_vitals(
                root / "vitalPeriodic.csv.gz",
                stay_to_idx,
                sums,
                counts,
                chunksize,
                event_writer=vital_writer,
            )
            print("reading eICU lab...", flush=True)
            lab_events = _process_eicu_labs(
                root / "lab.csv.gz",
                stay_to_idx,
                sums,
                counts,
                chunksize,
                event_writer=lab_writer,
            )
        finally:
            if vital_writer is not None:
                vital_writer.close()
            if lab_writer is not None:
                lab_writer.close()
        if event_base is not None:
            _write_event_cache_meta(
                event_base,
                "eicu",
                {"vitals": vital_events, "lab": lab_events},
                len(stays),
                int(stays["y"].sum()),
                hours,
                max_samples,
                seed,
                root,
            )
            print(f"wrote mortality event-cache: {event_base}", flush=True)
    X_ts, mask = _finalize_tensor(sums, counts)
    return _save_outputs(
        "eicu",
        "eicu_hospital_mortality_48h",
        X_ts,
        mask,
        stays["y"].to_numpy(dtype=np.int64),
        output_dir,
        hours,
        {
            "raw_root": str(root),
            "vital_events_seen": vital_events,
            "lab_events_seen": lab_events,
            "event_cache_path": str(event_base) if event_base is not None else None,
            "event_cache_used": event_cache_used,
        },
    )


BUILDERS = {
    "mimic3": build_mimic3,
    "mimic4": build_mimic4,
    "eicu": build_eicu,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["all"], choices=["all", *BUILDERS.keys()])
    p.add_argument("--raw-root", default=str(WORKSPACE_DIR), help="Workspace directory containing MIMIC-III, mimic-4 and eICU folders.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--max-samples", type=int, default=None, help="Optional stratified sample cap for smoke runs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--event-cache-dir",
        default=os.environ.get("MORTALITY_EVENT_CACHE_DIR", str(DEFAULT_EVENT_CACHE_DIR)),
        help="Directory for filtered Parquet event caches used before final NPZ creation.",
    )
    p.add_argument(
        "--no-event-cache",
        action="store_true",
        help="Read raw event tables directly and do not write/read Parquet event caches.",
    )
    p.add_argument(
        "--rebuild-event-cache",
        action="store_true",
        help="Rescan raw event tables and overwrite the filtered Parquet event cache.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild final temporal/tabular NPZ caches even if they already exist.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selected = list(BUILDERS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    event_cache_dir = None if args.no_event_cache else Path(args.event_cache_dir).expanduser().resolve()
    print(f"Building mortality caches: datasets={selected} hours={args.hours} output={output_dir}")
    if event_cache_dir is not None:
        print(f"event-cache: {event_cache_dir}")
    else:
        print("event-cache: disabled")
    for key in selected:
        print(f"\n=== {key} ===")
        result = None if (args.force or args.rebuild_event_cache) else _existing_result(key, output_dir, args.hours)
        if result is not None:
            print("cache exists; skipping raw-table preprocessing (use --force to rebuild)", flush=True)
        else:
            result = BUILDERS[key](
                raw_root=raw_root,
                output_dir=output_dir,
                hours=args.hours,
                chunksize=args.chunksize,
                max_samples=args.max_samples,
                seed=args.seed,
                event_cache_dir=event_cache_dir,
                rebuild_event_cache=args.rebuild_event_cache,
            )
        print(
            f"{result.dataset_name}: n={result.n_samples} positives={result.n_positive} "
            f"rate={result.n_positive / max(result.n_samples, 1):.4f}"
        )
        print(f"temporal: {result.temporal_path}")
        print(f"tabular:  {result.tabular_path}")
        print(f"meta:     {result.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
