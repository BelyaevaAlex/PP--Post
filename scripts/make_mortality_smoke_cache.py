#!/usr/bin/env python3
"""Create small stratified mortality NPZ caches from the full processed caches.

This does not read raw ICU tables. It samples existing v2 full mortality caches
and writes the same temporal/tabular/meta format expected by the experiment
entrypoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from temporal.tabularize import summary_feature_names, summary_flatten

DATASETS = ("mimic3", "mimic4", "eicu")


def _sample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or n >= len(y):
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    parts = []
    classes = sorted(np.unique(y).tolist())
    per_class = max(1, n // max(1, len(classes)))
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(len(idx), per_class)
        parts.append(rng.choice(idx, size=take, replace=False) if take < len(idx) else idx)
    out = np.concatenate(parts)
    rng.shuffle(out)
    if len(out) > n:
        out = out[:n]
    return np.sort(out.astype(np.int64, copy=False))


def build_one(dataset: str, source_dir: Path, output_dir: Path, n: int, seed: int) -> None:
    src = source_dir / f"{dataset}_mortality_48h_temporal.npz"
    if not src.exists():
        raise FileNotFoundError(src)
    arr = np.load(src, allow_pickle=True)
    X_ts = np.asarray(arr["X_ts"], dtype=np.float32)
    mask = np.asarray(arr["mask"], dtype=np.uint8)
    y = np.asarray(arr["y"], dtype=np.int64)
    var_names = [str(v) for v in arr["var_names"].tolist()]
    dataset_name = str(np.asarray(arr["dataset_name"]).item())

    idx = _sample_indices(y, n, seed)
    X_s = X_ts[idx]
    mask_s = mask[idx]
    y_s = y[idx]
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_path = output_dir / f"{dataset}_mortality_48h_temporal.npz"
    tabular_path = output_dir / f"{dataset}_mortality_48h_tabular.npz"
    meta_path = output_dir / f"{dataset}_mortality_48h_meta.json"

    np.savez_compressed(
        temporal_path,
        X_ts=X_s,
        mask=mask_s,
        y=y_s,
        var_names=np.asarray(var_names, dtype=str),
        dataset_name=np.asarray(dataset_name, dtype=str),
    )
    X_tab = summary_flatten(X_s, mask_s)
    np.savez_compressed(
        tabular_path,
        X=X_tab.astype(np.float32),
        y=y_s,
        feature_names=np.asarray(summary_feature_names(var_names), dtype=str),
        class_names=np.asarray(["survived", "died"], dtype=str),
        dataset_name=np.asarray(f"{dataset_name}_smoke{len(idx)}", dtype=str),
    )
    meta_path.write_text(json.dumps({
        "dataset_key": dataset,
        "dataset_name": f"{dataset_name}_smoke{len(idx)}",
        "source_dir": str(source_dir),
        "n_samples": int(len(idx)),
        "n_positive": int(y_s.sum()),
        "positive_rate": float(y_s.mean()) if len(y_s) else None,
        "indices_seed": int(seed),
        "source_n_samples": int(len(y)),
        "hours": int(X_s.shape[1]),
        "var_names": var_names,
    }, indent=2, sort_keys=True))
    print(
        f"{dataset}: wrote {len(idx)} samples, positives={int(y_s.sum())}, "
        f"temporal={temporal_path}, tabular={tabular_path}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS) + ["all"])
    ap.add_argument("--source-dir", default="data/processed/mortality")
    ap.add_argument("--output-dir", default="data/processed/mortality_job_smoke")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    datasets = list(DATASETS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    for ds in datasets:
        build_one(ds, Path(args.source_dir), Path(args.output_dir), args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
