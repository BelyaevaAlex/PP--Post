"""§5.4 — run all 17 static PPθ-Post inference variants
(``DH7-800→N``, ``PL-βAdMatch-wm``, ``PL-ens-3way`` …) on temporal
benchmarks by flattening the time-series with L1 / L2 / L3 first.

The static driver lives in :mod:`compare_datasets` and operates on a
``(X, y, feature_names, class_names)`` tuple.  This module is a thin
adapter that:

1. Loads a temporal dataset via :mod:`temporal.datasets`.
2. Flattens the ``[N, T, V]`` tensor into a plain feature matrix using
   :func:`summary_flatten` (L1), :func:`multi_window_flatten` (L2) or
   :class:`IntervalFeatureExtractor` (L3).
3. Calls :func:`compare_datasets.run_comparison` so all 17 static
   variants run on the resulting static dataset.

The printed log of each run is duplicated to a separate file under
``output/temporal/static_on_temporal_<level>_<dataset>_<timestamp>.txt``
so that direct comparison with the synthetic ablation runs is possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from io import StringIO
from typing import List, Optional, Sequence

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import compare_datasets as cd  # noqa: E402

from .datasets import load_temporal_dataset  # noqa: E402
from .interval_forest import IntervalFeatureExtractor  # noqa: E402
from .tabularize import (  # noqa: E402
    multi_window_feature_names,
    multi_window_flatten,
    summary_feature_names,
    summary_flatten,
)


# ─────────────────────────────────────────────────────────────────────────
# Tee helper — duplicates stdout to a per-run text file
# ─────────────────────────────────────────────────────────────────────────

class _DualTee:
    def __init__(self, stream, capture: StringIO):
        self.stream = stream
        self.capture = capture

    def write(self, s):
        self.stream.write(s)
        self.capture.write(s)

    def flush(self):
        self.stream.flush()


# ─────────────────────────────────────────────────────────────────────────
# Flatten helpers per level
# ─────────────────────────────────────────────────────────────────────────

def _flatten(
    level: str,
    X_ts: np.ndarray, mask: np.ndarray,
    var_names: Sequence[str], n_windows: int, n_intervals: int, seed: int,
):
    if level == "L1":
        X = summary_flatten(X_ts, mask)
        names = summary_feature_names(var_names)
        return X, names
    if level == "L2":
        X = multi_window_flatten(X_ts, mask, n_windows=n_windows)
        names = multi_window_feature_names(
            var_names, n_windows=n_windows, T=X_ts.shape[1],
        )
        return X, names
    if level == "L3":
        extractor = IntervalFeatureExtractor(
            var_names=var_names, T=X_ts.shape[1],
            n_intervals=n_intervals, seed=seed,
        )
        X = extractor.transform(X_ts, mask)
        names = [
            f"{m.stat}__{m.variable_name}__[{m.interval_start}:{m.interval_end}]"
            for m in extractor.feature_meta
        ]
        return X, names
    raise ValueError(f"unsupported level {level!r}")


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run static PPθ-Post 17-variant comparison on temporal "
                    "benchmarks via L1 / L2 / L3 flattening",
    )
    p.add_argument("--datasets", nargs="+", default=["pam"],
                   help="Temporal datasets registered in temporal.datasets.")
    p.add_argument("--levels", nargs="+", default=["L3"],
                   choices=["L1", "L2", "L3"],
                   help="Flattening level(s) to use.")
    p.add_argument("--n-windows", type=int, default=4,
                   help="Number of windows for L2 (ignored otherwise).")
    p.add_argument("--n-intervals", type=int, default=12,
                   help="Number of intervals for L3 (ignored otherwise).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default=os.path.join(THIS_DIR, "..", "output", "temporal"),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    summaries: List[str] = []
    for ds_name in args.datasets:
        X_ts, mask, y, var_names, dataset_name = load_temporal_dataset(ds_name)
        for level in args.levels:
            X, feat_names = _flatten(
                level, X_ts, mask, var_names,
                n_windows=args.n_windows,
                n_intervals=args.n_intervals,
                seed=args.seed,
            )
            cls_names = [f"class_{c}" for c in sorted(set(y.tolist()))]
            run_label = f"{dataset_name}__{level}"

            capture = StringIO()
            old_stdout = sys.stdout
            sys.stdout = _DualTee(old_stdout, capture)
            try:
                cd.run_comparison(
                    dataset_name=run_label,
                    X=X.astype(np.float32),
                    y=y.astype(np.int64),
                    feature_names=feat_names,
                    class_names=cls_names,
                )
            finally:
                sys.stdout = old_stdout

            log_path = os.path.join(
                args.output_dir,
                f"static_on_temporal_{run_label}_{timestamp}.txt",
            )
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(capture.getvalue())
            summaries.append(log_path)
            print(f"saved → {log_path}")

    print("\nAll runs complete.")
    for p in summaries:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
