#!/usr/bin/env python3
"""Paper Section 30: posterior parity and complexity checks.

This script verifies the posterior audit layer rather than training new models.
It compares the vectorized analytical posterior with the native ProbLog engine
when ProbLog is installed, and records timing curves for the main posterior
operations on synthetic branches.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from branch_schema import Branch, Condition
from problog_inference import ProbLogClassifier, compute_condition_activation

DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "30_posterior_parity_complexity"


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_branches(n_branches: int, n_features: int, depth: int, n_classes: int, rng: np.random.Generator) -> list[Branch]:
    branches: list[Branch] = []
    for b in range(n_branches):
        conds: list[Condition] = []
        for d in range(depth):
            feat = int((b + d) % n_features)
            threshold = float(rng.normal(loc=0.0, scale=0.8))
            direction = "le" if (b + d) % 2 == 0 else "gt"
            conds.append(Condition(feature_idx=feat, threshold=threshold, direction=direction, node_id=b * 100 + d))
        theta = rng.dirichlet(np.ones(n_classes) + 0.25).tolist()
        branches.append(Branch(branch_id=f"synthetic_b{b}", tree_id=b // 8, parent_node_id=b, conditions=conds, class_proportions=theta))
    return branches


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_parity(args: argparse.Namespace, out_dir: Path) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    branches = make_branches(args.n_branches, args.n_features, args.depth, args.n_classes, rng)
    X = rng.normal(size=(args.n_samples, args.n_features))
    branch_probs = rng.uniform(0.02, 0.98, size=(args.n_samples, args.n_branches))

    clf = ProbLogClassifier(branches, args.n_classes, mode="full", p_high=args.p_high, p_low=args.p_low)
    proba_full = clf.predict_proba(branch_probs, X=X, verbose=False)
    proba_repeat = clf.predict_proba(branch_probs, X=X, verbose=False)
    rows = [
        {
            "comparison": "vectorized_full_vs_repeat",
            "status": "ok",
            "n_samples": args.n_samples,
            "n_branches": args.n_branches,
            "depth": args.depth,
            "max_abs_error": float(np.max(np.abs(proba_full - proba_repeat))),
            "mean_abs_error": float(np.mean(np.abs(proba_full - proba_repeat))),
        }
    ]

    if args.skip_problog:
        rows.append({"comparison": "vectorized_full_vs_native_problog", "status": "skipped_by_flag"})
        return rows

    if importlib.util.find_spec("problog") is None:
        rows.append({"comparison": "vectorized_full_vs_native_problog", "status": "problog_not_installed"})
        return rows

    try:
        n_pg = min(args.problog_samples, args.n_samples)
        b_pg = min(args.problog_branches, args.n_branches)
        pg_branches = branches[:b_pg]
        pg_probs = branch_probs[:n_pg, :b_pg]
        pg_X = X[:n_pg]
        full_small = ProbLogClassifier(pg_branches, args.n_classes, mode="full", p_high=args.p_high, p_low=args.p_low).predict_proba(pg_probs, X=pg_X, verbose=False)
        pg_small = ProbLogClassifier(pg_branches, args.n_classes, mode="full_problog", p_high=args.p_high, p_low=args.p_low).predict_proba(pg_probs, X=pg_X, verbose=False)
        rows.append(
            {
                "comparison": "vectorized_full_vs_native_problog",
                "status": "ok",
                "n_samples": n_pg,
                "n_branches": b_pg,
                "depth": args.depth,
                "max_abs_error": float(np.max(np.abs(full_small - pg_small))),
                "mean_abs_error": float(np.mean(np.abs(full_small - pg_small))),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on optional ProbLog runtime.
        rows.append(
            {
                "comparison": "vectorized_full_vs_native_problog",
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc)[:300],
            }
        )
    return rows


def run_timing(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed + 17)
    rows: list[dict] = []
    for n_samples in _parse_int_list(args.timing_samples):
        for n_branches in _parse_int_list(args.timing_branches):
            branches = make_branches(n_branches, args.n_features, args.depth, args.n_classes, rng)
            X = rng.normal(size=(n_samples, args.n_features))
            branch_probs = rng.uniform(0.02, 0.98, size=(n_samples, n_branches))
            clf = ProbLogClassifier(branches, args.n_classes, mode="full", p_high=args.p_high, p_low=args.p_low)

            t0 = time.perf_counter()
            for _ in range(args.repeats):
                compute_condition_activation(branches, X, tau=args.tau)
            condition_seconds = (time.perf_counter() - t0) / max(args.repeats, 1)

            t0 = time.perf_counter()
            for _ in range(args.repeats):
                clf.predict_proba(branch_probs, X=X, verbose=False)
            posterior_seconds = (time.perf_counter() - t0) / max(args.repeats, 1)

            rows.append(
                {
                    "n_samples": n_samples,
                    "n_branches": n_branches,
                    "depth": args.depth,
                    "n_classes": args.n_classes,
                    "condition_seconds": condition_seconds,
                    "posterior_seconds": posterior_seconds,
                    "condition_us_per_sample_branch": 1e6 * condition_seconds / max(n_samples * n_branches, 1),
                    "posterior_us_per_sample_branch": 1e6 * posterior_seconds / max(n_samples * n_branches, 1),
                }
            )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-branches", type=int, default=6)
    p.add_argument("--problog-samples", type=int, default=3)
    p.add_argument("--problog-branches", type=int, default=4)
    p.add_argument("--n-features", type=int, default=6)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--p-high", type=float, default=0.95)
    p.add_argument("--p-low", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--timing-samples", default="16,64")
    p.add_argument("--timing-branches", default="8,32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-problog", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parity_rows = run_parity(args, out_dir)
    timing_rows = run_timing(args)
    parity_csv = out_dir / "posterior_parity.csv"
    timing_csv = out_dir / "complexity_timing.csv"
    _write_csv(parity_csv, parity_rows)
    _write_csv(timing_csv, timing_rows)

    status = ", ".join(f"{r.get('comparison')}={r.get('status')}" for r in parity_rows)
    report = out_dir / "POSTERIOR_PARITY_COMPLEXITY.md"
    report.write_text(
        "\n".join(
            [
                "# Posterior Parity and Complexity",
                "",
                f"Parity CSV: `{parity_csv}`",
                f"Timing CSV: `{timing_csv}`",
                "",
                f"Parity status: {status}",
                "",
                "Use the native ProbLog row as the formal implementation check when the optional `problog` package is available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"parity_csv={parity_csv}")
    print(f"timing_csv={timing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
