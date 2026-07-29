#!/usr/bin/env python3
"""Paper Section 33: OpenML/general-tabular generalization wrapper.

Builds a reproducible command plan for binary OpenML tabular checks using the
same compare_datasets.py runner and PPtheta-Post variants. By default the script
is a dry run so it is safe on login nodes; pass --execute on the cluster.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "33_openml_generalization"
DEFAULT_DATASETS = "sklearn:breast_cancer,openml:31,openml:37,openml:44,openml:1461,openml:1480"
DEFAULT_VARIANTS = "source_native,pp_theta_post_evidence_logit_aux,pp_theta_post_evlogit_likelihood,pp_theta_post_evlogit_threshold,pp_theta_post_teacher_anchored"
DEFAULT_RULE_SOURCES = "xgb,extratrees"
DEFAULT_BASELINES = "ebm,tabpfn"


def _split(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "compare_datasets.py"),
        "--datasets",
        *_split(args.datasets),
        "--variants",
        args.variants,
        "--rule-sources",
        args.rule_sources,
        "--baselines",
        args.baselines,
        "--folds",
        str(args.folds),
        "--epochs",
        str(args.epochs),
        "--expensive-epochs",
        str(args.expensive_epochs),
        "--n-estimators",
        str(args.n_estimators),
        "--max-leaf-nodes",
        str(args.max_leaf_nodes),
        "--output-dir",
        str(output_dir),
        "--save-predictions",
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated sklearn/openml dataset specs.")
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--rule-sources", default=DEFAULT_RULE_SOURCES)
    p.add_argument("--baselines", default=DEFAULT_BASELINES)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--expensive-epochs", type=int, default=80)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-leaf-nodes", type=int, default=64)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--execute", action="store_true", help="Run compare_datasets.py instead of only writing the command plan.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, out_dir)

    plan_csv = out_dir / "openml_generalization_plan.csv"
    with plan_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["datasets", "variants", "rule_sources", "baselines", "folds", "command"])
        writer.writeheader()
        writer.writerow(
            {
                "datasets": args.datasets,
                "variants": args.variants,
                "rule_sources": args.rule_sources,
                "baselines": args.baselines,
                "folds": args.folds,
                "command": " ".join(shlex.quote(x) for x in command),
            }
        )

    report = out_dir / "OPENML_GENERALIZATION_PLAN.md"
    report.write_text(
        "\n".join(
            [
                "# OpenML Generalization Plan",
                "",
                f"Plan CSV: `{plan_csv}`",
                "",
                "```bash",
                " ".join(shlex.quote(x) for x in command),
                "```",
                "",
                "Default datasets include one offline sklearn sanity task and several common OpenML binary tasks. OpenML rows require network/cache availability on the cluster.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"plan_csv={plan_csv}")
    if args.execute:
        return subprocess.call(command)
    print("dry_run=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
