#!/usr/bin/env python3
"""Paper Section 07: the interpretability story.

Answers the central question PPθ-Post is built around: **how much
accuracy do you trade for interpretability?**

The script:

1. Runs ``compare_datasets`` with a fixed sweep covering every
   interpretability tier — fully interpretable rule sources, distill-
   guided rule sources, the ``pl_ens_distill`` interpretable ensemble,
   the ``pl_ens_tabpfn`` black-box ensemble, and the four standalone
   Track-B competitors (EBM, FIGS, RuleFit, TabPFN).
2. Post-processes the resulting CSV by tagging each row with an
   interpretability tier:

   ===========================  ==================================================
   Tier                         What it means
   ===========================  ==================================================
   ``full_interpretable``       Every prediction is a sum of tree-branch
                                contributions; the student trees were learned
                                from real labels.  No black-box at training or
                                inference.
   ``distill_guided``           Inference is still a sum of branch
                                contributions, but the branches were grown
                                under a TabPFN teacher.  Local-explainable;
                                global-choice-of-structure opaque.
   ``black_box``                A black-box component (raw TabPFN, or
                                ``pl_ens_tabpfn`` mixing TabPFN in) contributes
                                to the final prediction.
   ===========================  ==================================================

3. Emits ``interpretability_summary.md`` next to the CSV with a per-
   dataset table:

   * Best **interpretable** accuracy + which method achieved it.
   * Best **distill-guided** accuracy.
   * Best **black-box** ceiling.
   * The **interpretability gap** = ceiling − best interpretable.

This is the table that goes into the paper's "what does interpretability
cost?" subsection.

    python paper_experiments/section_07_interpretability_story.py
    python paper_experiments/section_07_interpretability_story.py \\
        --datasets sklearn:digits --folds 5
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import (  # noqa: E402
    STANDALONE_BASELINE_TAG,
    main as run_compare_datasets,
)


OUTPUT_DIR = ROOT / "output" / "paper" / "07_interpretability_story"

DEFAULT_ARGS = [
    "--datasets", "sklearn:wine", "sklearn:breast_cancer",
    "--rule-sources", (
        "extratrees,xgb,catboost,figs,ebm_terms,"
        "tabpfn_distill_xgb,tabpfn_distill_et"
    ),
    "--baselines", "ebm,tabpfn,figs,rulefit",
    "--variants", (
        "source_native,pp_theta_post_frozen,pl_wmean,pl_full,pp_theta_post_warm,"
        "pp_theta_post_aux,pp_theta_post_learn_evidence,"
        "calibrated_e2e_noisy_or,pl_ens_distill,pl_ens_tabpfn"
    ),
    "--distill-student", "et",
    "--ensemble-shrinkage", "0.3",
    "--folds", "3",
    "--output-dir", str(OUTPUT_DIR),
]


# --------------------------------------------------------------------------- #
# Interpretability tagging
# --------------------------------------------------------------------------- #

# Standalone Track-B baselines that are themselves rule-based / glass-box.
_STANDALONE_INTERPRETABLE = {"figs", "rulefit", "ebm"}
_STANDALONE_BLACKBOX = {"tabpfn"}

# Variants that mix in a black-box ensemble member.
_BLACKBOX_VARIANTS = {"pl_ens_tabpfn"}
# Variants that mix in a TabPFN-distilled student (still interpretable since
# the student is a tree ensemble, but its structure was teacher-guided).
_DISTILL_VARIANTS = {"pl_ens_distill"}


def classify_interpretability(rule_source: str, variant: str) -> str:
    if rule_source == STANDALONE_BASELINE_TAG:
        if variant in _STANDALONE_BLACKBOX:
            return "black_box"
        if variant in _STANDALONE_INTERPRETABLE:
            return "full_interpretable"
        return "unknown"
    if variant in _BLACKBOX_VARIANTS:
        return "black_box"
    if variant in _DISTILL_VARIANTS:
        return "distill_guided"
    if str(rule_source).startswith("tabpfn_distill_"):
        return "distill_guided"
    return "full_interpretable"


_TIER_LABEL = {
    "full_interpretable": "Full interpretable",
    "distill_guided":     "Distill-guided",
    "black_box":          "Black-box",
}
_TIER_ORDER = ("full_interpretable", "distill_guided", "black_box")


def _latest_csv(directory: Path) -> Path:
    candidates = sorted(directory.glob("compare_datasets_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no compare_datasets_*.csv in {directory}")
    return candidates[-1]


def _per_dataset_best(df: pd.DataFrame) -> Iterable[Dict]:
    for ds_name, ds_df in df.groupby("dataset"):
        # Average accuracy per (method, tier) over folds.
        ds_df = ds_df.copy()
        ds_df["method"] = ds_df["label"]
        agg = (
            ds_df.groupby(["interp_tier", "method"])["accuracy"]
            .mean()
            .reset_index()
        )
        row = {"dataset": ds_name}
        per_tier_best: Dict[str, float] = {}
        per_tier_method: Dict[str, str] = {}
        for tier in _TIER_ORDER:
            sub = agg[agg["interp_tier"] == tier]
            if sub.empty:
                continue
            top = sub.loc[sub["accuracy"].idxmax()]
            per_tier_best[tier] = float(top["accuracy"])
            per_tier_method[tier] = str(top["method"])
        for tier in _TIER_ORDER:
            row[f"{tier}_best"] = per_tier_best.get(tier)
            row[f"{tier}_method"] = per_tier_method.get(tier)
        if "full_interpretable" in per_tier_best and "black_box" in per_tier_best:
            row["interpretability_gap"] = (
                per_tier_best["black_box"] - per_tier_best["full_interpretable"]
            )
        else:
            row["interpretability_gap"] = None
        yield row


def post_process(csv_path: Path, md_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df["interp_tier"] = [
        classify_interpretability(rs, v)
        for rs, v in zip(df["rule_source"].astype(str), df["variant"].astype(str))
    ]
    lines = [
        "# Interpretability story (auto-generated)",
        "",
        f"Source CSV: `{csv_path.name}`",
        "",
        "## Per-dataset accuracy by interpretability tier",
        "",
        "| Dataset | Full interpretable (best, by) | Distill-guided (best, by) | Black-box ceiling (best, by) | Gap (ceiling − interp) |",
        "|---|---|---|---|---|",
    ]
    for row in _per_dataset_best(df):
        def fmt(tier: str) -> str:
            acc = row.get(f"{tier}_best")
            meth = row.get(f"{tier}_method")
            if acc is None:
                return "—"
            return f"{acc:.3f} ({meth})"
        gap = row["interpretability_gap"]
        gap_str = "—" if gap is None else f"{gap:+.3f}"
        lines.append(
            f"| {row['dataset']} | {fmt('full_interpretable')} | "
            f"{fmt('distill_guided')} | {fmt('black_box')} | {gap_str} |"
        )
    lines.append("")
    lines.append("## Row-level breakdown by tier")
    lines.append("")
    for tier in _TIER_ORDER:
        sub = df[df["interp_tier"] == tier]
        if sub.empty:
            continue
        methods = sorted(set(sub["label"].astype(str)))
        lines.append(f"### {_TIER_LABEL[tier]}")
        lines.append("")
        for m in methods:
            lines.append(f"- `{m}`")
        lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"interpretability summary → {md_path}")


def _resolved_output_dir(args: list[str]) -> Path:
    all_args = DEFAULT_ARGS + args
    out = str(OUTPUT_DIR)
    i = 0
    while i < len(all_args):
        item = all_args[i]
        if item == "--output-dir" and i + 1 < len(all_args):
            out = all_args[i + 1]
            i += 2
            continue
        if item.startswith("--output-dir="):
            out = item.split("=", 1)[1]
        i += 1
    return Path(out)


def main(argv: list[str] | None = None) -> int:
    extra_args = list(argv or sys.argv[1:])
    rc = run_compare_datasets(DEFAULT_ARGS + extra_args)
    if rc != 0:
        return rc
    out_dir = _resolved_output_dir(extra_args)
    try:
        csv_path = _latest_csv(out_dir)
    except FileNotFoundError as exc:
        print(f"[warn] {exc}; skipping interpretability post-processing")
        return rc
    post_process(csv_path, out_dir / "interpretability_summary.md")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
