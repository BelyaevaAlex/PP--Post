#!/usr/bin/env python3
"""Section 45: focused interpretable PPtheta-Post v3 jobs.

This sweep follows the dual-residual result: do not add another posterior head.
Instead, test stronger fully/interpretable evidence substrates and stricter
validation gates around an EBM/GA2M-style base risk.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402

FAMILY_VARIANTS = (
    "pp_theta_post_rule_family_calibrated,"
    "pp_theta_post_tabpfn_ebm_family_calibrated,"
    "pp_theta_post_monotone_ebm_families"
)
UTILITY_VARIANTS = (
    "pp_theta_post_family_utility_pruned_topk,"
    "pp_theta_post_ebm_bounded_residual_gate,"
    "pp_theta_post_agreement_gated"
)
BAYES_VARIANTS = (
    "pp_theta_post_bayes_llr_beta,"
    "pp_theta_post_bayes_llr_posneg,"
    "pp_theta_post_bayes_llr_posneg_mcc,"
    "pp_theta_post_bayes_llr_posneg_sens92"
)
OPERATING_VARIANTS = (
    "pp_theta_post_operating_calibrated,"
    "pp_theta_post_operating_mcc,"
    "pp_theta_post_operating_sens90,"
    "pp_theta_post_operating_sens92,"
    "pp_theta_post_operating_sens95"
)
RESIDUAL_VARIANTS = (
    "pp_theta_post_ebm_residual_calibrated,"
    "pp_theta_post_ebm_residual_mcc,"
    "pp_theta_post_ebm_residual_sens92,"
    "pp_theta_post_ebm_residual_sens95,"
    "pp_theta_post_ebm_bounded_residual_gate"
)

EXPERIMENTS = {
    # EBM/GA2M terms become posterior evidence objects rather than only a baseline.
    "v3_ebm_evidence_objects": [
        "--rule-sources", "ebm_terms,tabpfn_distill_ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "source_native," + FAMILY_VARIANTS,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    # Fallback to EBM unless validation evidence concentration/agreement supports correction.
    "v3_utility_gated_fallback": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", UTILITY_VARIANTS + ",pp_theta_post_operating_mcc,pp_theta_post_operating_sens92",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    # Fully posterior semantics: family-level LLR, Beta shrinkage, pos/neg evidence split.
    "v3_bayesian_family_llr": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", BAYES_VARIANTS + ",pp_theta_post_rule_family_calibrated",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    # Official clinical operating regimes selected validation-only.
    "v3_operating_points": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", OPERATING_VARIANTS,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    # Tighter residual correction: preserve calibration unless selected families give utility.
    "v3_residual_calibrated_gate": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", RESIDUAL_VARIANTS,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    # Best-effort interpretable combo, intentionally still decomposed into auditable variants.
    "v3_interpretable_combo": [
        "--rule-sources", "tabpfn_distill_ebm_terms,ebm_terms,xgb",
        "--baselines", "ebm,tabpfn",
        "--variants", (
            "source_native,"
            "pp_theta_post_tabpfn_ebm_family_calibrated,"
            "pp_theta_post_family_utility_pruned_topk,"
            "pp_theta_post_monotone_ebm_families,"
            "pp_theta_post_bayes_llr_posneg,"
            "pp_theta_post_ebm_residual_sens95,"
            "pp_theta_post_operating_sens92"
        ),
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
}

EXPERIMENT_ENVS = {
    "v3_ebm_evidence_objects": {
        "PPPOST_FAMILY_UTILITY_TOPK": "32",
        "PPPOST_BAYES_LLR_TOPK": "32",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
    "v3_utility_gated_fallback": {
        "PPPOST_FAMILY_UTILITY_TOPK": "16",
        "PPPOST_EBM_RESIDUAL_MAX_SCALE": "0.45",
        "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL": "0.012",
        "PPPOST_EBM_RESIDUAL_ACC_TOL": "0.006",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
    "v3_bayesian_family_llr": {
        "PPPOST_BAYES_LLR_TOPK": "24",
        "PPPOST_BAYES_LLR_BETA_STRENGTH": "96",
        "PPPOST_BAYES_LLR_CLIP": "1.75",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
    "v3_operating_points": {
        "PPPOST_FAMILY_UTILITY_TOPK": "32",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
    "v3_residual_calibrated_gate": {
        "PPPOST_EBM_RESIDUAL_TOPK": "16",
        "PPPOST_EBM_RESIDUAL_TRUE_WEIGHT": "0.65",
        "PPPOST_EBM_RESIDUAL_RIDGE_L2": "6.0",
        "PPPOST_EBM_RESIDUAL_MAX_SCALE": "0.40",
        "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL": "0.010",
        "PPPOST_EBM_RESIDUAL_ACC_TOL": "0.004",
        "PPPOST_EBM_RESIDUAL_CLIP": "1.25",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
    "v3_interpretable_combo": {
        "PPPOST_FAMILY_UTILITY_TOPK": "24",
        "PPPOST_BAYES_LLR_TOPK": "24",
        "PPPOST_BAYES_LLR_BETA_STRENGTH": "72",
        "PPPOST_BAYES_LLR_CLIP": "2.0",
        "PPPOST_EBM_RESIDUAL_TOPK": "24",
        "PPPOST_EBM_RESIDUAL_TRUE_WEIGHT": "0.60",
        "PPPOST_EBM_RESIDUAL_RIDGE_L2": "4.0",
        "PPPOST_EBM_RESIDUAL_MAX_SCALE": "0.55",
        "PPPOST_EBM_RESIDUAL_LOGLOSS_TOL": "0.018",
        "PPPOST_EBM_RESIDUAL_ACC_TOL": "0.008",
        "PPPOST_SENS_SPEC_FLOOR": "0.92",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = build_arg_parser()
    known, passthrough = parser.parse_known_args(argv)
    for key, value in EXPERIMENT_ENVS.get(known.experiment, {}).items():
        os.environ.setdefault(key, value)
    default_out = ROOT / "output" / "paper" / "45_interpretable_v3" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section45] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
