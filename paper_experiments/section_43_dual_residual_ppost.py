#!/usr/bin/env python3
"""Section 43: constrained dual-residual PPtheta-Post jobs.

This wave combines the follow-up ideas after EBM residual PPtheta-Post:
separate calibrated risk and clinical residuals, teacher-confidence residual
blending, validation clinical-utility family selection, and calibrated
sensitivity operating points.
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

DUAL_VARIANTS = "pp_theta_post_dual_residual_calibrated,pp_theta_post_dual_residual_mcc,pp_theta_post_dual_residual_sens92,pp_theta_post_dual_residual_sens95_cal"

EXPERIMENTS = {
    "dual_residual_core": [
        "--rule-sources", "xgb,ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", DUAL_VARIANTS + ",pp_theta_post_ebm_residual_calibrated,pp_theta_post_ebm_residual_sens95",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "dual_residual_teacher_conf": [
        "--rule-sources", "tabpfn_distill_xgb_soft,tabpfn_distill_ebm_terms,ebm_terms",
        "--baselines", "tabpfn,ebm",
        "--variants", "source_native," + DUAL_VARIANTS,
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "dual_residual_clinical_utility": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", DUAL_VARIANTS + ",pp_theta_post_operating_sens92",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "dual_residual_stratified_cal": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", DUAL_VARIANTS + ",pp_theta_post_ebm_bounded_residual_gate",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
}

EXPERIMENT_ENVS = {
    "dual_residual_core": {
        "PPPOST_DUAL_CLINICAL_UTILITY": "1",
        "PPPOST_DUAL_RISK_TRUE_WEIGHT": "0.60",
        "PPPOST_DUAL_CLINICAL_TRUE_WEIGHT": "0.85",
        "PPPOST_DUAL_POS_WEIGHT": "2.5",
    },
    "dual_residual_teacher_conf": {
        "PPPOST_DUAL_TEACHER_CONF": "1",
        "PPPOST_DUAL_CLINICAL_UTILITY": "1",
        "PPPOST_DUAL_RISK_TRUE_WEIGHT": "0.45",
        "PPPOST_DUAL_CLINICAL_TRUE_WEIGHT": "0.75",
        "PPPOST_DUAL_POS_WEIGHT": "2.0",
    },
    "dual_residual_clinical_utility": {
        "PPPOST_DUAL_CLINICAL_UTILITY": "1",
        "PPPOST_DUAL_POS_WEIGHT": "3.5",
        "PPPOST_DUAL_UNCERT_WEIGHT": "1.5",
        "PPPOST_DUAL_CLINICAL_MAX_SCALE": "1.0",
    },
    "dual_residual_stratified_cal": {
        "PPPOST_DUAL_CLINICAL_UTILITY": "1",
        "PPPOST_DUAL_LOGLOSS_TOL": "0.035",
        "PPPOST_DUAL_SPEC_TOL": "0.040",
        "PPPOST_DUAL_CLINICAL_MAX_SCALE": "0.70",
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
    default_out = ROOT / "output" / "paper" / "43_dual_residual_ppost" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section43] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
