#!/usr/bin/env python3
"""Section 42: EBM residual PPtheta-Post jobs.

EBM gives the calibrated base risk. PPtheta-Post rule families predict only a
bounded residual logit, with TabPFN soft residuals used at train time when a
TabPFN-distilled rule source is available. Families are selected by validation
conditional residual utility, and a validation-tuned gate depends on EBM
uncertainty and posterior evidence concentration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_datasets import main as run_compare_datasets  # noqa: E402


EXPERIMENTS = {
    "ebm_residual_core": [
        "--rule-sources", "xgb,ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_ebm_residual_calibrated,pp_theta_post_ebm_residual_mcc,pp_theta_post_ebm_residual_sens92,pp_theta_post_ebm_bounded_residual_gate",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "ebm_residual_distilled": [
        "--rule-sources", "tabpfn_distill_xgb_soft,tabpfn_distill_ebm_terms,ebm_terms",
        "--baselines", "tabpfn,ebm",
        "--variants", "source_native,pp_theta_post_ebm_residual_calibrated,pp_theta_post_ebm_residual_mcc,pp_theta_post_ebm_residual_sens92,pp_theta_post_ebm_residual_sens95",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "ebm_residual_operating_modes": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_ebm_residual_calibrated,pp_theta_post_ebm_residual_mcc,pp_theta_post_ebm_residual_sens92,pp_theta_post_ebm_residual_sens95,pp_theta_post_operating_mcc,pp_theta_post_operating_sens92",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    parser = build_arg_parser()
    known, passthrough = parser.parse_known_args(argv)
    default_out = ROOT / "output" / "paper" / "42_ebm_residual_ppost" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section42] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
