#!/usr/bin/env python3
"""Section 41: Bayesian log-likelihood-ratio PP-Post jobs.

This wave evaluates an explicitly posterior-style PP-Post inference rule:
family evidence contributes log-likelihood ratios against the class prior,
Beta shrinkage regularizes family reliability, and positive/negative evidence
streams are calibrated separately on the validation fold.
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
    "bayes_llr_core": [
        "--rule-sources", "xgb,ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_bayes_llr,pp_theta_post_bayes_llr_beta,pp_theta_post_bayes_llr_posneg,pp_theta_post_rule_family_calibrated",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "bayes_llr_distilled_substrate": [
        "--rule-sources", "tabpfn_distill_xgb_soft,tabpfn_distill_ebm_terms,ebm_terms",
        "--baselines", "tabpfn,ebm",
        "--variants", "source_native,pp_theta_post_bayes_llr_beta,pp_theta_post_bayes_llr_posneg,pp_theta_post_tabpfn_ebm_family_calibrated",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "bayes_llr_operating_modes": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_bayes_llr_posneg,pp_theta_post_bayes_llr_posneg_mcc,pp_theta_post_bayes_llr_posneg_sens92,pp_theta_post_operating_mcc,pp_theta_post_operating_sens92",
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
    default_out = ROOT / "output" / "paper" / "41_bayesian_llr_ppost" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section41] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
