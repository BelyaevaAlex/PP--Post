#!/usr/bin/env python3
"""Section 40: interpretable substrate v2 reviewer-response jobs.

This wave keeps the same mortality protocol and evaluates targeted fixes for
"why not just EBM/TabPFN?": bounded EBM residuals, agreement gates,
TabPFN-to-EBM distillation, validation-pruned rule families, explicit clinical
operating points, and monotone clinical priors over EBM/rule families.
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
    "ebm_bounded_residual_gate": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_ebm_bounded_residual_gate,pp_theta_post_ebm_correction_calibrated,pp_theta_post_ebm_correction_mcc",
        "--rule-selection", "diverse",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "agreement_gated_ppost": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_agreement_gated,pp_theta_post_rule_family_calibrated,pp_theta_post_ebm_correction_calibrated",
        "--rule-selection", "diverse",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "tabpfn_to_ebm_distill": [
        "--rule-sources", "tabpfn_distill_ebm_terms,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "tabpfn,ebm",
        "--variants", "source_native,pp_theta_post_tabpfn_ebm_family_calibrated,pp_theta_post_rule_family_calibrated",
        "--rule-selection", "diverse",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "family_utility_pruned_topk": [
        "--rule-sources", "xgb,ebm_terms,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_family_utility_pruned_topk,pp_theta_post_rule_family_calibrated,pp_theta_post_selective_evidence",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "48",
        "--save-predictions",
    ],
    "operating_point_sweep": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_operating_calibrated,pp_theta_post_operating_mcc,pp_theta_post_operating_sens90,pp_theta_post_operating_sens92,pp_theta_post_operating_sens95",
        "--rule-selection", "diverse",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "monotone_plus_ebm_families": [
        "--rule-sources", "xgb,ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_monotone_ebm_families,pp_theta_post_rule_family_calibrated,pp_theta_post_rule_family_sensitivity",
        "--rule-selection", "diverse",
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
    default_out = ROOT / "output" / "paper" / "40_interpretable_substrate_v2" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section40] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
