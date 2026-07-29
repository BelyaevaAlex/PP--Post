#!/usr/bin/env python3
"""Section 37: reviewer-response PPtheta-Post sweeps.

Focused experiments for the main AAAI reviewer risks:
* why not just EBM -> EBM base logit plus auditable PPtheta correction;
* small gains -> explicit calibrated-risk, MCC, and sensitivity modes;
* TeacherAnchor calibration -> compare MCC vs calibrated operating points;
* weak frozen symbolic story -> compact rule-family calibrated/sensitivity modes;
* applied/benchmark-heavy risk -> save prediction artifacts for audit semantics.
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
    "ebm_correction": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_ebm_correction_calibrated,pp_theta_post_ebm_correction_mcc,pp_theta_post_ebm_correction_sensitivity,pp_theta_post_ebm_anchor",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "clinical_operating_modes": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_selective_evidence,pp_theta_post_clinical_objective,pp_theta_post_ebm_correction_mcc,pp_theta_post_ebm_correction_sensitivity",
        "--sparse-logit-top-k", "96",
        "--save-predictions",
    ],
    "teacher_anchor_modes": [
        "--rule-sources", "xgb,tabpfn_distill_xgb,tabpfn_distill_xgb_soft",
        "--baselines", "tabpfn,ebm",
        "--variants", "pp_theta_post_teacher_anchored,pp_theta_post_teacher_calibrated,pp_theta_post_ebm_correction_calibrated,pp_theta_post_ebm_correction_mcc",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "rule_family_symbolic": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_rule_family,pp_theta_post_rule_family_calibrated,pp_theta_post_rule_family_sensitivity,pp_theta_post_frozen,pp_theta_post_frozen_support_prior",
        "--sparse-logit-top-k", "64",
        "--rule-selection", "diverse",
        "--save-predictions",
    ],
    "audit_semantics": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_evidence_layer_v2,pp_theta_post_rule_family_calibrated,pp_theta_post_ebm_correction_calibrated,pp_theta_post_ebm_correction_sensitivity",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    known, passthrough = parser.parse_known_args(list(argv or sys.argv[1:]))
    default_out = ROOT / "output" / "paper" / "37_reviewer_response_sweep" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section37] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
