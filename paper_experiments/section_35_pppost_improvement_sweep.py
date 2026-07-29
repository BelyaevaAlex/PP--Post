#!/usr/bin/env python3
"""Section 35: PPtheta-Post improvement hypotheses.

Each experiment is deliberately narrow so cluster jobs can isolate which idea
helps: better rule sources, contextual branch support, selective/family evidence,
teacher calibration, EBM anchoring, and clinical-metric optimization.
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
    "rule_source_soft": [
        "--rule-sources", "xgb,tabpfn_distill_xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "tabpfn,ebm",
        "--variants", "source_native,neural,pp_theta_post_evidence_logit_aux,pp_theta_post_evlogit_kd,pp_theta_post_evidence_layer_v2",
        "--sparse-logit-top-k", "64",
    ],
    "contextual_support": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft",
        "--baselines", "none",
        "--variants", "pp_theta_post_contextual_support,pp_theta_post_evidence_layer_v2",
        "--sparse-logit-top-k", "64",
    ],
    "selective_aggregation": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft",
        "--baselines", "none",
        "--variants", "pp_theta_post_selective_evidence,pp_theta_post_rule_family,pp_theta_post_sparse_logit",
        "--sparse-logit-top-k", "64",
        "--rule-selection", "diverse",
    ],
    "teacher_calibration": [
        "--rule-sources", "xgb,tabpfn_distill_xgb,tabpfn_distill_xgb_soft",
        "--baselines", "tabpfn",
        "--variants", "pp_theta_post_teacher_anchored,pp_theta_post_teacher_calibrated",
        "--sparse-logit-top-k", "64",
    ],
    "ebm_anchor": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_ebm_anchor,pp_theta_post_evidence_layer_v2",
        "--sparse-logit-top-k", "64",
    ],
    "clinical_objective": [
        "--rule-sources", "xgb,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_clinical_objective,pp_theta_post_selective_evidence,pp_theta_post_teacher_calibrated",
        "--sparse-logit-top-k", "96",
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
    default_out = ROOT / "output" / "paper" / "35_pppost_improvement_sweep" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section35] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
