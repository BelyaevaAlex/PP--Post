#!/usr/bin/env python3
"""Section 39: fully interpretable rule-substrate refinement jobs.

These experiments target the next metric step without adding a black-box
inference head: EBM/GA2M-style terms as evidence objects, TabPFN soft
distillation only at train time, family-level calibration, monotone clinical
rule families, redundancy-pruned top-k aggregation, and a combined best-effort
interpretable substrate.
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
    "ebm_terms_as_evidence": [
        "--rule-sources", "ebm_terms",
        "--baselines", "ebm,tabpfn",
        "--variants", "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_ebm_correction_calibrated,pp_theta_post_ebm_correction_mcc",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "ga2m_soft_distill": [
        "--rule-sources", "tabpfn_distill_xgb_soft,ebm_terms",
        "--baselines", "tabpfn,ebm",
        "--variants", "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_evidence_layer_v2",
        "--rule-selection", "diverse",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "family_theta_calibration": [
        "--rule-sources", "xgb,ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_rule_family_calibrated,pp_theta_post_rule_family_sensitivity",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "monotone_clinical_families": [
        "--rule-sources", "clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "source_native,pp_theta_post_rule_family_calibrated,pp_theta_post_rule_family_sensitivity,pp_theta_post_sparse_logit",
        "--sparse-logit-top-k", "64",
        "--save-predictions",
    ],
    "redundancy_pruned_topk": [
        "--rule-sources", "xgb,ebm_terms,clinical_monotone",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_selective_evidence,pp_theta_post_rule_family,pp_theta_post_rule_family_calibrated",
        "--rule-selection", "diverse",
        "--rule-budget", "384",
        "--sparse-logit-top-k", "48",
        "--save-predictions",
    ],
    "combo_interpretable_best": [
        "--rule-sources", "ebm_terms,clinical_monotone,tabpfn_distill_xgb_soft",
        "--baselines", "ebm,tabpfn",
        "--variants", "pp_theta_post_ebm_correction_calibrated,pp_theta_post_ebm_correction_mcc,pp_theta_post_rule_family_calibrated,pp_theta_post_rule_family_sensitivity,pp_theta_post_selective_evidence",
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
    default_out = ROOT / "output" / "paper" / "39_interpretable_substrate" / known.experiment
    args = EXPERIMENTS[known.experiment] + ["--output-dir", str(default_out)] + passthrough
    print(f"[section39] experiment={known.experiment} args={' '.join(args)}")
    return run_compare_datasets(args)


if __name__ == "__main__":
    raise SystemExit(main())
