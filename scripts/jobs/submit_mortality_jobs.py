#!/usr/bin/env python3
"""Submit PP--Post mortality paper jobs to MLSpace (SR004).

Usage examples:
  python scripts/jobs/submit_mortality_jobs.py --mode smoke --stages smoke_all
  python scripts/jobs/submit_mortality_jobs.py --mode full \
    --stages tabular_main tabular_ablations tabular_rule_sources temporal_main temporal_ablations case_studies
  python scripts/jobs/submit_mortality_jobs.py --mode full_tabpfn --ver mortality_full_tabpfn_v1
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR_CANDIDATES = (
    os.environ.get("PPPOST_LOCAL_DIR", ""),
    str(REPO_ROOT),
)
WORKER_DIR = os.environ.get("PPPOST_WORKER_DIR", str(REPO_ROOT))
ENV_FILE = os.environ.get("MLSPACE_JOB_ENV", "")
DEFAULT_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.11-torch2.4.0:0.0.40"
DATASETS = ("mimic3", "mimic4", "eicu")
DEFAULT_FULL_STAGES = (
    "tabular_main",
    "tabular_ablations",
    "tabular_rule_sources",
    "temporal_main",
    "temporal_ablations",
    "case_studies",
)
FULL_TABPFN_STAGES = (
    "tabular_main",                 # section 02
    "tabular_ablations",            # section 03, --variants all via TABPFN_STAGES=1
    "tabular_rule_sources",         # section 04, all rule sources + TabPFN baseline
    "tabular_tabpfn_distill",       # section 05
    "tabular_ensembles",            # section 06
    "interpretability_story",       # section 07
    "temporal_main",                # section 08, TabPFN-TS + all temporal baselines
    "temporal_ablations",           # section 09, TabPFN-TS distill + baseline
    "case_studies",                 # section 10
)
PPPOST_ARCH_STAGES = (
    "pppost_teacher_rule_sources",  # section 11
    "pppost_short_rule_budget",     # section 12
    "pppost_theta_shrinkage",       # section 13
    "pppost_signed_logit",          # section 14
    "pppost_sparse_logit",          # section 15
    "pppost_support_prior",         # section 16
    "pppost_feature_reliability",   # section 17
    "pppost_posterior_likelihood",  # section 18
)
SOURCE_CAL_STAGES = (
    "source_cal_teacher_rule_sources",
    "source_cal_short_rule_budget",
    "source_cal_theta_shrinkage",
    "source_cal_signed_logit",
    "source_cal_sparse_logit",
    "source_cal_support_prior",
    "source_cal_feature_reliability",
    "source_cal_posterior_likelihood",
)
FUNDAMENTAL_STAGES = (
    "fund_teacher_rule_sources",
    "fund_short_rule_budget",
    "fund_theta_shrinkage",
    "fund_signed_logit",
    "fund_sparse_logit",
    "fund_support_prior",
    "fund_feature_reliability",
    "fund_posterior_likelihood",
)
DEEP_STAGES = (
    "deep_teacher_rule_sources",
    "deep_short_rule_budget",
    "deep_theta_shrinkage",
    "deep_signed_logit",
    "deep_sparse_logit",
    "deep_support_prior",
    "deep_feature_reliability",
    "deep_posterior_likelihood",
)
EVIDENCE_V2_STAGES = (
    "ev2_teacher_rule_sources",
    "ev2_short_rule_budget",
    "ev2_theta_shrinkage",
    "ev2_signed_logit",
    "ev2_sparse_logit",
    "ev2_support_prior",
    "ev2_feature_reliability",
    "ev2_posterior_likelihood",
)
TEACHER_ANCHOR_STAGES = (
    "teacher_anchor_teacher_rule_sources",
    "teacher_anchor_short_rule_budget",
    "teacher_anchor_theta_shrinkage",
    "teacher_anchor_signed_logit",
    "teacher_anchor_sparse_logit",
    "teacher_anchor_support_prior",
    "teacher_anchor_feature_reliability",
    "teacher_anchor_posterior_likelihood",
)
TEACHER_ANCHOR_MISSING_STAGES = (
    "teacher_anchor_missing_short_rule_budget",
    "teacher_anchor_missing_theta_shrinkage",
    "teacher_anchor_missing_sparse_logit",
    "teacher_anchor_missing_posterior_likelihood",
)
RAHMATULLAEV_IMPROVEMENT_STAGES = (
    "rahmatullaev_rule_source_soft",
    "rahmatullaev_contextual_support",
    "rahmatullaev_selective_aggregation",
    "rahmatullaev_teacher_calibration",
    "rahmatullaev_ebm_anchor",
    "rahmatullaev_clinical_objective",
)
RAHMATULLAEV_REVIEWER_RESPONSE_STAGES = (
    "rahmatullaev_ebm_correction",
    "rahmatullaev_clinical_operating_modes",
    "rahmatullaev_teacher_anchor_modes",
    "rahmatullaev_rule_family_symbolic",
    "rahmatullaev_audit_semantics",
)
RAHMATULLAEV_INTERPRETABLE_SUBSTRATE_STAGES = (
    "rahmatullaev_ebm_terms_as_evidence",
    "rahmatullaev_ga2m_soft_distill",
    "rahmatullaev_family_theta_calibration",
    "rahmatullaev_monotone_clinical_families",
    "rahmatullaev_redundancy_pruned_topk",
    "rahmatullaev_combo_interpretable_best",
)
RAHMATULLAEV_INTERPRETABLE_SUBSTRATE_V2_STAGES = (
    "rahmatullaev_ebm_bounded_residual_gate",
    "rahmatullaev_agreement_gated_ppost",
    "rahmatullaev_tabpfn_to_ebm_distill",
    "rahmatullaev_family_utility_pruned_topk",
    "rahmatullaev_operating_point_sweep",
    "rahmatullaev_monotone_plus_ebm_families",
)
RAHMATULLAEV_BAYESIAN_LLR_PPOST_STAGES = (
    "rahmatullaev_bayes_llr_core",
    "rahmatullaev_bayes_llr_distilled_substrate",
    "rahmatullaev_bayes_llr_operating_modes",
)
RAHMATULLAEV_EBM_RESIDUAL_PPOST_STAGES = (
    "rahmatullaev_ebm_residual_core",
    "rahmatullaev_ebm_residual_distilled",
    "rahmatullaev_ebm_residual_operating_modes",
)
RAHMATULLAEV_DUAL_RESIDUAL_PPOST_STAGES = (
    "rahmatullaev_dual_residual_core",
    "rahmatullaev_dual_residual_teacher_conf",
    "rahmatullaev_dual_residual_clinical_utility",
    "rahmatullaev_dual_residual_stratified_cal",
)
RAHMATULLAEV_INTERPRETABLE_V3_STAGES = (
    "rahmatullaev_v3_ebm_evidence_objects",
    "rahmatullaev_v3_utility_gated_fallback",
    "rahmatullaev_v3_bayesian_family_llr",
    "rahmatullaev_v3_operating_points",
    "rahmatullaev_v3_residual_calibrated_gate",
    "rahmatullaev_v3_interpretable_combo",
)
RAHMATULLAEV_PPOST_PROOF_STAGES = (
    "rahmatullaev_proof_evidence_ablation",
    "rahmatullaev_proof_selective_utility",
    "rahmatullaev_proof_strong_base_repair",
    "rahmatullaev_proof_audit_sufficiency",
    "rahmatullaev_proof_operating_points",
    "rahmatullaev_proof_randomized_controls",
)
RAHMATULLAEV_ACCEPTANCE_STRENGTHENING_STAGES = (
    "rahmatullaev_accept_trace_sufficiency_curve",
    "rahmatullaev_accept_proof_statistics",
    "rahmatullaev_accept_case_study_trace_candidates",
)
RAHMATULLAEV_ACCEPTANCE_NEXT_STEPS_STAGES = (
    "rahmatullaev_next_compact_residual_trace",
    "rahmatullaev_next_subset_sufficiency",
    "rahmatullaev_next_trace_bootstrap_ci",
    "rahmatullaev_next_ebm_ppost_case_study",
    "rahmatullaev_next_claim_checklist",
    "rahmatullaev_next_ebm_source_diagnostics",
)

RAHMATULLAEV_AAAI_EVIDENCE_V2_STAGES = (
    "rahmatullaev_v2_paired_utility_ci",
    "rahmatullaev_v2_rich_randomized_controls",
    "rahmatullaev_v2_source_compatibility_matrix",
    "rahmatullaev_v2_extended_trace_curve",
    "rahmatullaev_v2_native_wrong_correction",
    "rahmatullaev_v2_operating_point_separation",
    "rahmatullaev_v2_case_trace_candidates",
    "rahmatullaev_v2_external_tabular_sanity",
    "rahmatullaev_v2_component_ablation",
    "rahmatullaev_v2_statistical_summary",
)

RAHMATULLAEV_AAAI_CLAIM_PACKAGE_STAGES = (
    "rahmatullaev_claim_contract",
    "rahmatullaev_claim_source_boundary_map",
    "rahmatullaev_claim_control_gap_audit",
    "rahmatullaev_claim_trace_sufficiency_refresh",
    "rahmatullaev_claim_reviewer_trace_examples",
    "rahmatullaev_claim_package_summary",
)

RAHMATULLAEV_AAAI_FINAL_STRENGTHENING_STAGES = (
    "rahmatullaev_final_slim_usefulness",
    "rahmatullaev_final_replay_integrity",
    "rahmatullaev_final_clinical_trace",
    "rahmatullaev_final_deletion_sufficiency",
    "rahmatullaev_final_failure_modes",
)

RAHMATULLAEV_EICU_STRENGTHENING_STAGES = (
    "rahmatullaev_eicu_rulefit_official",
    "rahmatullaev_eicu_operating_points",
    "rahmatullaev_eicu_measurement_pattern_families",
    "rahmatullaev_eicu_measurement_policy_calibration",
    "rahmatullaev_eicu_family_pruning_sweep",
)

RAHMATULLAEV_AAAI_REVIEWER_STRESS_STAGES = (
    "rahmatullaev_stress_ebm_vs_ppost_audit",
    "rahmatullaev_stress_conditional_utility",
    "rahmatullaev_stress_trace_compression_v2",
    "rahmatullaev_stress_teacher_anchor_calibration",
    "rahmatullaev_stress_measurement_policy_v2",
)

RAHMATULLAEV_AAAI_ACCEPTANCE_CLINICIAN_SYMBOLIC_STAGES = (
    "rahmatullaev_accept_clinician_audit_packet",
    "rahmatullaev_accept_clean_interpretable_calibrated",
    "rahmatullaev_accept_symbolic_family_calibrated",
    "rahmatullaev_accept_supplement_slimming_manifest",
)

RAHMATULLAEV_SYMBOLIC_CLEAN_PPOST_STAGES = (
    "rahmatullaev_symbolic_rulefit_calibrated",
    "rahmatullaev_symbolic_figs_bounded",
    "rahmatullaev_symbolic_auditselect",
    "rahmatullaev_symbolic_family_ppost",
    "rahmatullaev_symbolic_thresholding",
)
THEORY_STAGE = "theoretical_limits"  # section 01, data-independent


def _resolve_local_dir() -> Path:
    for raw in LOCAL_DIR_CANDIDATES:
        if not raw:
            continue
        p = Path(raw)
        if (p / "scripts" / "jobs" / "run_mortality_paper_job.sh").exists():
            return p
    raise FileNotFoundError(
        "Could not find PP--Post local dir; set PPPOST_LOCAL_DIR explicitly"
    )


def _load_env() -> None:
    if not ENV_FILE:
        raise FileNotFoundError("Missing MLSpace env file; set MLSPACE_JOB_ENV")
    p = Path(ENV_FILE).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Missing MLSpace env file: {ENV_FILE}")
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        if v.strip():
            os.environ[k.strip()] = v.strip()


def _state_load(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"sweep": "pppost_mortality_paper", "submitted_at": date.today().isoformat(), "jobs": {}}


def _state_save(state_path: Path, st: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(st, indent=2, sort_keys=True))


def _env_list(items: list[str]) -> dict[str, str]:
    out = {}
    for kv in items:
        k, _, v = kv.partition("=")
        if not k or not _:
            raise ValueError(f"--env must be KEY=VAL, got {kv!r}")
        out[k.strip()] = v.strip()
    return out


def _default_stages(mode: str) -> list[str]:
    if mode == "smoke":
        return ["smoke_all"]
    if mode == "full_tabpfn":
        return list(FULL_TABPFN_STAGES)
    if mode == "pppost_arch":
        return list(PPPOST_ARCH_STAGES)
    if mode == "pppost_source_cal":
        return list(SOURCE_CAL_STAGES)
    if mode == "pppost_fundamental":
        return list(FUNDAMENTAL_STAGES)
    if mode == "pppost_deep":
        return list(DEEP_STAGES)
    if mode == "pppost_evidence_v2":
        return list(EVIDENCE_V2_STAGES)
    if mode == "pppost_teacher_anchor":
        return list(TEACHER_ANCHOR_STAGES)
    if mode == "pppost_teacher_anchor_missing":
        return list(TEACHER_ANCHOR_MISSING_STAGES)
    if mode == "rahmatullaev_improvements":
        return list(RAHMATULLAEV_IMPROVEMENT_STAGES)
    if mode == "rahmatullaev_reviewer_response":
        return list(RAHMATULLAEV_REVIEWER_RESPONSE_STAGES)
    if mode == "rahmatullaev_interpretable_substrate":
        return list(RAHMATULLAEV_INTERPRETABLE_SUBSTRATE_STAGES)
    if mode == "rahmatullaev_interpretable_substrate_v2":
        return list(RAHMATULLAEV_INTERPRETABLE_SUBSTRATE_V2_STAGES)
    if mode == "rahmatullaev_bayesian_llr_ppost":
        return list(RAHMATULLAEV_BAYESIAN_LLR_PPOST_STAGES)
    if mode == "rahmatullaev_ebm_residual_ppost":
        return list(RAHMATULLAEV_EBM_RESIDUAL_PPOST_STAGES)
    if mode == "rahmatullaev_dual_residual_ppost":
        return list(RAHMATULLAEV_DUAL_RESIDUAL_PPOST_STAGES)
    if mode == "rahmatullaev_interpretable_v3":
        return list(RAHMATULLAEV_INTERPRETABLE_V3_STAGES)
    if mode == "rahmatullaev_ppost_proof":
        return list(RAHMATULLAEV_PPOST_PROOF_STAGES)
    if mode == "rahmatullaev_acceptance_strengthening":
        return list(RAHMATULLAEV_ACCEPTANCE_STRENGTHENING_STAGES)
    if mode == "rahmatullaev_acceptance_next_steps":
        return list(RAHMATULLAEV_ACCEPTANCE_NEXT_STEPS_STAGES)
    if mode == "rahmatullaev_aaai_evidence_v2":
        return list(RAHMATULLAEV_AAAI_EVIDENCE_V2_STAGES)
    if mode == "rahmatullaev_aaai_claim_package":
        return list(RAHMATULLAEV_AAAI_CLAIM_PACKAGE_STAGES)
    if mode == "rahmatullaev_aaai_final_strengthening":
        return list(RAHMATULLAEV_AAAI_FINAL_STRENGTHENING_STAGES)
    if mode == "rahmatullaev_eicu_strengthening":
        return list(RAHMATULLAEV_EICU_STRENGTHENING_STAGES)
    if mode == "rahmatullaev_aaai_reviewer_stress":
        return list(RAHMATULLAEV_AAAI_REVIEWER_STRESS_STAGES)
    if mode == "rahmatullaev_aaai_acceptance_clinician_symbolic":
        return list(RAHMATULLAEV_AAAI_ACCEPTANCE_CLINICIAN_SYMBOLIC_STAGES)
    if mode == "rahmatullaev_symbolic_clean_ppost":
        return list(RAHMATULLAEV_SYMBOLIC_CLEAN_PPOST_STAGES)
    return list(DEFAULT_FULL_STAGES)


def _preset_env(mode: str) -> dict[str, str]:
    if mode not in {"full_tabpfn", "pppost_arch", "pppost_source_cal", "pppost_fundamental", "pppost_deep", "pppost_evidence_v2", "pppost_teacher_anchor", "pppost_teacher_anchor_missing", "rahmatullaev_improvements", "rahmatullaev_reviewer_response", "rahmatullaev_interpretable_substrate", "rahmatullaev_interpretable_substrate_v2", "rahmatullaev_bayesian_llr_ppost", "rahmatullaev_ebm_residual_ppost", "rahmatullaev_dual_residual_ppost", "rahmatullaev_interpretable_v3", "rahmatullaev_ppost_proof", "rahmatullaev_acceptance_strengthening", "rahmatullaev_acceptance_next_steps", "rahmatullaev_aaai_evidence_v2", "rahmatullaev_aaai_claim_package", "rahmatullaev_aaai_final_strengthening", "rahmatullaev_eicu_strengthening", "rahmatullaev_aaai_reviewer_stress"}:
        return {}
    return {
        "PAPER_PRESET": "full_tabpfn",
        "TABPFN_STAGES": "1",
        "TABPFN_DEVICE": "cuda",
        "TABPFN_IGNORE_PRETRAINING_LIMITS": "1",
        "INCLUDE_TABPFN_TS_DISTILL": "1",
        "INCLUDE_TABPFN_TS_BASELINE": "1",
        "TS_TEACHER_BACKEND": "tabpfn_ts",
        "TS_TEACHER_DEVICE": "cuda",
        "TEMPORAL_BASELINES": "all",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--stages", nargs="+", default=None)
    ap.add_argument("--mode", choices=("smoke", "full", "full_tabpfn", "pppost_arch", "pppost_source_cal", "pppost_fundamental", "pppost_deep", "pppost_evidence_v2", "pppost_teacher_anchor", "pppost_teacher_anchor_missing", "rahmatullaev_improvements", "rahmatullaev_reviewer_response", "rahmatullaev_interpretable_substrate", "rahmatullaev_interpretable_substrate_v2", "rahmatullaev_bayesian_llr_ppost", "rahmatullaev_ebm_residual_ppost", "rahmatullaev_dual_residual_ppost", "rahmatullaev_interpretable_v3", "rahmatullaev_ppost_proof", "rahmatullaev_acceptance_strengthening", "rahmatullaev_acceptance_next_steps", "rahmatullaev_aaai_evidence_v2", "rahmatullaev_aaai_claim_package", "rahmatullaev_aaai_final_strengthening", "rahmatullaev_eicu_strengthening", "rahmatullaev_aaai_reviewer_stress", "rahmatullaev_aaai_acceptance_clinician_symbolic", "rahmatullaev_symbolic_clean_ppost"), default="full")
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--instance-type", default="", help="Override MLSpace instance type; default is a100.{gpus}gpu")
    ap.add_argument("--queue-name", default="fusionbrainlab-job", help="Optional MLSpace queue_name override")
    ap.add_argument("--allocation-name", default="", help="Optional MLSpace allocation_name override")
    ap.add_argument("--desc", default="PP--Post mortality paper")
    ap.add_argument("--env", action="append", default=[], help="Extra job env KEY=VAL; repeatable")
    ap.add_argument("--force", action="store_true", help="Submit even if axis is already in state")
    ap.add_argument("--no-theory", action="store_true", help="Do not submit section 01 for full_tabpfn mode")
    ap.add_argument("--dry-run", action="store_true", help="Print job axes/env without submitting")
    args = ap.parse_args()

    local_dir = _resolve_local_dir()
    state_path = local_dir / "output" / "mlspace_state" / "pppost_mortality_paper_jobs.json"
    stages = args.stages or _default_stages(args.mode)
    extra_env = _env_list(args.env)
    preset_env = _preset_env(args.mode)
    worker_script = f"{WORKER_DIR}/scripts/jobs/run_mortality_paper_job.sh"

    if not args.dry_run:
        _load_env()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            os.environ.pop(key, None)
        import client_lib  # noqa: E402
        client_lib.environment = client_lib.Environment()
    else:
        client_lib = None

    st = _state_load(state_path)
    submitted = 0

    def submit_one(dataset: str, stage: str) -> None:
        nonlocal submitted
        axis = f"{args.mode}.{args.ver}.{dataset}.{stage}"
        if axis in st["jobs"] and not args.force:
            print(f"[skip] axis {axis} already submitted as {st['jobs'][axis]['job_name']}")
            return
        env_vars = {
            "DATASET": dataset,
            "STAGE": stage,
            "MODE": args.mode,
            "VER": args.ver,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            **preset_env,
            **extra_env,
        }
        if args.dry_run:
            print(f"[dry-run] axis={axis} env={json.dumps(env_vars, sort_keys=True)}")
            return
        job = client_lib.Job(
            script=f"bash {worker_script}",
            base_image=DEFAULT_IMAGE,
            instance_type=args.instance_type or f"a100.{args.gpus}gpu",
            region=client_lib.RegionMT.SR004,
            type=client_lib.Job.Type.binary,
            job_desc=f"{args.desc}: {axis} #rahmatullaev",
            n_workers=1,
            env_variables=env_vars,
            queue_name=args.queue_name or None,
            allocation_name=args.allocation_name or None,
        )
        resp = job.submit()
        print(resp)
        if not job.job_name or str(resp).startswith("Error"):
            print(f"[failed] axis={axis} response={resp}")
            return
        print(f"[submitted] axis={axis} job={job.job_name}")
        st["jobs"][axis] = {
            "job_name": job.job_name,
            "dataset": dataset,
            "stage": stage,
            "mode": args.mode,
            "ver": args.ver,
            "gpus": args.gpus,
            "instance_type": args.instance_type or f"a100.{args.gpus}gpu",
            "queue_name": args.queue_name,
            "allocation_name": args.allocation_name,
            "env": env_vars,
            "submitted_at": date.today().isoformat(),
        }
        submitted += 1

    if args.mode == "full_tabpfn" and not args.no_theory and THEORY_STAGE not in stages:
        submit_one("global", THEORY_STAGE)

    for dataset in args.datasets:
        for stage in stages:
            if stage == THEORY_STAGE:
                continue
            submit_one(dataset, stage)

    if not args.dry_run:
        _state_save(state_path, st)
    print(f"submitted={submitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
