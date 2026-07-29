# Paper Experiment Entry Points

This folder is a thin map from future paper sections to runnable scripts.
The implementation stays in the main modules; these wrappers make the
experimental structure explicit and keep section-level defaults in one place.

| Paper section | Wrapper | Core implementation | Purpose |
|---|---|---|---|
| Model limits / theoretical analysis | `section_01_theoretical_limits.py` | `study_expressivity.py` | tau-consistency, XOR/noisy-or limitation, depth likelihood, branch-independence diagnostics |
| Tabular main methods | `section_02_tabular_main_methods.py` | `compare_datasets.py` | core variants plus PPtheta-Post, warm-start, auxiliary branch loss, learned evidence reliability, e2e-NoisyOr, and calibrated e2e-NoisyOr |
| Tabular ablations | `section_03_tabular_ablations.py` | `compare_datasets.py` | full tabular ablation grid (`--variants all`) with each new posterior/noisy-or pipeline as its own row |
| Rule-source sweep | `section_04_tabular_rule_sources_sweep.py` | `compare_datasets.py` | ExtraTrees/XGBoost/CatBoost/FIGS/RuleFit branch sources across core variants |
| TabPFN distillation | `section_05_tabular_tabpfn_distill.py` | `compare_datasets.py` | teacher-distilled rule sources and tabular distillation comparisons |
| Tabular ensembles | `section_06_tabular_ensembles.py` | `compare_datasets.py` | interpretable and black-box ensemble variants |
| Interpretability story | `section_07_interpretability_story.py` | `compare_datasets.py` | interpretability tiers including the new warm/aux/learn-evidence/calibrated ablations |
| Temporal benchmark | `section_08_temporal_benchmark.py` | `temporal/compare_temporal.py` | L1-L4 plus TabPFN-TS teacher -> tree-student distillation and optional vendored SOTA baselines |
| Temporal ablations | `section_09_temporal_ablations.py` | `temporal/ablations.py` | fixed L4 aggregation/head sweeps plus TabPFN-TS distill students and black-box baseline |
| Case studies | `section_10_case_studies.py` | `temporal/case_studies.py` | per-sample top-rule explanations for paper examples |
| PPtheta rule resources | `section_11_pppost_teacher_rule_sources.py` | `compare_datasets.py` | XGBoost and TabPFN-to-XGBoost rule resources for PPtheta-Post |
| Short rule budgets | `section_12_pppost_short_rule_budget.py` | `compare_datasets.py` | short/subpath rules, support filtering, diversity, and 256/512/1024 rule budgets |
| Theta shrinkage | `section_13_pppost_theta_shrinkage.py` | `compare_datasets.py` | empirical-Bayes theta stabilization with multiple pseudo-count strengths |
| Signed evidence | `section_14_pppost_signed_logit_aggregation.py` | `compare_datasets.py` | signed support/opposition aggregation in logit space |
| Sparse evidence | `section_15_pppost_sparse_logit_aggregation.py` | `compare_datasets.py` | top-k posterior evidence as a correlation-aware aggregation ablation |
| Support prior | `section_16_pppost_support_prior.py` | `compare_datasets.py` | frozen empirical branch-support priors for fully interpretable posterior updates |
| Feature reliability | `section_17_pppost_feature_reliability.py` | `compare_datasets.py` | per-feature condition reliability weights for signed evidence |
| Posterior likelihood tuning | `section_18_pppost_posterior_likelihood_tuning.py` | `compare_datasets.py` | tau and p_high/p_low sweeps for posterior evidence calibration |
| Source-calibrated delta | `section_19_pppost_source_calibrated_delta.py` | `compare_datasets.py` | append only source-calibrated PPtheta-Post rows to existing architecture CSVs |
| Fundamental delta variants | `section_20_pppost_fundamental_delta.py` | `compare_datasets.py` | append only new fundamental PPtheta-Post variants to existing architecture CSVs |
| Deep delta variants | `section_21_pppost_deep_delta.py` | `compare_datasets.py` | append only deeper evidence-logit variants to existing architecture CSVs |
| Evidence Layer v2 delta | `section_22_pppost_evidence_layer_v2_delta.py` | `compare_datasets.py` | append only Evidence Layer v2 rows to existing architecture CSVs |
| Teacher-anchored delta | `section_23_pppost_teacher_anchored_delta.py` | `compare_datasets.py` | append only teacher-anchored PPtheta-Post rows to existing architecture CSVs |
| Teacher-anchored missing-only delta | `section_24_pppost_teacher_anchored_missing_delta.py` | `compare_datasets.py` | fill missing teacher-anchored grid rows without rerunning completed rows |
| Audit validation | `section_25_audit_validation.py` | case-study JSON post-processing | top-rule trace summaries, exported-case balance, and missing-field checklist for coverage/sufficiency/deletion/stability |
| Clinical metrics | `section_26_clinical_metrics.py` | `compare_datasets.py` | AUROC/AUPRC/Brier/ECE/sensitivity/specificity/net-benefit rows plus prediction artifacts |
| Uncertainty and non-inferiority | `section_27_uncertainty_noninferiority.py` | CSV post-processing | method CIs and paired bootstrap non-inferiority checks over matched dataset/fold keys |
| Patient-level prediction artifacts | `section_28_prediction_artifact_metrics.py` | prediction `.npz` post-processing | patient-bootstrap CIs and calibration-bin summaries from saved probabilities |
| Reviewer-defense bundle | `section_29_reviewer_defense_report.py` | sections 25-33 outputs | markdown checklist mapping reviewer concerns to available evidence artifacts |
| Posterior parity and complexity | `section_30_posterior_parity_complexity.py` | `problog_inference.py` plus optional ProbLog | vectorized-vs-ProbLog parity checks and timing curves |
| Audit faithfulness | `section_31_audit_faithfulness.py` | case-study/audit JSON post-processing | coverage, sufficiency, deletion, and stability metrics when exported fields are present |
| Explanation baselines | `section_32_explanation_baselines.py` | protocol plus optional baseline CSVs | SHAP/TreeSHAP/feature/rule/surrogate comparison protocol for audit faithfulness |
| OpenML generalization | `section_33_openml_generalization.py` | `compare_datasets.py` | external binary tabular command plan using the same PPtheta-Post runner |
| Clinical task feasibility | `section_34_clinical_task_feasibility.py` | raw stay/admission tables plus existing mortality NPZ caches | audit and optionally write ICU mortality / LOS task caches using the same 48h feature window |

TabPFN v3 weights are gated.  After accepting the
`Prior-Labs/tabpfn_3` terms, run
`python download_tabpfn_ts_weights.py --kind ts` for temporal TabPFN-TS
distillation / baseline rows.  Run
`python download_tabpfn_ts_weights.py --kind classifier` for TabPFN
classifier-head or tabular baseline/distillation rows.

Vendored temporal baselines are tracked as Git submodules under
`temporal/vendor/*`.  Use `git submodule update --init --recursive` after
cloning.  The parent repository ignores untracked cache files inside those
submodules, so local `__pycache__` directories do not look like missing
vendor code.

## Reviewer-defense workflow

After the main mortality runs are complete, use these sections to close the most likely reviewer concerns:

```bash
python paper_experiments/section_25_audit_validation.py \
  --case-root output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v3

python paper_experiments/section_26_clinical_metrics.py \
  --datasets <mortality_dataset_spec> --folds 3

python paper_experiments/section_27_uncertainty_noninferiority.py \
  --csv 'output/paper/26_clinical_metrics/compare_datasets_*.csv' \
  --method-contains 'PPtheta-Post' --comparator-contains 'TabPFN'

python paper_experiments/section_28_prediction_artifact_metrics.py \
  --csv 'output/paper/26_clinical_metrics/compare_datasets_*.csv'

python paper_experiments/section_30_posterior_parity_complexity.py

python paper_experiments/section_31_audit_faithfulness.py \
  --case-root output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v3

python paper_experiments/section_32_explanation_baselines.py

python paper_experiments/section_33_openml_generalization.py

python paper_experiments/section_34_clinical_task_feasibility.py

# To materialize relabeled tabular caches after reviewing prevalence/alignment:
python paper_experiments/section_34_clinical_task_feasibility.py \
  --tasks icu_mortality,icu_los_gt_7d_at48,hospital_los_gt_7d_at48 \
  --write --modalities tabular

python paper_experiments/section_29_reviewer_defense_report.py \
  --clinical-csv output/paper/26_clinical_metrics/<compare_csv>.csv
```

The clinical-metrics section enables `--save-predictions`, which writes per-fold `prediction_artifacts/*.npz`. These files are intentionally separate from the summary CSV so patient-level bootstrap and calibration analyses can be rerun without retraining.

Sections 30-34 close the remaining reviewer-risk items: implementation parity, quantitative audit faithfulness, post-hoc explanation baseline protocol, external general-tabular robustness, and new clinical-task feasibility. The OpenML wrapper is dry-run by default; pass `--execute` on the cluster after confirming network/cache access.
