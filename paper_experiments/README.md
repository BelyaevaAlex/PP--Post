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
| Temporal benchmark | `section_08_temporal_benchmark.py` | `temporal/compare_temporal.py` | L1-L4 plus L2T/L3T temporal feature-teacher comparison and optional vendored SOTA baselines |
| Temporal ablations | `section_09_temporal_ablations.py` | `temporal/ablations.py` | fixed L4 aggregation/head sweeps plus L2T/L3T feature-teacher ablations |
| Case studies | `section_10_case_studies.py` | `temporal/case_studies.py` | per-sample top-rule explanations for paper examples |

TabPFN v3 weights are gated.  After accepting the
`Prior-Labs/tabpfn_3` terms, run
`python download_tabpfn_ts_weights.py --kind ts` for the temporal L2T/L3T
rows and `python download_tabpfn_ts_weights.py --kind classifier` for
tabular TabPFN baseline/distillation rows.

Vendored temporal baselines are tracked as Git submodules under
`temporal/vendor/*`.  Use `git submodule update --init --recursive` after
cloning.  The parent repository ignores untracked cache files inside those
submodules, so local `__pycache__` directories do not look like missing
vendor code.
