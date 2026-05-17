# Paper Experiment Entry Points

This folder is a thin map from future paper sections to runnable scripts.
The implementation stays in the main modules; these wrappers make the
experimental structure explicit and keep section-level defaults in one place.

| Paper section | Wrapper | Core implementation | Purpose |
|---|---|---|---|
| Model limits / theoretical analysis | `section_01_theoretical_limits.py` | `study_expressivity.py` | tau-consistency, XOR/noisy-or limitation, depth likelihood, branch-independence diagnostics |
| Large tabular benchmark | `section_02_large_tabular_main_methods.py` | `compare_datasets.py` | large-dataset loaders, chunked prediction, streaming CSV/JSONL, PPtheta-Post and e2e-NoisyOr |
| Temporal benchmark | `section_03_temporal_benchmark.py` | `temporal/compare_temporal.py` | L1-L4 PPtheta-Post temporal comparison plus optional vendored SOTA baselines |
| Temporal ablations | `section_04_temporal_ablations.py` | `temporal/ablations.py` | fixed L4 backbone, temporal aggregation/head sweeps |
| Interpretability case studies | `section_05_case_studies.py` | `temporal/case_studies.py` | per-sample top-rule explanations for paper examples |

Vendored temporal baselines are tracked as Git submodules under
`temporal/vendor/*`.  Use `git submodule update --init --recursive` after
cloning.  The parent repository ignores untracked cache files inside those
submodules, so local `__pycache__` directories do not look like missing
vendor code.

