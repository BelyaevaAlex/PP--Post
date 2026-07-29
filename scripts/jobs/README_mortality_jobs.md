# Mortality Paper Jobs

These jobs run the PP--Post mortality benchmark one dataset and one paper stage
at a time. Datasets are never mixed.

Smoke locally:

```bash
for ds in mimic3 mimic4 eicu; do
  DATASET=$ds MODE=smoke STAGE=smoke_all VER=v1 \
    scripts/jobs/run_mortality_paper_job.sh
done
```

Submit smoke jobs to MLSpace:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode smoke --stages smoke_all --ver v1
```



Before submitting `full_tabpfn`, make sure the gated TabPFN checkpoints are in a
shared repo-local directory visible to workers:

```bash
mkdir -p data/tabpfn_checkpoints
HF_TOKEN=... .venv/bin/python download_tabpfn_ts_weights.py --kind classifier \
  --output-dir data/tabpfn_checkpoints
HF_TOKEN=... .venv/bin/python download_tabpfn_ts_weights.py --kind ts \
  --output-dir data/tabpfn_checkpoints
.venv/bin/python download_tabpfn_ts_weights.py --kind classifier \
  --output-dir data/tabpfn_checkpoints --local-files-only
.venv/bin/python download_tabpfn_ts_weights.py --kind ts \
  --output-dir data/tabpfn_checkpoints --local-files-only
```

`run_mortality_paper_job.sh` auto-exports `TABPFN_CLASSIFIER_MODEL_PATH` and
`TABPFN_TS_MODEL_PATH` from `data/tabpfn_checkpoints/` when those files exist.

Submit the complete TabPFN-heavy paper matrix to MLSpace. This is the main
full run: section 01 once, sections 02-10 for each of `mimic3`, `mimic4`, and
`eicu`, with TabPFN / TabPFN-distill / TabPFN-TS rows enabled and temporal
baselines set to `all`.

```bash
python scripts/jobs/submit_mortality_jobs.py --mode full_tabpfn --ver mortality_full_tabpfn_v1 \
  --env FOLDS=3 --env EPOCHS=80 --env EXPENSIVE_EPOCHS=80
```

The submitter creates one `global/theoretical_limits` job plus 27 dataset-stage
jobs. Results are written under:

```text
output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v1/<dataset>/<stage>/
```

Use `--dry-run` to print the exact job matrix without submitting:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode full_tabpfn --ver check --dry-run
```

Submit stable full paper jobs, excluding TabPFN-heavy stages, ensemble variants, and TabPFN-distilled rule sources:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode full --ver v1 \
  --stages tabular_main tabular_ablations tabular_rule_sources \
           temporal_main temporal_ablations case_studies \
  --env FOLDS=3 --env EPOCHS=80 --env EXPENSIVE_EPOCHS=80
```

Submit optional TabPFN-dependent tabular rows separately:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode full --ver tabpfn_v1 \
  --stages tabular_ablations tabular_rule_sources tabular_tabpfn_distill tabular_ensembles \
  --env TABPFN_STAGES=1 --env FOLDS=3 --env EPOCHS=80 --env EXPENSIVE_EPOCHS=80
```

Useful environment knobs:

```text
DATASET=mimic3|mimic4|eicu
STAGE=theoretical_limits|smoke_all|tabular_main|tabular_ablations|tabular_rule_sources|tabular_tabpfn_distill|tabular_ensembles|interpretability_story|temporal_main|temporal_ablations|case_studies|all|all_with_tabpfn|paper_full_tabpfn
MODE=smoke|full|full_tabpfn
SMOKE_N=120
FOLDS=3
EPOCHS=80
EXPENSIVE_EPOCHS=80
TS_TEACHER_BACKEND=extratrees|tabpfn_ts|auto
INCLUDE_TABPFN_TS_DISTILL=0|1
TEMPORAL_BASELINES=none|all|"gru transformer ..."
PAPER_PRESET=stable|full_tabpfn
TABPFN_STAGES=0|1
TABULAR_ABLATION_VARIANTS=core,...|all
TABULAR_RULE_SOURCES=extratrees,xgb,catboost,figs,rulefit|all
```
