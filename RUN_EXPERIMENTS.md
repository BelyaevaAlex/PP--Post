# PPtheta-Post Runbook

This note records the local environment status and the experiment map for this
checkout.

## Environment

Project path:

```bash
cd /path/to/PP--Post
```

A project-local experiment environment exists at `.venv`.

```bash
source .venv/bin/activate
python --version
```

The environment uses local project pins for the core stack and a `.pth` link
(`.venv/lib/python3.11/site-packages/pppost-cuda-cmr.pth`) to the existing
`../envs/cmr` CUDA 12.4 package stack for PyTorch/XGBoost. This avoids the
slow 766 MB PyPI torch download while still making GPU PyTorch available from
`.venv/bin/python`.

Verified imports:

| Package | Version / source |
|---|---|
| numpy | 1.26.4, local `.venv` |
| pandas | 2.2.3, local `.venv` |
| scipy | 1.13.1, local `.venv` |
| scikit-learn | 1.5.1, local `.venv` |
| torch | 2.6.0+cu124, from `../envs/cmr` |
| xgboost | 3.2.0, from `../envs/cmr` |
| catboost | 1.2.10, local `.venv` |
| imodels | 2.0.4, local `.venv` |
| interpret | 0.7.3, local `.venv` |
| tabpfn | 8.0.8, local `.venv` |
| tabpfn-time-series | 1.2.0, local `.venv` |
| openml | 0.15.1, local `.venv` |
| beexai | 0.0.6, local `.venv` |

CUDA sanity check passed from `.venv`: `torch.cuda.is_available() == True`,
`torch.version.cuda == "12.4"`, two `NVIDIA A100-SXM4-80GB` devices visible,
and a small CUDA matrix operation completed successfully.

`pip check` is still noisy because `.venv` intentionally sees external
system/cmr packages; reported conflicts are from unrelated packages such as
`mlflow`, `mlspace`, `client-lib`, and `python-language-server`, not from the
PP--Post requirement set used by the experiment drivers.

For TabPFN / TabPFN-TS rows, accept the gated model terms first, then download
weights:

```bash
python download_tabpfn_ts_weights.py --kind classifier
python download_tabpfn_ts_weights.py --kind ts
```

## Verified Smoke Runs

Tabular core smoke, completed successfully:

```bash
.venv/bin/python compare_datasets.py \
  --datasets sklearn:iris \
  --folds 2 \
  --epochs 1 \
  --variants source_native,neural,pl_wmean \
  --n-estimators 3 \
  --max-leaf-nodes 8 \
  --output-dir output/smoke_env \
  --no-roc-auc
```

Outputs:

```text
output/smoke_env/compare_datasets_20260701_122624.csv
output/smoke_env/compare_datasets_20260701_122624.jsonl
```

Temporal L1-L3 smoke, completed successfully:

```bash
.venv/bin/python -m temporal.compare_temporal \
  --datasets p12 \
  --levels L1 L2 L3 \
  --folds 2 \
  --epochs 1 \
  --output-dir output/smoke_env/temporal_l1_l3
```

Output:

```text
output/smoke_env/temporal_l1_l3/compare_temporal_20260701_123012.md
```

Temporal L4 smoke was intentionally interrupted after 2:28 on CPU during the
first L4 RuleNetwork step. The code had not crashed; it was slow because L4
created 1182 branches for the P12 per-timestep representation. The
`PPThetaPostTemporal` class supports `n_estimators` and `max_leaf_nodes`, but
`temporal.compare_temporal` does not currently expose them as CLI flags.

## Main Entrypoints

| Entrypoint | Purpose | Default datasets |
|---|---|---|
| `study_expressivity.py` | theoretical/model-limit probes | synthetic probes internal to the script |
| `compare_datasets.py` | scalable tabular benchmark | `sklearn:iris`, `sklearn:wine`, `sklearn:breast_cancer`, `sklearn:digits` |
| `compare_wine.py` | legacy 9-mode Wine-only comparison | sklearn Wine |
| `temporal.compare_temporal` | temporal L1-L4 benchmark | `p12`, `pam`, `mimic3` synthetic loaders |
| `temporal.ablations` | L4 temporal aggregation/head ablations | `pam` |
| `temporal.case_studies` | top-K rule traces | `p12`, level `L3` |
| `temporal.compare_static_on_temporal` | static PPtheta-Post on temporal L1/L2/L3 flattening | `pam`, level `L3` |
| `temporal.problog_spotcheck` | native ProbLog vs analytical parity check | tiny internal temporal instance |

## Tabular Datasets

`compare_datasets.py` accepts:

| Spec | Meaning |
|---|---|
| `sklearn:iris` / `wine` / `breast_cancer` / `digits` | built-in sklearn datasets |
| `openml:<name_or_id>` | `sklearn.fetch_openml` loader |
| `csv:/path/file.csv:target_col` | CSV with target column |
| `npz:/path/file.npz` | NPZ with arrays `X` and `y` |

Tabular rule sources:

```text
extratrees, xgb, catboost, figs, rulefit,
tabpfn_distill_xgb, tabpfn_distill_et, tabpfn_distill_cb
```

Standalone tabular baselines:

```text
ebm, figs, rulefit, tabpfn
```

Core variants:

```text
source_native, neural, condition_wmean, hybrid_wmean, hybrid_noisy_or,
pl_fast, pl_full, pl_wmean
```

Expensive variants:

```text
theta_learn, pp_theta_post_e2e, pp_theta_post_warm, pp_theta_post_aux,
pp_theta_post_learn_evidence, e2e_noisy_or, calibrated_e2e_noisy_or,
pl_ens_tabpfn, pl_ens_distill
```

## Temporal Datasets

`temporal.datasets.load_temporal_dataset` currently registers synthetic
prototype loaders:

| Key | Dataset name | Task shape |
|---|---|---|
| `p12` | `synthetic_p12` | ICU mortality-like binary task, high missingness |
| `pam` | `synthetic_pam` | wearable activity multiclass task |
| `mimic3` | `synthetic_mimic3_mortality` | MIMIC-III mortality-like binary task |

These are smoke/prototype loaders, not paper-quality real benchmarks. The
README says real PhysioNet/PAMAP2/MIMIC loaders should drop into the same
registry once credentialing and IO are wired.

Temporal levels:

| Level | Meaning |
|---|---|
| `L1` | per-variable summary stats |
| `L2` | multi-window summary stats and deltas |
| `L3` | interval forest features with temporal rule metadata |
| `L4` | per-timestep latent `z(b, X, t)` plus temporal aggregation |

Default L4 variants are the first `--n-l4-variants` entries from
`DEFAULT_TEMPORAL_VARIANTS`:

```text
PL-tMean, PL-tMax, PL-tNoisyOr, PL-tNoisyOr-top10,
PL-tNoisyOr-top25, PL-tForall, PL-kOfT-25, PL-kOfT-50,
PL-tLast, PL-tAttn, PL-tAttnPB, PL-tAttnMH
```

Temporal baselines:

```text
lr, xgb, tabpfn_ts, transformer,
sand, mtan, gru_d, raindrop, interp_gn, seft, camelot
```

The SOTA baselines use vendored submodules under `temporal/vendor/*`.

## Paper Section Commands

Section wrappers live in `paper_experiments/` and forward extra CLI args to
the underlying driver.

| Section | Command | Default datasets / scope |
|---|---|---|
| 01 theoretical limits | `python paper_experiments/section_01_theoretical_limits.py` | internal synthetic probes |
| 02 tabular main | `python paper_experiments/section_02_tabular_main_methods.py` | inherits `compare_datasets.py` defaults: iris, wine, breast_cancer, digits; ExtraTrees; core + main expensive methods |
| 03 tabular ablations | `python paper_experiments/section_03_tabular_ablations.py` | wine, breast_cancer, digits; ExtraTrees; all variants |
| 04 rule-source sweep | `python paper_experiments/section_04_tabular_rule_sources_sweep.py` | wine, breast_cancer; all rule sources + EBM/FIGS/RuleFit/TabPFN baselines; core variants |
| 05 TabPFN distill | `python paper_experiments/section_05_tabular_tabpfn_distill.py` | wine, breast_cancer; ExtraTrees/XGB/CatBoost plus TabPFN-distill sources; TabPFN baseline |
| 06 ensembles | `python paper_experiments/section_06_tabular_ensembles.py` | wine, breast_cancer; ExtraTrees/CatBoost; `pl_ens_distill` vs `pl_ens_tabpfn` |
| 07 interpretability story | `python paper_experiments/section_07_interpretability_story.py` | wine, breast_cancer; interpretable, distill-guided and black-box tiers |
| 08 temporal benchmark | `python paper_experiments/section_08_temporal_benchmark.py` | temporal defaults p12, pam, mimic3; L1-L4; TabPFN-TS distill + TabPFN-TS baseline |
| 09 temporal ablations | `python paper_experiments/section_09_temporal_ablations.py` | `pam`; L4 aggregation/head sweep + TabPFN-TS rows |
| 10 case studies | `python paper_experiments/section_10_case_studies.py` | p12 L3 top-K rule traces |

## Useful Direct Runs

Fast tabular core:

```bash
python compare_datasets.py \
  --datasets sklearn:wine sklearn:breast_cancer \
  --rule-sources extratrees \
  --variants core \
  --folds 3 \
  --epochs 20 \
  --output-dir output/tabular_core
```

Rule-source sweep, requires optional rule-source packages:

```bash
python compare_datasets.py \
  --datasets sklearn:breast_cancer \
  --rule-sources all \
  --baselines ebm,figs,rulefit,tabpfn \
  --variants core \
  --folds 3 \
  --output-dir output/rule_sources
```

Temporal smoke without L4:

```bash
python -m temporal.compare_temporal \
  --datasets p12 pam mimic3 \
  --levels L1 L2 L3 \
  --folds 3 \
  --epochs 20 \
  --output-dir output/temporal_l1_l3
```

Temporal full benchmark:

```bash
python -m temporal.compare_temporal \
  --datasets p12 pam mimic3 \
  --levels L1 L2 L3 L4 \
  --folds 3 \
  --epochs 80 \
  --n-l4-variants 4 \
  --output-dir output/temporal_full
```

Temporal with external baselines:

```bash
git submodule update --init --recursive
python -m temporal.compare_temporal \
  --datasets p12 \
  --baselines all \
  --folds 5 \
  --epochs 80
```



## Real Mortality Benchmarks

The three real ICU databases are processed as **separate** binary hospital
mortality benchmarks.  Do not concatenate or mix them: every preprocessing and
experiment command below names exactly one source dataset.

Task definition:

| Dataset | Raw source | Unit | Target | Window |
|---|---|---|---|---|
| MIMIC-III | `../MIMIC-III` | first ICU stay per `SUBJECT_ID` | `ADMISSIONS.HOSPITAL_EXPIRE_FLAG` | first 48 ICU hours |
| MIMIC-IV | `../mimic-4/physionet.org/files/mimiciv/3.1` | first ICU stay per `subject_id` | `hosp/admissions.hospital_expire_flag` | first 48 ICU hours |
| eICU | `../eICU/physionet.org/files/eicu-crd/2.0` | first unit stay per `uniquepid` | `apachePatientResult.actualhospitalmortality` | first 48 ICU hours |

Features are a shared vitals/labs panel:

```text
heart_rate, systolic_bp, diastolic_bp, mean_bp, respiratory_rate,
temperature, spo2, glucose, creatinine, bun, sodium, potassium,
chloride, bicarbonate, hematocrit, hemoglobin, platelet, wbc,
lactate, bilirubin_total
```

The preprocessing has two cache layers. The final model-ready caches are:

```text
data/processed/mortality/<dataset>_mortality_48h_temporal.npz
data/processed/mortality/<dataset>_mortality_48h_tabular.npz
```

Before those are built, the raw event tables are filtered once into Parquet
event-caches:

```text
data/processed/mortality_event_cache/<dataset>/v1_48h_full/<stream>.parquet
```

For smoke runs with `--max-samples`, the cache scope is separate, for example
`v2_48h_sample100_seed42`, so sample/debug caches cannot accidentally replace
full-dataset caches. Event-cache version `v2` applies conservative physiologic
range filters before hourly aggregation, converts eICU Fahrenheit-like
temperatures to Celsius, and maps eICU `platelets x 1000` / `WBC x 1000`.
Existing `temporal`, `tabular` and `meta` files are reused automatically; add
`--force` only when you deliberately want to rebuild the final NPZ. Add
`--rebuild-event-cache` when the raw tables should be rescanned and the
filtered Parquet event-cache overwritten.

Aggregation semantics: raw observations are binned into the first 48 ICU hours.
For each `(stay, hour, variable)` cell, multiple valid measurements are averaged.
Missing cells remain `NaN` with `mask=0`. The tabular cache is then computed from
the 48-hour tensor using per-variable summary statistics: mean, std, min, max,
first, last, slope, count, and fraction observed.

Build the caches one dataset at a time:

```bash
PYTHONUNBUFFERED=1 python -m temporal.mortality_preprocess \
  --datasets mimic3 \
  --output-dir data/processed/mortality \
  --event-cache-dir data/processed/mortality_event_cache

PYTHONUNBUFFERED=1 python -m temporal.mortality_preprocess \
  --datasets mimic4 \
  --output-dir data/processed/mortality \
  --event-cache-dir data/processed/mortality_event_cache

PYTHONUNBUFFERED=1 python -m temporal.mortality_preprocess \
  --datasets eicu \
  --output-dir data/processed/mortality \
  --event-cache-dir data/processed/mortality_event_cache
```

Smoke caches were validated with `--max-samples 100` for all three sources in
`data/processed/mortality_smoke/`; each produced `[100, 48, 20]` temporal
tensors and a 180-column L1 tabular matrix.

Run temporal models separately:

```bash
python -m temporal.compare_temporal \
  --datasets mimic3_mortality \
  --levels L1 L2 L3 L4 \
  --folds 3 --epochs 80 \
  --output-dir output/mortality/mimic3_temporal

python -m temporal.compare_temporal \
  --datasets mimic4_mortality \
  --levels L1 L2 L3 L4 \
  --folds 3 --epochs 80 \
  --output-dir output/mortality/mimic4_temporal

python -m temporal.compare_temporal \
  --datasets eicu_mortality \
  --levels L1 L2 L3 L4 \
  --folds 3 --epochs 80 \
  --output-dir output/mortality/eicu_temporal
```

Run all tabular variants separately on the tabular caches:

```bash
python compare_datasets.py \
  --datasets npz:data/processed/mortality/mimic3_mortality_48h_tabular.npz \
  --variants all \
  --rule-sources all \
  --baselines all \
  --folds 3 --epochs 80 --expensive-epochs 80 \
  --output-dir output/mortality/mimic3_tabular

python compare_datasets.py \
  --datasets npz:data/processed/mortality/mimic4_mortality_48h_tabular.npz \
  --variants all \
  --rule-sources all \
  --baselines all \
  --folds 3 --epochs 80 --expensive-epochs 80 \
  --output-dir output/mortality/mimic4_tabular

python compare_datasets.py \
  --datasets npz:data/processed/mortality/eicu_mortality_48h_tabular.npz \
  --variants all \
  --rule-sources all \
  --baselines all \
  --folds 3 --epochs 80 --expensive-epochs 80 \
  --output-dir output/mortality/eicu_tabular
```

For a first runtime pass on the full caches, use `--variants core` and
`--rule-sources extratrees`; then expand to `all` once preprocessing and GPU
runtime look healthy.

### One-command separated mortality batch

The full mortality batch can be launched with one script. It still keeps every
source database independent: preprocess, temporal run, and tabular run are
completed for one dataset before moving to the next one.

```bash
./scripts/run_mortality_all.sh
```

Default behavior:

```text
DATASETS="mimic3 mimic4 eicu"
LEVELS="L1 L2 L3 L4"
TEMPORAL_BASELINES="all"
TABULAR_VARIANTS="all"
RULE_SOURCES="all"
TABULAR_BASELINES="all"
FOLDS=3
EPOCHS=80
EXPENSIVE_EPOCHS=80
PREPROCESS_OUTPUT=data/processed/mortality
EVENT_CACHE_DIR=data/processed/mortality_event_cache
OUTPUT_ROOT=output/mortality
LOG_ROOT=logs/mortality
FORCE_PREPROCESS=0
DISABLE_EVENT_CACHE=0
REBUILD_EVENT_CACHE=0
```

Useful smoke/debug overrides:

```bash
env DATASETS=eicu   PREPROCESS_OUTPUT=data/processed/mortality_script_smoke   OUTPUT_ROOT=output/smoke_mortality_script   LOG_ROOT=logs/smoke_mortality_script   PREPROCESS_MAX_SAMPLES=40   FOLDS=2 EPOCHS=1 EXPENSIVE_EPOCHS=1   LEVELS=L1 TEMPORAL_BASELINES=none   TABULAR_VARIANTS=source_native,neural   RULE_SOURCES=extratrees TABULAR_BASELINES=none   TABULAR_EXTRA_ARGS="--n-estimators 3 --max-leaf-nodes 8 --no-roc-auc"   ./scripts/run_mortality_all.sh
```

Set `INCLUDE_TABPFN_TS_DISTILL=1` to append the explicit TabPFN-TS
distillation rows to the temporal run. Set `FORCE_PREPROCESS=1` to rebuild final
NPZ files. Set `REBUILD_EVENT_CACHE=1` only when the raw event tables should be
rescanned and the Parquet event-cache overwritten. The script exports
`MORTALITY_PROCESSED_DIR=$PREPROCESS_OUTPUT` and
`MORTALITY_EVENT_CACHE_DIR=$EVENT_CACHE_DIR`, so loaders and preprocessing use
consistent cache directories.

### Mortality Paper Jobs

The paper pipeline for the mortality task is packaged as cluster-style jobs under
`scripts/jobs/`, following the env-driven `nscbm-sat/scripts/jobs` convention.
Each job runs one dataset and one paper stage; datasets are never mixed.

Create or refresh small stratified smoke caches from the full v2 mortality
caches:

```bash
python scripts/make_mortality_smoke_cache.py \
  --datasets mimic3 mimic4 eicu \
  --source-dir data/processed/mortality \
  --output-dir data/processed/mortality_job_smoke \
  --n 120 --seed 42
```

Run local smoke through the same job runner used on workers:

```bash
for ds in mimic3 mimic4 eicu; do
  DATASET=$ds MODE=smoke STAGE=smoke_all VER=local_smoke \
    SMOKE_N=120 SMOKE_FOLDS=2 SMOKE_EPOCHS=1 \
    scripts/jobs/run_mortality_paper_job.sh
done
```

The local smoke completed successfully for all three datasets on 2026-07-01:

```text
mimic3: temporal L1 + tabular core mini-run, rc=0
mimic4: temporal L1 + tabular core mini-run, rc=0
eicu:   temporal L1 + tabular core mini-run, rc=0
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

Submit stable full jobs to MLSpace, excluding TabPFN-heavy optional stages, ensemble variants, and TabPFN-distilled rule sources:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode full --ver mortality_v1 \
  --stages tabular_main tabular_ablations tabular_rule_sources \
           temporal_main temporal_ablations case_studies \
  --env FOLDS=3 --env EPOCHS=80 --env EXPENSIVE_EPOCHS=80
```

Submit smoke jobs to MLSpace:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode smoke --ver smoke_v1 \
  --stages smoke_all --env SMOKE_N=120
```

Submit TabPFN-dependent tabular rows separately after confirming local/gated weights:

```bash
python scripts/jobs/submit_mortality_jobs.py --mode full --ver mortality_tabpfn_v1 \
  --stages tabular_ablations tabular_rule_sources tabular_tabpfn_distill tabular_ensembles \
  --env TABPFN_STAGES=1 \
  --env FOLDS=3 --env EPOCHS=80 --env EXPENSIVE_EPOCHS=80
```

Main job knobs:

```text
DATASET=mimic3|mimic4|eicu
STAGE=theoretical_limits|smoke_all|tabular_main|tabular_ablations|tabular_rule_sources|tabular_tabpfn_distill|tabular_ensembles|interpretability_story|temporal_main|temporal_ablations|case_studies
MODE=smoke|full|full_tabpfn
TS_TEACHER_BACKEND=extratrees|tabpfn_ts|auto
INCLUDE_TABPFN_TS_DISTILL=0|1
TEMPORAL_BASELINES=none|all
TABPFN_STAGES=0|1
TABULAR_ABLATION_VARIANTS=core,...|all
TABULAR_RULE_SOURCES=extratrees,xgb,catboost,figs,rulefit|all
PAPER_PRESET=stable|full_tabpfn
```

By default the stable full jobs avoid TabPFN-dependent paths: tabular ablations
exclude `pl_ens_tabpfn` / `pl_ens_distill`, rule-source sweeps exclude
TabPFN-distilled sources, and temporal jobs run L1-L4 / ablations with
`TS_TEACHER_BACKEND=extratrees` and no black-box temporal baselines. Set
`TABPFN_STAGES=1`, `INCLUDE_TABPFN_TS_DISTILL=1`, `TEMPORAL_BASELINES=all`,
and `TS_TEACHER_BACKEND=tabpfn_ts` only when the TabPFN/TabPFN-TS dependencies
and checkpoints are available.
