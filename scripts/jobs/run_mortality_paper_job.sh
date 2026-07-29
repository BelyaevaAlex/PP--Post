#!/usr/bin/env bash
# PP--Post mortality paper job runner.
# Mirrors the nscbm-sat scripts/jobs pattern: env-driven, worker-visible paths,
# tee'd logs, and one dataset/stage per job.
set +e
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATASET="${DATASET:-eicu}"
STAGE="${STAGE:-smoke_all}"
MODE="${MODE:-smoke}"
VER="${VER:-v1}"
SEED="${SEED:-42}"
FOLDS="${FOLDS:-3}"
EPOCHS="${EPOCHS:-80}"
EXPENSIVE_EPOCHS="${EXPENSIVE_EPOCHS:-80}"
SMOKE_N="${SMOKE_N:-120}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
N_ESTIMATORS="${N_ESTIMATORS:-}"
MAX_LEAF_NODES="${MAX_LEAF_NODES:-}"
EXPENSIVE_SUBSAMPLE="${EXPENSIVE_SUBSAMPLE:-5000}"
if [ -z "${PAPER_PRESET+x}" ]; then
  PAPER_PRESET=stable
  [ "$MODE" = "full_tabpfn" ] && PAPER_PRESET=full_tabpfn
fi
if [ "$PAPER_PRESET" = "full_tabpfn" ]; then
  TABPFN_STAGES="${TABPFN_STAGES:-1}"
  TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
  TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
  TS_TEACHER_BACKEND="${TS_TEACHER_BACKEND:-tabpfn_ts}"
  TEMPORAL_BASELINES="${TEMPORAL_BASELINES:-all}"
  INCLUDE_TABPFN_TS_DISTILL="${INCLUDE_TABPFN_TS_DISTILL:-1}"
  INCLUDE_TABPFN_TS_BASELINE="${INCLUDE_TABPFN_TS_BASELINE:-1}"
  TS_TEACHER_DEVICE="${TS_TEACHER_DEVICE:-cuda}"
else
  TABPFN_STAGES="${TABPFN_STAGES:-0}"
  TABPFN_DEVICE="${TABPFN_DEVICE:-auto}"
  TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-0}"
  TS_TEACHER_BACKEND="${TS_TEACHER_BACKEND:-extratrees}"
  TEMPORAL_BASELINES="${TEMPORAL_BASELINES:-none}"
  INCLUDE_TABPFN_TS_DISTILL="${INCLUDE_TABPFN_TS_DISTILL:-0}"
  INCLUDE_TABPFN_TS_BASELINE="${INCLUDE_TABPFN_TS_BASELINE:-0}"
  TS_TEACHER_DEVICE="${TS_TEACHER_DEVICE:-cpu}"
fi
TS_TEACHER_MAX_ROWS="${TS_TEACHER_MAX_ROWS:-4096}"
TS_TEACHER_N_ESTIMATORS="${TS_TEACHER_N_ESTIMATORS:-8}"
TS_TEACHER_WORKERS="${TS_TEACHER_WORKERS:-1}"
TABPFN_TS_TEACHER_HEAD="${TABPFN_TS_TEACHER_HEAD:-tabpfn}"
TEMPORAL_RULE_N_ESTIMATORS="${TEMPORAL_RULE_N_ESTIMATORS:-}"
TEMPORAL_RULE_MAX_LEAF_NODES="${TEMPORAL_RULE_MAX_LEAF_NODES:-}"
TEMPORAL_L4_BATCH_SIZE="${TEMPORAL_L4_BATCH_SIZE:-256}"
TEMPORAL_ATTENTION_MAX_SAMPLES="${TEMPORAL_ATTENTION_MAX_SAMPLES:-2048}"
STABLE_TABULAR_ABLATION_VARIANTS="${STABLE_TABULAR_ABLATION_VARIANTS:-core,theta_learn,pp_theta_post_e2e,pp_theta_post_warm,pp_theta_post_aux,pp_theta_post_learn_evidence,e2e_noisy_or,calibrated_e2e_noisy_or}"
STABLE_TABULAR_RULE_SOURCES="${STABLE_TABULAR_RULE_SOURCES:-extratrees,xgb,catboost,figs,rulefit,ebm_terms}"

if [[ ! "$DATASET" =~ ^(mimic3|mimic4|eicu|global)$ ]]; then
  echo "Unknown DATASET=$DATASET; expected mimic3|mimic4|eicu|global" >&2
  exit 2
fi
if [ "$DATASET" = "global" ] && [ "$STAGE" != "theoretical_limits" ]; then
  echo "DATASET=global is only valid for STAGE=theoretical_limits" >&2
  exit 2
fi

cd "$REPO" || { echo "NO REPO at $REPO"; exit 3; }
TABPFN_CKPT_DIR="${TABPFN_CKPT_DIR:-$REPO/data/tabpfn_checkpoints}"
if [ -z "${TABPFN_TS_MODEL_PATH:-}" ] && [ -f "$TABPFN_CKPT_DIR/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt" ]; then
  export TABPFN_TS_MODEL_PATH="$TABPFN_CKPT_DIR/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"
fi
if [ -z "${TABPFN_CLASSIFIER_MODEL_PATH:-}" ] && [ -f "$TABPFN_CKPT_DIR/tabpfn-v3-classifier-v3_default.ckpt" ]; then
  export TABPFN_CLASSIFIER_MODEL_PATH="$TABPFN_CKPT_DIR/tabpfn-v3-classifier-v3_default.ckpt"
fi
PY="${PY_OVERRIDE:-$REPO/.venv/bin/python}"
"$PY" -c "import torch, pandas, sklearn" >/dev/null 2>&1 || { [ -n "${CMR_ENV_PY:-}" ] && PY="$CMR_ENV_PY"; }
"$PY" -c "import torch, pandas, sklearn" >/dev/null 2>&1 || PY="python3"
VENV_SITE="$REPO/.venv/lib/python3.11/site-packages"

FULL_CACHE_DIR="${FULL_CACHE_DIR:-$REPO/data/processed/mortality}"
SMOKE_CACHE_DIR="${SMOKE_CACHE_DIR:-$REPO/data/processed/mortality_job_smoke}"
if [ "$MODE" = "smoke" ]; then
  CACHE_DIR="$SMOKE_CACHE_DIR"
  FOLDS="${SMOKE_FOLDS:-2}"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  EXPENSIVE_EPOCHS="${SMOKE_EXPENSIVE_EPOCHS:-1}"
  N_ESTIMATORS="${N_ESTIMATORS:-3}"
  MAX_LEAF_NODES="${MAX_LEAF_NODES:-8}"
  EXPENSIVE_SUBSAMPLE="${SMOKE_EXPENSIVE_SUBSAMPLE:-200}"
else
  CACHE_DIR="$FULL_CACHE_DIR"
fi

OUT_ROOT="${OUT_ROOT:-$REPO/output/mortality_paper_jobs/${MODE}_${VER}}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/mortality_paper_jobs/${MODE}_${VER}}"
OUT="$OUT_ROOT/${DATASET}/${STAGE}"
LOG="$LOG_ROOT/${DATASET}_${STAGE}.out"
mkdir -p "$OUT" "$LOG_ROOT" "$CACHE_DIR"
exec > >(tee "$LOG") 2>&1

export PYTHONUNBUFFERED=1
if [ -d "$VENV_SITE" ]; then
  export PYTHONPATH="$REPO:$VENV_SITE:${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
fi
export MORTALITY_PROCESSED_DIR="$CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TABPFN_DEVICE
export TABPFN_IGNORE_PRETRAINING_LIMITS

echo "USING_PY=$PY"
echo "REPO=$REPO DATASET=$DATASET STAGE=$STAGE MODE=$MODE VER=$VER PRESET=$PAPER_PRESET"
echo "CACHE_DIR=$CACHE_DIR OUT=$OUT FOLDS=$FOLDS EPOCHS=$EPOCHS EXPENSIVE_EPOCHS=$EXPENSIVE_EPOCHS"
echo "TABPFN_STAGES=$TABPFN_STAGES TABPFN_DEVICE=$TABPFN_DEVICE TABPFN_IGNORE_PRETRAINING_LIMITS=$TABPFN_IGNORE_PRETRAINING_LIMITS TS_TEACHER_BACKEND=$TS_TEACHER_BACKEND TEMPORAL_BASELINES=$TEMPORAL_BASELINES"
echo "TEMPORAL_RULE_N_ESTIMATORS=${TEMPORAL_RULE_N_ESTIMATORS:-auto} TEMPORAL_RULE_MAX_LEAF_NODES=${TEMPORAL_RULE_MAX_LEAF_NODES:-auto} TEMPORAL_L4_BATCH_SIZE=$TEMPORAL_L4_BATCH_SIZE TEMPORAL_ATTENTION_MAX_SAMPLES=$TEMPORAL_ATTENTION_MAX_SAMPLES"
echo "TABPFN_CLASSIFIER_MODEL_PATH=${TABPFN_CLASSIFIER_MODEL_PATH:-missing} TABPFN_TS_MODEL_PATH=${TABPFN_TS_MODEL_PATH:-missing}"
"$PY" - <<'PYCHECK'
import sys, torch, numpy, pandas, sklearn
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "pandas", pandas.__version__, "sklearn", sklearn.__version__)
PYCHECK
if [ "$PAPER_PRESET" = "full_tabpfn" ]; then
  "$PY" - <<'PYCHECK' || exit $?
import importlib
import os
from pathlib import Path

for key in ("TABPFN_CLASSIFIER_MODEL_PATH", "TABPFN_TS_MODEL_PATH"):
    value = os.environ.get(key)
    if not value or not Path(value).exists():
        raise SystemExit(f"missing required {key}: {value or '<unset>'}")

for module in ("tabpfn", "tabpfn_time_series"):
    mod = importlib.import_module(module)
    print("tabpfn_check", module, getattr(mod, "__version__", "no_version"))
print("tabpfn_check ok")
PYCHECK
fi

if [ "$MODE" = "smoke" ] && [ "$DATASET" != "global" ]; then
  "$PY" scripts/make_mortality_smoke_cache.py \
    --datasets "$DATASET" --source-dir "$FULL_CACHE_DIR" --output-dir "$CACHE_DIR" \
    --n "$SMOKE_N" --seed "$SEED" || exit $?
fi

TABULAR_NPZ="$CACHE_DIR/${DATASET}_mortality_48h_tabular.npz"
TEMPORAL_DS="${DATASET}_mortality"
APPEND_ARGS=()
if [ -n "${APPEND_RESULTS_TO:-}" ]; then
  APPEND_ARGS+=(--append-results-to "$APPEND_RESULTS_TO")
  [ -n "${APPEND_JSONL_TO:-}" ] && APPEND_ARGS+=(--append-jsonl-to "$APPEND_JSONL_TO")
elif [ -n "${APPEND_STAMP:-}" ]; then
  APPEND_ARGS+=(--append-results-to "$OUT/compare_datasets_${APPEND_STAMP}.csv")
fi
COMMON_TABULAR=(--datasets "npz:$TABULAR_NPZ" --folds "$FOLDS" --epochs "$EPOCHS" --expensive-epochs "$EXPENSIVE_EPOCHS" --batch-size "$BATCH_SIZE" --train-batch-size "$TRAIN_BATCH_SIZE" --expensive-subsample "$EXPENSIVE_SUBSAMPLE" --output-dir "$OUT" "${APPEND_ARGS[@]}")
[ -n "$N_ESTIMATORS" ] && COMMON_TABULAR+=(--n-estimators "$N_ESTIMATORS")
[ -n "$MAX_LEAF_NODES" ] && COMMON_TABULAR+=(--max-leaf-nodes "$MAX_LEAF_NODES")
[ -n "${REFINEMENT_MAX_SAMPLES:-}" ] && COMMON_TABULAR+=(--refinement-max-samples "$REFINEMENT_MAX_SAMPLES")

add_tabpfn_ts_args() {
  local -n _args=$1
  _args+=(--ts-teacher-device "$TS_TEACHER_DEVICE")
  _args+=(--ts-teacher-max-rows "$TS_TEACHER_MAX_ROWS")
  _args+=(--ts-teacher-n-estimators "$TS_TEACHER_N_ESTIMATORS")
  _args+=(--ts-teacher-workers "$TS_TEACHER_WORKERS")
  _args+=(--tabpfn-ts-teacher-head "$TABPFN_TS_TEACHER_HEAD")
  [ -n "${TABPFN_TS_MODEL_PATH:-}" ] && _args+=(--ts-teacher-model-path "$TABPFN_TS_MODEL_PATH")
  [ -n "${TABPFN_CLASSIFIER_MODEL_PATH:-}" ] && _args+=(--tabpfn-classifier-model-path "$TABPFN_CLASSIFIER_MODEL_PATH")
}

run_theoretical_limits() {
  "$PY" paper_experiments/section_01_theoretical_limits.py \
    --output "$OUT/section_01_theoretical_limits.json"
}

run_smoke_all() {
  echo "=== smoke temporal main ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m temporal.compare_temporal \
    --datasets "$TEMPORAL_DS" --levels L1 --folds "$FOLDS" --epochs "$EPOCHS" \
    --output-dir "$OUT/temporal_main" || return $?
  echo "=== smoke tabular core ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" compare_datasets.py \
    --datasets "npz:$TABULAR_NPZ" --variants source_native,neural \
    --rule-sources extratrees --baselines none --folds "$FOLDS" \
    --epochs "$EPOCHS" --expensive-epochs "$EXPENSIVE_EPOCHS" \
    --n-estimators "${N_ESTIMATORS:-3}" --max-leaf-nodes "${MAX_LEAF_NODES:-8}" \
    --output-dir "$OUT/tabular_core" --no-roc-auc || return $?
}

run_tabular_main() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${TABULAR_MAIN_RULE_SOURCES:-}" ] && args+=(--rule-sources "$TABULAR_MAIN_RULE_SOURCES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_02_tabular_main_methods.py \
    "${args[@]}"
}

run_tabular_ablations() {
  local variants="${TABULAR_ABLATION_VARIANTS:-$STABLE_TABULAR_ABLATION_VARIANTS}"
  local rule_sources="${TABULAR_ABLATION_RULE_SOURCES:-extratrees,ebm_terms}"
  [ "$TABPFN_STAGES" = "1" ] && variants="${TABULAR_ABLATION_VARIANTS:-all}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_03_tabular_ablations.py \
    "${COMMON_TABULAR[@]}" --rule-sources "$rule_sources" --variants "$variants"
}

run_tabular_rule_sources() {
  local rule_sources="${TABULAR_RULE_SOURCES:-$STABLE_TABULAR_RULE_SOURCES}"
  local baselines="${TABULAR_BASELINES:-ebm,figs,rulefit}"
  [ "$TABPFN_STAGES" = "1" ] && rule_sources="${TABULAR_RULE_SOURCES:-all}"
  [ "$TABPFN_STAGES" = "1" ] && baselines="${TABULAR_BASELINES:-ebm,figs,rulefit,tabpfn}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_04_tabular_rule_sources_sweep.py \
    "${COMMON_TABULAR[@]}" --rule-sources "$rule_sources" --baselines "$baselines" --variants core
}

run_tabular_tabpfn_distill() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${TABPFN_DISTILL_RULE_SOURCES:-}" ] && args+=(--rule-sources "$TABPFN_DISTILL_RULE_SOURCES")
  [ -n "${TABPFN_DISTILL_VARIANTS:-}" ] && args+=(--variants "$TABPFN_DISTILL_VARIANTS")
  [ -n "${TABPFN_DISTILL_BASELINES:-}" ] && args+=(--baselines "$TABPFN_DISTILL_BASELINES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_05_tabular_tabpfn_distill.py \
    "${args[@]}"
}

run_tabular_ensembles() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${ENSEMBLES_RULE_SOURCES:-}" ] && args+=(--rule-sources "$ENSEMBLES_RULE_SOURCES")
  [ -n "${ENSEMBLES_VARIANTS:-}" ] && args+=(--variants "$ENSEMBLES_VARIANTS")
  [ -n "${ENSEMBLES_BASELINES:-}" ] && args+=(--baselines "$ENSEMBLES_BASELINES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_06_tabular_ensembles.py \
    "${args[@]}"
}

run_interpretability_story() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${INTERPRETABILITY_RULE_SOURCES:-}" ] && args+=(--rule-sources "$INTERPRETABILITY_RULE_SOURCES")
  [ -n "${INTERPRETABILITY_VARIANTS:-}" ] && args+=(--variants "$INTERPRETABILITY_VARIANTS")
  [ -n "${INTERPRETABILITY_BASELINES:-}" ] && args+=(--baselines "$INTERPRETABILITY_BASELINES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_07_interpretability_story.py \
    "${args[@]}"
}

run_pppost_teacher_rule_sources() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_TEACHER_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_TEACHER_RULE_SOURCES")
  [ -n "${PPPOST_TEACHER_VARIANTS:-}" ] && args+=(--variants "$PPPOST_TEACHER_VARIANTS")
  [ -n "${PPPOST_TEACHER_BASELINES:-}" ] && args+=(--baselines "$PPPOST_TEACHER_BASELINES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_11_pppost_teacher_rule_sources.py \
    "${args[@]}"
}

run_pppost_short_rule_budget() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_RULE_BUDGET_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_RULE_BUDGET_RULE_SOURCES")
  [ -n "${PPPOST_RULE_BUDGET_VARIANTS:-}" ] && args+=(--variants "$PPPOST_RULE_BUDGET_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_12_pppost_short_rule_budget.py \
    "${args[@]}"
}

run_pppost_theta_shrinkage() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_THETA_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_THETA_RULE_SOURCES")
  [ -n "${PPPOST_THETA_VARIANTS:-}" ] && args+=(--variants "$PPPOST_THETA_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_13_pppost_theta_shrinkage.py \
    "${args[@]}"
}

run_pppost_signed_logit() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_SIGNED_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_SIGNED_RULE_SOURCES")
  [ -n "${PPPOST_SIGNED_VARIANTS:-}" ] && args+=(--variants "$PPPOST_SIGNED_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_14_pppost_signed_logit_aggregation.py \
    "${args[@]}"
}

run_pppost_sparse_logit() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_SPARSE_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_SPARSE_RULE_SOURCES")
  [ -n "${PPPOST_SPARSE_VARIANTS:-}" ] && args+=(--variants "$PPPOST_SPARSE_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_15_pppost_sparse_logit_aggregation.py \
    "${args[@]}"
}

run_pppost_support_prior() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_SUPPORT_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_SUPPORT_RULE_SOURCES")
  [ -n "${PPPOST_SUPPORT_VARIANTS:-}" ] && args+=(--variants "$PPPOST_SUPPORT_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_16_pppost_support_prior.py \
    "${args[@]}"
}

run_pppost_feature_reliability() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_FEATREL_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_FEATREL_RULE_SOURCES")
  [ -n "${PPPOST_FEATREL_VARIANTS:-}" ] && args+=(--variants "$PPPOST_FEATREL_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_17_pppost_feature_reliability.py \
    "${args[@]}"
}

run_pppost_posterior_likelihood() {
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${PPPOST_POSTERIOR_RULE_SOURCES:-}" ] && args+=(--rule-sources "$PPPOST_POSTERIOR_RULE_SOURCES")
  [ -n "${PPPOST_POSTERIOR_VARIANTS:-}" ] && args+=(--variants "$PPPOST_POSTERIOR_VARIANTS")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_18_pppost_posterior_likelihood_tuning.py \
    "${args[@]}"
}

run_pppost_arch_all() {
  run_pppost_teacher_rule_sources && run_pppost_short_rule_budget && \
  run_pppost_theta_shrinkage && run_pppost_signed_logit && \
  run_pppost_sparse_logit && run_pppost_support_prior && \
  run_pppost_feature_reliability && run_pppost_posterior_likelihood
}

run_pppost_source_calibrated_delta() {
  local target_stage="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_19_pppost_source_calibrated_delta.py \
    "${args[@]}" --target-stage "$target_stage" --target-dataset "$DATASET" \
    --append-root "${SOURCE_CAL_APPEND_ROOT:-$REPO/output/mortality_paper_jobs/pppost_arch_mortality_pppost_arch_v1}"
}


run_pppost_fundamental_delta() {
  local target_stage="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_20_pppost_fundamental_delta.py \
    "${args[@]}" --target-stage "$target_stage" --target-dataset "$DATASET" \
    --append-root "${PPPOST_FUNDAMENTAL_APPEND_ROOT:-$REPO/output/mortality_paper_jobs/pppost_arch_mortality_pppost_arch_v1}"
}


run_pppost_deep_delta() {
  local target_stage="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_21_pppost_deep_delta.py \
    "${args[@]}" --target-stage "$target_stage" --target-dataset "$DATASET" \
    --append-root "${PPPOST_DEEP_APPEND_ROOT:-$REPO/output/mortality_paper_jobs/pppost_arch_mortality_pppost_arch_v1}"
}

run_pppost_evidence_v2_delta() {
  local target_stage="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_22_pppost_evidence_layer_v2_delta.py \
    "${args[@]}" --target-stage "$target_stage" --target-dataset "$DATASET" \
    --append-root "${PPPOST_EVIDENCE_V2_APPEND_ROOT:-$REPO/output/mortality_paper_jobs/pppost_arch_mortality_pppost_arch_v1}"
}

run_pppost_teacher_anchor_delta() {
  local target_stage="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_23_pppost_teacher_anchored_delta.py \
    "${args[@]}" --target-stage "$target_stage" --target-dataset "$DATASET" \
    --append-root "${PPPOST_TEACHER_ANCHOR_APPEND_ROOT:-$REPO/output/mortality_paper_jobs/pppost_arch_mortality_pppost_arch_v1}"
}

run_pppost_teacher_anchor_missing_delta() {
  local target_stage="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_24_pppost_teacher_anchored_missing_delta.py \
    "${args[@]}" --target-stage "$target_stage" --target-dataset "$DATASET" \
    --append-root "${PPPOST_TEACHER_ANCHOR_APPEND_ROOT:-$REPO/output/mortality_paper_jobs/pppost_arch_mortality_pppost_arch_v1}"
}


run_rahmatullaev_improvement() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_35_pppost_improvement_sweep.py     --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_reviewer_response() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_37_reviewer_response_sweep.py \
    --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_interpretable_substrate() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_39_interpretable_substrate.py \
    --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_interpretable_substrate_v2() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_40_interpretable_substrate_v2.py     --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_bayesian_llr_ppost() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_41_bayesian_llr_ppost.py     --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_ebm_residual_ppost() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_42_ebm_residual_ppost.py     --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_dual_residual_ppost() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_43_dual_residual_ppost.py     --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_interpretable_v3() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_45_interpretable_v3.py     --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_ppost_proof() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_46_ppost_proof.py     --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_acceptance_strengthening() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_48_acceptance_strengthening.py \
    --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_acceptance_next_steps() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_50_aaai_next_steps.py     --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_aaai_evidence_v2() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_51_aaai_evidence_v2.py \
    --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_aaai_claim_package() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_53_aaai_claim_package.py \
    --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_aaai_final_strengthening() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_55_aaai_final_strengthening.py \
    --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_eicu_strengthening() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_57_eicu_strengthening.py \
    --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_aaai_reviewer_stress() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_59_aaai_reviewer_stress.py \
    --experiment "$experiment" "${args[@]}"
}

run_rahmatullaev_aaai_acceptance_clinician_symbolic() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_60_acceptance_clinician_symbolic.py \
    --experiment "$experiment" "${args[@]}"
}


run_rahmatullaev_symbolic_clean_ppost() {
  local experiment="$1"
  local args=("${COMMON_TABULAR[@]}")
  [ -n "${RAHMATULLAEV_RULE_BUDGET:-}" ] && args+=(--rule-budget "$RAHMATULLAEV_RULE_BUDGET")
  [ -n "${RAHMATULLAEV_RULE_MAX_DEPTH:-}" ] && args+=(--rule-max-depth "$RAHMATULLAEV_RULE_MAX_DEPTH")
  [ -n "${RAHMATULLAEV_RULE_MIN_SUPPORT:-}" ] && args+=(--rule-min-support "$RAHMATULLAEV_RULE_MIN_SUPPORT")
  [ -n "${RAHMATULLAEV_REFINEMENT_MAX_SAMPLES:-}" ] && args+=(--refinement-max-samples "$RAHMATULLAEV_REFINEMENT_MAX_SAMPLES")
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" paper_experiments/section_66_symbolic_clean_ppost.py \
    --experiment "$experiment" "${args[@]}"
}

run_temporal_main() {
  local args=(--datasets "$TEMPORAL_DS" --levels L1 L2 L3 L4 --folds "$FOLDS" --epochs "$EPOCHS" --output-dir "$OUT" --ts-teacher-backend "$TS_TEACHER_BACKEND")
  add_tabpfn_ts_args args
  [ -n "$TEMPORAL_RULE_N_ESTIMATORS" ] && args+=(--rule-n-estimators "$TEMPORAL_RULE_N_ESTIMATORS")
  [ -n "$TEMPORAL_RULE_MAX_LEAF_NODES" ] && args+=(--rule-max-leaf-nodes "$TEMPORAL_RULE_MAX_LEAF_NODES")
  args+=(--l4-batch-size "$TEMPORAL_L4_BATCH_SIZE")
  [ "$INCLUDE_TABPFN_TS_DISTILL" = "1" ] && args+=(--include-tabpfn-ts-distill)
  if [ "$TEMPORAL_BASELINES" != "none" ]; then
    # shellcheck disable=SC2206
    local bl=( $TEMPORAL_BASELINES )
    args+=(--baselines "${bl[@]}")
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m temporal.compare_temporal "${args[@]}"
}

run_temporal_ablations() {
  local args=(--datasets "$TEMPORAL_DS" --folds "$FOLDS" --epochs "$EPOCHS" --output-dir "$OUT" --ts-teacher-backend "$TS_TEACHER_BACKEND" --l4-batch-size "$TEMPORAL_L4_BATCH_SIZE" --attention-max-samples "$TEMPORAL_ATTENTION_MAX_SAMPLES")
  add_tabpfn_ts_args args
  [ "$INCLUDE_TABPFN_TS_DISTILL" = "1" ] && args+=(--include-tabpfn-ts-distill)
  [ "$INCLUDE_TABPFN_TS_BASELINE" = "1" ] && args+=(--include-tabpfn-ts-baseline)
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m temporal.ablations "${args[@]}"
}

run_case_studies() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -m temporal.case_studies \
    --dataset "$TEMPORAL_DS" --level "${CASE_LEVEL:-L4}" --n-samples "${CASE_N:-3}" \
    --top-k "${CASE_TOP_K:-5}" --epochs "$EPOCHS" --output-dir "$OUT"
}

case "$STAGE" in
  theoretical_limits) run_theoretical_limits ;;
  smoke_all) run_smoke_all ;;
  tabular_main) run_tabular_main ;;
  tabular_ablations) run_tabular_ablations ;;
  tabular_rule_sources) run_tabular_rule_sources ;;
  tabular_tabpfn_distill) run_tabular_tabpfn_distill ;;
  tabular_ensembles) run_tabular_ensembles ;;
  interpretability_story) run_interpretability_story ;;
  pppost_teacher_rule_sources) run_pppost_teacher_rule_sources ;;
  pppost_short_rule_budget) run_pppost_short_rule_budget ;;
  pppost_theta_shrinkage) run_pppost_theta_shrinkage ;;
  pppost_signed_logit) run_pppost_signed_logit ;;
  pppost_sparse_logit) run_pppost_sparse_logit ;;
  pppost_support_prior) run_pppost_support_prior ;;
  pppost_feature_reliability) run_pppost_feature_reliability ;;
  pppost_posterior_likelihood) run_pppost_posterior_likelihood ;;
  pppost_arch_all) run_pppost_arch_all ;;
  source_cal_teacher_rule_sources) run_pppost_source_calibrated_delta pppost_teacher_rule_sources ;;
  source_cal_short_rule_budget) run_pppost_source_calibrated_delta pppost_short_rule_budget ;;
  source_cal_theta_shrinkage) run_pppost_source_calibrated_delta pppost_theta_shrinkage ;;
  source_cal_signed_logit) run_pppost_source_calibrated_delta pppost_signed_logit ;;
  source_cal_sparse_logit) run_pppost_source_calibrated_delta pppost_sparse_logit ;;
  source_cal_support_prior) run_pppost_source_calibrated_delta pppost_support_prior ;;
  source_cal_feature_reliability) run_pppost_source_calibrated_delta pppost_feature_reliability ;;
  source_cal_posterior_likelihood) run_pppost_source_calibrated_delta pppost_posterior_likelihood ;;
  fund_teacher_rule_sources) run_pppost_fundamental_delta pppost_teacher_rule_sources ;;
  fund_short_rule_budget) run_pppost_fundamental_delta pppost_short_rule_budget ;;
  fund_theta_shrinkage) run_pppost_fundamental_delta pppost_theta_shrinkage ;;
  fund_signed_logit) run_pppost_fundamental_delta pppost_signed_logit ;;
  fund_sparse_logit) run_pppost_fundamental_delta pppost_sparse_logit ;;
  fund_support_prior) run_pppost_fundamental_delta pppost_support_prior ;;
  fund_feature_reliability) run_pppost_fundamental_delta pppost_feature_reliability ;;
  fund_posterior_likelihood) run_pppost_fundamental_delta pppost_posterior_likelihood ;;
  deep_teacher_rule_sources) run_pppost_deep_delta pppost_teacher_rule_sources ;;
  deep_short_rule_budget) run_pppost_deep_delta pppost_short_rule_budget ;;
  deep_theta_shrinkage) run_pppost_deep_delta pppost_theta_shrinkage ;;
  deep_signed_logit) run_pppost_deep_delta pppost_signed_logit ;;
  deep_sparse_logit) run_pppost_deep_delta pppost_sparse_logit ;;
  deep_support_prior) run_pppost_deep_delta pppost_support_prior ;;
  deep_feature_reliability) run_pppost_deep_delta pppost_feature_reliability ;;
  deep_posterior_likelihood) run_pppost_deep_delta pppost_posterior_likelihood ;;
  ev2_teacher_rule_sources) run_pppost_evidence_v2_delta pppost_teacher_rule_sources ;;
  ev2_short_rule_budget) run_pppost_evidence_v2_delta pppost_short_rule_budget ;;
  ev2_theta_shrinkage) run_pppost_evidence_v2_delta pppost_theta_shrinkage ;;
  ev2_signed_logit) run_pppost_evidence_v2_delta pppost_signed_logit ;;
  ev2_sparse_logit) run_pppost_evidence_v2_delta pppost_sparse_logit ;;
  ev2_support_prior) run_pppost_evidence_v2_delta pppost_support_prior ;;
  ev2_feature_reliability) run_pppost_evidence_v2_delta pppost_feature_reliability ;;
  ev2_posterior_likelihood) run_pppost_evidence_v2_delta pppost_posterior_likelihood ;;
  teacher_anchor_teacher_rule_sources) run_pppost_teacher_anchor_delta pppost_teacher_rule_sources ;;
  teacher_anchor_short_rule_budget) run_pppost_teacher_anchor_delta pppost_short_rule_budget ;;
  teacher_anchor_theta_shrinkage) run_pppost_teacher_anchor_delta pppost_theta_shrinkage ;;
  teacher_anchor_signed_logit) run_pppost_teacher_anchor_delta pppost_signed_logit ;;
  teacher_anchor_sparse_logit) run_pppost_teacher_anchor_delta pppost_sparse_logit ;;
  teacher_anchor_support_prior) run_pppost_teacher_anchor_delta pppost_support_prior ;;
  teacher_anchor_feature_reliability) run_pppost_teacher_anchor_delta pppost_feature_reliability ;;
  teacher_anchor_posterior_likelihood) run_pppost_teacher_anchor_delta pppost_posterior_likelihood ;;
  teacher_anchor_missing_short_rule_budget) run_pppost_teacher_anchor_missing_delta pppost_short_rule_budget ;;
  teacher_anchor_missing_theta_shrinkage) run_pppost_teacher_anchor_missing_delta pppost_theta_shrinkage ;;
  teacher_anchor_missing_sparse_logit) run_pppost_teacher_anchor_missing_delta pppost_sparse_logit ;;
  teacher_anchor_missing_posterior_likelihood) run_pppost_teacher_anchor_missing_delta pppost_posterior_likelihood ;;
  rahmatullaev_rule_source_soft) run_rahmatullaev_improvement rule_source_soft ;;
  rahmatullaev_contextual_support) run_rahmatullaev_improvement contextual_support ;;
  rahmatullaev_selective_aggregation) run_rahmatullaev_improvement selective_aggregation ;;
  rahmatullaev_teacher_calibration) run_rahmatullaev_improvement teacher_calibration ;;
  rahmatullaev_ebm_anchor) run_rahmatullaev_improvement ebm_anchor ;;
  rahmatullaev_clinical_objective) run_rahmatullaev_improvement clinical_objective ;;
  rahmatullaev_ebm_correction) run_rahmatullaev_reviewer_response ebm_correction ;;
  rahmatullaev_clinical_operating_modes) run_rahmatullaev_reviewer_response clinical_operating_modes ;;
  rahmatullaev_teacher_anchor_modes) run_rahmatullaev_reviewer_response teacher_anchor_modes ;;
  rahmatullaev_rule_family_symbolic) run_rahmatullaev_reviewer_response rule_family_symbolic ;;
  rahmatullaev_audit_semantics) run_rahmatullaev_reviewer_response audit_semantics ;;
  rahmatullaev_ebm_terms_as_evidence) run_rahmatullaev_interpretable_substrate ebm_terms_as_evidence ;;
  rahmatullaev_ga2m_soft_distill) run_rahmatullaev_interpretable_substrate ga2m_soft_distill ;;
  rahmatullaev_family_theta_calibration) run_rahmatullaev_interpretable_substrate family_theta_calibration ;;
  rahmatullaev_monotone_clinical_families) run_rahmatullaev_interpretable_substrate monotone_clinical_families ;;
  rahmatullaev_redundancy_pruned_topk) run_rahmatullaev_interpretable_substrate redundancy_pruned_topk ;;
  rahmatullaev_combo_interpretable_best) run_rahmatullaev_interpretable_substrate combo_interpretable_best ;;
  rahmatullaev_ebm_bounded_residual_gate) run_rahmatullaev_interpretable_substrate_v2 ebm_bounded_residual_gate ;;
  rahmatullaev_agreement_gated_ppost) run_rahmatullaev_interpretable_substrate_v2 agreement_gated_ppost ;;
  rahmatullaev_tabpfn_to_ebm_distill) run_rahmatullaev_interpretable_substrate_v2 tabpfn_to_ebm_distill ;;
  rahmatullaev_family_utility_pruned_topk) run_rahmatullaev_interpretable_substrate_v2 family_utility_pruned_topk ;;
  rahmatullaev_operating_point_sweep) run_rahmatullaev_interpretable_substrate_v2 operating_point_sweep ;;
  rahmatullaev_monotone_plus_ebm_families) run_rahmatullaev_interpretable_substrate_v2 monotone_plus_ebm_families ;;
  rahmatullaev_bayes_llr_core) run_rahmatullaev_bayesian_llr_ppost bayes_llr_core ;;
  rahmatullaev_bayes_llr_distilled_substrate) run_rahmatullaev_bayesian_llr_ppost bayes_llr_distilled_substrate ;;
  rahmatullaev_bayes_llr_operating_modes) run_rahmatullaev_bayesian_llr_ppost bayes_llr_operating_modes ;;
  rahmatullaev_ebm_residual_core) run_rahmatullaev_ebm_residual_ppost ebm_residual_core ;;
  rahmatullaev_ebm_residual_distilled) run_rahmatullaev_ebm_residual_ppost ebm_residual_distilled ;;
  rahmatullaev_ebm_residual_operating_modes) run_rahmatullaev_ebm_residual_ppost ebm_residual_operating_modes ;;
  rahmatullaev_dual_residual_core) run_rahmatullaev_dual_residual_ppost dual_residual_core ;;
  rahmatullaev_dual_residual_teacher_conf) run_rahmatullaev_dual_residual_ppost dual_residual_teacher_conf ;;
  rahmatullaev_dual_residual_clinical_utility) run_rahmatullaev_dual_residual_ppost dual_residual_clinical_utility ;;
  rahmatullaev_dual_residual_stratified_cal) run_rahmatullaev_dual_residual_ppost dual_residual_stratified_cal ;;
  rahmatullaev_v3_ebm_evidence_objects) run_rahmatullaev_interpretable_v3 v3_ebm_evidence_objects ;;
  rahmatullaev_v3_utility_gated_fallback) run_rahmatullaev_interpretable_v3 v3_utility_gated_fallback ;;
  rahmatullaev_v3_bayesian_family_llr) run_rahmatullaev_interpretable_v3 v3_bayesian_family_llr ;;
  rahmatullaev_v3_operating_points) run_rahmatullaev_interpretable_v3 v3_operating_points ;;
  rahmatullaev_v3_residual_calibrated_gate) run_rahmatullaev_interpretable_v3 v3_residual_calibrated_gate ;;
  rahmatullaev_v3_interpretable_combo) run_rahmatullaev_interpretable_v3 v3_interpretable_combo ;;
  rahmatullaev_proof_evidence_ablation) run_rahmatullaev_ppost_proof proof_evidence_ablation ;;
  rahmatullaev_proof_selective_utility) run_rahmatullaev_ppost_proof proof_selective_utility ;;
  rahmatullaev_proof_strong_base_repair) run_rahmatullaev_ppost_proof proof_strong_base_repair ;;
  rahmatullaev_proof_audit_sufficiency) run_rahmatullaev_ppost_proof proof_audit_sufficiency ;;
  rahmatullaev_proof_operating_points) run_rahmatullaev_ppost_proof proof_operating_points ;;
  rahmatullaev_proof_randomized_controls) run_rahmatullaev_ppost_proof proof_randomized_controls ;;
  rahmatullaev_accept_trace_sufficiency_curve) run_rahmatullaev_acceptance_strengthening trace_sufficiency_curve ;;
  rahmatullaev_accept_proof_statistics) run_rahmatullaev_acceptance_strengthening proof_statistics ;;
  rahmatullaev_accept_case_study_trace_candidates) run_rahmatullaev_acceptance_strengthening case_study_trace_candidates ;;
  rahmatullaev_next_compact_residual_trace) run_rahmatullaev_acceptance_next_steps compact_residual_trace ;;
  rahmatullaev_next_subset_sufficiency) run_rahmatullaev_acceptance_next_steps subset_sufficiency ;;
  rahmatullaev_next_trace_bootstrap_ci) run_rahmatullaev_acceptance_next_steps trace_bootstrap_ci ;;
  rahmatullaev_next_ebm_ppost_case_study) run_rahmatullaev_acceptance_next_steps ebm_ppost_case_study ;;
  rahmatullaev_next_claim_checklist) run_rahmatullaev_acceptance_next_steps claim_checklist ;;
  rahmatullaev_next_ebm_source_diagnostics) run_rahmatullaev_acceptance_next_steps ebm_source_diagnostics ;;
  rahmatullaev_v2_paired_utility_ci) run_rahmatullaev_aaai_evidence_v2 paired_utility_ci ;;
  rahmatullaev_v2_rich_randomized_controls) run_rahmatullaev_aaai_evidence_v2 rich_randomized_controls ;;
  rahmatullaev_v2_source_compatibility_matrix) run_rahmatullaev_aaai_evidence_v2 source_compatibility_matrix ;;
  rahmatullaev_v2_extended_trace_curve) run_rahmatullaev_aaai_evidence_v2 extended_trace_curve ;;
  rahmatullaev_v2_native_wrong_correction) run_rahmatullaev_aaai_evidence_v2 native_wrong_correction ;;
  rahmatullaev_v2_operating_point_separation) run_rahmatullaev_aaai_evidence_v2 operating_point_separation ;;
  rahmatullaev_v2_case_trace_candidates) run_rahmatullaev_aaai_evidence_v2 case_trace_candidates ;;
  rahmatullaev_v2_external_tabular_sanity) run_rahmatullaev_aaai_evidence_v2 external_tabular_sanity ;;
  rahmatullaev_v2_component_ablation) run_rahmatullaev_aaai_evidence_v2 component_ablation ;;
  rahmatullaev_v2_statistical_summary) run_rahmatullaev_aaai_evidence_v2 statistical_summary ;;
  rahmatullaev_claim_contract) run_rahmatullaev_aaai_claim_package claim_contract ;;
  rahmatullaev_claim_source_boundary_map) run_rahmatullaev_aaai_claim_package source_boundary_map ;;
  rahmatullaev_claim_control_gap_audit) run_rahmatullaev_aaai_claim_package control_gap_audit ;;
  rahmatullaev_claim_trace_sufficiency_refresh) run_rahmatullaev_aaai_claim_package trace_sufficiency_refresh ;;
  rahmatullaev_claim_reviewer_trace_examples) run_rahmatullaev_aaai_claim_package reviewer_trace_examples ;;
  rahmatullaev_claim_package_summary) run_rahmatullaev_aaai_claim_package claim_package_summary ;;
  rahmatullaev_final_slim_usefulness) run_rahmatullaev_aaai_final_strengthening slim_usefulness ;;
  rahmatullaev_final_replay_integrity) run_rahmatullaev_aaai_final_strengthening replay_integrity ;;
  rahmatullaev_final_clinical_trace) run_rahmatullaev_aaai_final_strengthening clinical_trace ;;
  rahmatullaev_final_deletion_sufficiency) run_rahmatullaev_aaai_final_strengthening deletion_sufficiency ;;
  rahmatullaev_final_failure_modes) run_rahmatullaev_aaai_final_strengthening failure_modes ;;
  rahmatullaev_eicu_rulefit_official) run_rahmatullaev_eicu_strengthening rulefit_official ;;
  rahmatullaev_eicu_operating_points) run_rahmatullaev_eicu_strengthening operating_points ;;
  rahmatullaev_eicu_measurement_pattern_families) run_rahmatullaev_eicu_strengthening measurement_pattern_families ;;
  rahmatullaev_eicu_measurement_policy_calibration) run_rahmatullaev_eicu_strengthening measurement_policy_calibration ;;
  rahmatullaev_eicu_family_pruning_sweep) run_rahmatullaev_eicu_strengthening family_pruning_sweep ;;
  rahmatullaev_stress_ebm_vs_ppost_audit) run_rahmatullaev_aaai_reviewer_stress ebm_vs_ppost_audit_mechanism ;;
  rahmatullaev_stress_conditional_utility) run_rahmatullaev_aaai_reviewer_stress conditional_utility_slices ;;
  rahmatullaev_stress_trace_compression_v2) run_rahmatullaev_aaai_reviewer_stress trace_compression_curve_v2 ;;
  rahmatullaev_stress_teacher_anchor_calibration) run_rahmatullaev_aaai_reviewer_stress teacher_anchor_calibration_modes ;;
  rahmatullaev_stress_measurement_policy_v2) run_rahmatullaev_aaai_reviewer_stress eicu_measurement_policy_v2 ;;
  rahmatullaev_accept_clinician_audit_packet) run_rahmatullaev_aaai_acceptance_clinician_symbolic clinician_audit_packet ;;
  rahmatullaev_accept_clean_interpretable_calibrated) run_rahmatullaev_aaai_acceptance_clinician_symbolic clean_interpretable_calibrated ;;
  rahmatullaev_accept_symbolic_family_calibrated) run_rahmatullaev_aaai_acceptance_clinician_symbolic symbolic_family_calibrated ;;
  rahmatullaev_accept_supplement_slimming_manifest) run_rahmatullaev_aaai_acceptance_clinician_symbolic supplement_slimming_manifest ;;
  rahmatullaev_symbolic_rulefit_calibrated) run_rahmatullaev_symbolic_clean_ppost rulefit_calibrated_evidence ;;
  rahmatullaev_symbolic_figs_bounded) run_rahmatullaev_symbolic_clean_ppost figs_bounded_residual ;;
  rahmatullaev_symbolic_auditselect) run_rahmatullaev_symbolic_clean_ppost rulefit_figs_auditselect ;;
  rahmatullaev_symbolic_family_ppost) run_rahmatullaev_symbolic_clean_ppost symbolic_family_ppost ;;
  rahmatullaev_symbolic_thresholding) run_rahmatullaev_symbolic_clean_ppost calibration_constrained_thresholding ;;
  temporal_main) run_temporal_main ;;
  temporal_ablations) run_temporal_ablations ;;
  case_studies) run_case_studies ;;
  all)
    run_tabular_main && run_tabular_ablations && run_tabular_rule_sources && \
    run_interpretability_story && run_temporal_main && run_temporal_ablations && run_case_studies
    ;;
  all_with_tabpfn|paper_full_tabpfn)
    PAPER_PRESET=full_tabpfn
    TABPFN_STAGES=1
    TS_TEACHER_BACKEND=tabpfn_ts
    TS_TEACHER_DEVICE="${TS_TEACHER_DEVICE:-cuda}"
    TEMPORAL_BASELINES=all
    INCLUDE_TABPFN_TS_DISTILL=1
    INCLUDE_TABPFN_TS_BASELINE=1
    run_tabular_main && run_tabular_ablations && run_tabular_rule_sources && \
    run_tabular_tabpfn_distill && run_tabular_ensembles && run_interpretability_story && \
    run_temporal_main && run_temporal_ablations && run_case_studies
    ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
rc=$?
echo "[done] DATASET=$DATASET STAGE=$STAGE MODE=$MODE rc=$rc $(date -Iseconds)"
exit $rc
