#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATASETS_STR="${DATASETS:-mimic3 mimic4 eicu}"
LEVELS_STR="${LEVELS:-L1 L2 L3 L4}"
TEMPORAL_BASELINES_STR="${TEMPORAL_BASELINES:-all}"
TEMPORAL_EXTRA_ARGS_STR="${TEMPORAL_EXTRA_ARGS:-}"
TABULAR_EXTRA_ARGS_STR="${TABULAR_EXTRA_ARGS:-}"

read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a LEVELS_ARR <<< "$LEVELS_STR"
read -r -a TEMPORAL_BASELINES_ARR <<< "$TEMPORAL_BASELINES_STR"
read -r -a TEMPORAL_EXTRA_ARGS_ARR <<< "$TEMPORAL_EXTRA_ARGS_STR"
read -r -a TABULAR_EXTRA_ARGS_ARR <<< "$TABULAR_EXTRA_ARGS_STR"

FOLDS="${FOLDS:-3}"
EPOCHS="${EPOCHS:-80}"
EXPENSIVE_EPOCHS="${EXPENSIVE_EPOCHS:-80}"
PREPROCESS_OUTPUT="${PREPROCESS_OUTPUT:-data/processed/mortality}"
EVENT_CACHE_DIR="${EVENT_CACHE_DIR:-data/processed/mortality_event_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/mortality}"
LOG_ROOT="${LOG_ROOT:-logs/mortality}"
TABULAR_VARIANTS="${TABULAR_VARIANTS:-all}"
RULE_SOURCES="${RULE_SOURCES:-all}"
TABULAR_BASELINES="${TABULAR_BASELINES:-all}"
PREPROCESS_MAX_SAMPLES="${PREPROCESS_MAX_SAMPLES:-}"
PREPROCESS_CHUNKSIZE="${PREPROCESS_CHUNKSIZE:-}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
DISABLE_EVENT_CACHE="${DISABLE_EVENT_CACHE:-0}"
REBUILD_EVENT_CACHE="${REBUILD_EVENT_CACHE:-0}"
INCLUDE_TABPFN_TS_DISTILL="${INCLUDE_TABPFN_TS_DISTILL:-0}"

mkdir -p "$PREPROCESS_OUTPUT" "$EVENT_CACHE_DIR" "$OUTPUT_ROOT" "$LOG_ROOT"
export MORTALITY_PROCESSED_DIR="$PREPROCESS_OUTPUT"
export MORTALITY_EVENT_CACHE_DIR="$EVENT_CACHE_DIR"

run_logged() {
  local name="$1"
  shift
  local log="$LOG_ROOT/${name}.log"
  printf '
[%s] %s
' "$(date '+%F %T')" "$*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
}

for dataset in "${DATASETS_ARR[@]}"; do
  case "$dataset" in
    mimic3|mimic4|eicu) ;;
    *) echo "Unknown mortality dataset: $dataset" >&2; exit 2 ;;
  esac

  temporal_dataset="${dataset}_mortality"
  tabular_cache="$PREPROCESS_OUTPUT/${dataset}_mortality_48h_tabular.npz"
  temporal_out="$OUTPUT_ROOT/${dataset}_temporal"
  tabular_out="$OUTPUT_ROOT/${dataset}_tabular"

  preprocess_args=(
    --datasets "$dataset"
    --output-dir "$PREPROCESS_OUTPUT"
    --event-cache-dir "$EVENT_CACHE_DIR"
  )
  if [[ -n "$PREPROCESS_MAX_SAMPLES" ]]; then
    preprocess_args+=(--max-samples "$PREPROCESS_MAX_SAMPLES")
  fi
  if [[ -n "$PREPROCESS_CHUNKSIZE" ]]; then
    preprocess_args+=(--chunksize "$PREPROCESS_CHUNKSIZE")
  fi
  if [[ "$FORCE_PREPROCESS" == "1" ]]; then
    preprocess_args+=(--force)
  fi
  if [[ "$DISABLE_EVENT_CACHE" == "1" ]]; then
    preprocess_args+=(--no-event-cache)
  fi
  if [[ "$REBUILD_EVENT_CACHE" == "1" ]]; then
    preprocess_args+=(--rebuild-event-cache)
  fi

  run_logged "${dataset}_01_preprocess"     python -m temporal.mortality_preprocess "${preprocess_args[@]}"

  temporal_args=(
    --datasets "$temporal_dataset"
    --levels "${LEVELS_ARR[@]}"
    --folds "$FOLDS"
    --epochs "$EPOCHS"
    --output-dir "$temporal_out"
  )
  if [[ "${TEMPORAL_BASELINES_ARR[*]}" != "none" ]]; then
    temporal_args+=(--baselines "${TEMPORAL_BASELINES_ARR[@]}")
  fi
  if [[ "$INCLUDE_TABPFN_TS_DISTILL" == "1" ]]; then
    temporal_args+=(--include-tabpfn-ts-distill)
  fi
  if (( ${#TEMPORAL_EXTRA_ARGS_ARR[@]} )); then
    temporal_args+=("${TEMPORAL_EXTRA_ARGS_ARR[@]}")
  fi

  run_logged "${dataset}_02_temporal"     python -m temporal.compare_temporal "${temporal_args[@]}"

  tabular_args=(
    --datasets "npz:$tabular_cache"
    --variants "$TABULAR_VARIANTS"
    --rule-sources "$RULE_SOURCES"
    --baselines "$TABULAR_BASELINES"
    --folds "$FOLDS"
    --epochs "$EPOCHS"
    --expensive-epochs "$EXPENSIVE_EPOCHS"
    --output-dir "$tabular_out"
  )
  if (( ${#TABULAR_EXTRA_ARGS_ARR[@]} )); then
    tabular_args+=("${TABULAR_EXTRA_ARGS_ARR[@]}")
  fi

  run_logged "${dataset}_03_tabular"     python compare_datasets.py "${tabular_args[@]}"
done
