#!/usr/bin/env bash
# Regenerate counterfactual-ready case-study audit JSON and rerun Sections 25/31.
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO" || exit 3

PY="${PY_OVERRIDE:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}"
export MORTALITY_PROCESSED_DIR="${MORTALITY_PROCESSED_DIR:-$REPO/data/processed/mortality}"

VER="${VER:-audit_counterfactual_v1}"
CASE_ROOT="${CASE_ROOT:-$REPO/output/mortality_paper_jobs/$VER}"
RUN_ROOT="${RUN_ROOT:-$REPO/output/paper/hospital_mortality_${VER}}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/audit_counterfactual_25_31/$VER}"
CASE_EPOCHS="${CASE_EPOCHS:-80}"
CASE_N="${CASE_N:-3}"
CASE_TOP_K="${CASE_TOP_K:-5}"
AUDIT_K_VALUES="${AUDIT_K_VALUES:-1,3,5}"
CASE_LEVEL="${CASE_LEVEL:-L4}"
CASE_DATASETS="${CASE_DATASETS:-mimic3 mimic4 eicu}"
CASE_SELECTION="${CASE_SELECTION:-balanced_true}"
CASE_DECISION_THRESHOLD="${CASE_DECISION_THRESHOLD:-}"
RANDOM_BASELINE_SAMPLES="${RANDOM_BASELINE_SAMPLES:-20}"
PREVIEW_TOP_N="${PREVIEW_TOP_N:-5}"

mkdir -p "$CASE_ROOT" "$RUN_ROOT" "$LOG_ROOT"
STATUS="$RUN_ROOT/RUN_STATUS.tsv"
printf "timestamp\tstep\tstatus\trc\tduration_sec\n" > "$STATUS"

run_step() {
  local step="$1"
  shift
  local start end rc
  start=$(date +%s)
  echo "[$(date -Is)] start $step"
  "$@"
  rc=$?
  end=$(date +%s)
  if [ "$rc" -eq 0 ]; then
    printf "%s\t%s\tdone\t%s\t%s\n" "$(date -Is)" "$step" "$rc" "$((end-start))" >> "$STATUS"
  else
    printf "%s\t%s\tfailed\t%s\t%s\n" "$(date -Is)" "$step" "$rc" "$((end-start))" >> "$STATUS"
    return "$rc"
  fi
}

for ds in $CASE_DATASETS; do
  case_args=(
    "$PY" -m temporal.case_studies
    --dataset "${ds}_mortality"
    --level "$CASE_LEVEL"
    --n-samples "$CASE_N"
    --top-k "$CASE_TOP_K"
    --audit-k-values "$AUDIT_K_VALUES"
    --case-selection "$CASE_SELECTION"
    --random-baseline-samples "$RANDOM_BASELINE_SAMPLES"
    --preview-top-n "$PREVIEW_TOP_N"
    --epochs "$CASE_EPOCHS"
    --output-dir "$CASE_ROOT/$ds/case_studies"
  )
  if [ -n "$CASE_DECISION_THRESHOLD" ]; then
    case_args+=(--decision-threshold "$CASE_DECISION_THRESHOLD")
  fi
  run_step "case_studies_${ds}" "${case_args[@]}" || exit $?
done

run_step section_25_audit_validation \
  "$PY" paper_experiments/section_25_audit_validation.py \
    --case-root "$CASE_ROOT" \
    --output-dir "$RUN_ROOT/25_audit_validation" || exit $?

run_step section_31_audit_faithfulness \
  "$PY" paper_experiments/section_31_audit_faithfulness.py \
    --case-root "$CASE_ROOT" \
    --k-values "$AUDIT_K_VALUES" \
    --output-dir "$RUN_ROOT/31_audit_faithfulness" || exit $?

cat > "$RUN_ROOT/LATEST_ARTIFACTS.tsv" <<EOF
case_root	$CASE_ROOT
section25_summary	$RUN_ROOT/25_audit_validation/AUDIT_VALIDATION_SUMMARY.md
section31_summary	$RUN_ROOT/31_audit_faithfulness/AUDIT_FAITHFULNESS_SUMMARY.md
section31_csv	$RUN_ROOT/31_audit_faithfulness/audit_faithfulness_summary.csv
EOF

echo "[$(date -Is)] complete"
echo "CASE_ROOT=$CASE_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
