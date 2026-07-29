#!/usr/bin/env bash
# Full reviewer-defense run for the hospital mortality 48h task.
# Sections: 25, 26, 27, 28, 29, 30, 31, 32.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO"
PY="${PY_OVERRIDE:-$REPO/.venv/bin/python}"
RUN_ID="${RUN_ID:-hospital_mortality_48h_sections_25_32_v1}"
RUN_ROOT="${RUN_ROOT:-$REPO/output/paper/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/paper/$RUN_ID}"
CASE_ROOT="${CASE_ROOT:-$REPO/output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v3}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
LOG="$LOG_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$LOG" "$LOG_ROOT/latest.log"
exec > >(tee -a "$LOG") 2>&1

STATUS="$RUN_ROOT/RUN_STATUS.tsv"
printf "section\tstatus\ttimestamp\trc\tdetails\n" > "$STATUS"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
export TABPFN_IGNORE_PRETRAINING_LIMITS="${TABPFN_IGNORE_PRETRAINING_LIMITS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CKPT_DIR="$REPO/data/tabpfn_checkpoints"
if [ -z "${TABPFN_CLASSIFIER_MODEL_PATH:-}" ] && [ -f "$CKPT_DIR/tabpfn-v3-classifier-v3_default.ckpt" ]; then
  export TABPFN_CLASSIFIER_MODEL_PATH="$CKPT_DIR/tabpfn-v3-classifier-v3_default.ckpt"
fi

log_status() {
  local section="$1" status="$2" rc="$3" details="${4:-}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$section" "$status" "$(date -Iseconds)" "$rc" "$details" >> "$STATUS"
}

run_step() {
  local section="$1"
  shift
  echo
  echo "===================================================================================================="
  echo "[$(date -Iseconds)] START $section"
  echo "CMD: $*"
  echo "===================================================================================================="
  log_status "$section" start 0 "$*"
  set +e
  "$@"
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    log_status "$section" done "$rc" ok
    echo "[$(date -Iseconds)] DONE $section rc=$rc"
  else
    log_status "$section" failed "$rc" failed
    echo "[$(date -Iseconds)] FAILED $section rc=$rc" >&2
    exit "$rc"
  fi
}

cat > "$RUN_ROOT/RUN_MANIFEST.md" <<EOF
# Hospital Mortality 48h Reviewer-Defense Run

Run id: $RUN_ID
Started: $(date -Iseconds)
Python: $PY
Case root: $CASE_ROOT
Run root: $RUN_ROOT
Log: $LOG

Task: hospital mortality prediction from first 48 ICU hours.
Datasets:
- npz:data/processed/mortality/mimic3_mortality_48h_tabular.npz
- npz:data/processed/mortality/mimic4_mortality_48h_tabular.npz
- npz:data/processed/mortality/eicu_mortality_48h_tabular.npz

Sections: 25, 26, 27, 28, 29, 30, 31, 32.
Section 33 is not launched here because it is OpenML/generalization, not hospital mortality.
EOF

"$PY" - <<'PYCHECK'
import sys, torch, numpy, pandas, sklearn, problog
print("python", sys.executable, sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "pandas", pandas.__version__, "sklearn", sklearn.__version__)
print("problog", getattr(problog, "__version__", "no_version"))
PYCHECK

# Run Sections 25-32 in numeric order. Section 29 is first written in-order,
# then refreshed after 30-32 so the final bundle points to every artifact.
run_step section_25_audit_validation \
  "$PY" paper_experiments/section_25_audit_validation.py \
  --case-root "$CASE_ROOT" \
  --output-dir "$RUN_ROOT/25_audit_validation"

# Long step: recomputes clinical metrics and saves prediction_artifacts/*.npz.
run_step section_26_clinical_metrics \
  "$PY" paper_experiments/section_26_clinical_metrics.py \
  --datasets \
    "npz:data/processed/mortality/mimic3_mortality_48h_tabular.npz" \
    "npz:data/processed/mortality/mimic4_mortality_48h_tabular.npz" \
    "npz:data/processed/mortality/eicu_mortality_48h_tabular.npz" \
  --folds "${FOLDS:-3}" \
  --output-dir "$RUN_ROOT/26_clinical_metrics"

CLIN_CSV="$(ls -t "$RUN_ROOT"/26_clinical_metrics/compare_datasets_*.csv | head -n 1)"
CLIN_JSONL="${CLIN_CSV%.csv}.jsonl"
printf "clinical_csv\t%s\nclinical_jsonl\t%s\n" "$CLIN_CSV" "$CLIN_JSONL" > "$RUN_ROOT/LATEST_ARTIFACTS.tsv"

run_step section_27_uncertainty_noninferiority \
  "$PY" paper_experiments/section_27_uncertainty_noninferiority.py \
  --csv "$CLIN_CSV" \
  --output-dir "$RUN_ROOT/27_uncertainty_noninferiority" \
  --metrics accuracy,mcc,balanced_accuracy,f1_macro,roc_auc_ovr,auprc_ovr,log_loss,brier_score,ece_10,ece_20,sensitivity,specificity,net_benefit_0_10,net_benefit_0_20 \
  --method-contains "PPtheta-Post" \
  --comparator-contains "TabPFN" \
  --n-bootstrap "${N_BOOTSTRAP_FOLD:-10000}"

run_step section_28_prediction_artifact_metrics \
  "$PY" paper_experiments/section_28_prediction_artifact_metrics.py \
  --csv "$CLIN_CSV" \
  --output-dir "$RUN_ROOT/28_prediction_artifact_metrics" \
  --n-bootstrap "${N_BOOTSTRAP_PATIENT:-1000}" \
  --resume \
  --checkpoint-every "${SECTION28_CHECKPOINT_EVERY:-1}" \
  --max-artifacts "${SECTION28_MAX_ARTIFACTS:-0}"

run_step section_29_reviewer_defense_report \
  "$PY" paper_experiments/section_29_reviewer_defense_report.py \
  --audit-summary "$RUN_ROOT/25_audit_validation/AUDIT_VALIDATION_SUMMARY.md" \
  --clinical-csv "$CLIN_CSV" \
  --uncertainty-csv "$RUN_ROOT/27_uncertainty_noninferiority/method_metric_ci.csv" \
  --patient-bootstrap-csv "$RUN_ROOT/28_prediction_artifact_metrics/patient_bootstrap_metrics.csv" \
  --output-dir "$RUN_ROOT/29_reviewer_defense_report"

run_step section_30_posterior_parity_complexity \
  "$PY" paper_experiments/section_30_posterior_parity_complexity.py \
  --output-dir "$RUN_ROOT/30_posterior_parity_complexity"

run_step section_31_audit_faithfulness \
  "$PY" paper_experiments/section_31_audit_faithfulness.py \
  --case-root "$CASE_ROOT" \
  --output-dir "$RUN_ROOT/31_audit_faithfulness"

run_step section_32_explanation_baselines \
  "$PY" paper_experiments/section_32_explanation_baselines.py \
  --audit-faithfulness-csv "$RUN_ROOT/31_audit_faithfulness/audit_faithfulness_summary.csv" \
  --output-dir "$RUN_ROOT/32_explanation_baselines"

run_step section_29_final_refresh \
  "$PY" paper_experiments/section_29_reviewer_defense_report.py \
  --audit-summary "$RUN_ROOT/25_audit_validation/AUDIT_VALIDATION_SUMMARY.md" \
  --clinical-csv "$CLIN_CSV" \
  --uncertainty-csv "$RUN_ROOT/27_uncertainty_noninferiority/method_metric_ci.csv" \
  --patient-bootstrap-csv "$RUN_ROOT/28_prediction_artifact_metrics/patient_bootstrap_metrics.csv" \
  --parity-summary "$RUN_ROOT/30_posterior_parity_complexity/POSTERIOR_PARITY_COMPLEXITY.md" \
  --audit-faithfulness-summary "$RUN_ROOT/31_audit_faithfulness/AUDIT_FAITHFULNESS_SUMMARY.md" \
  --explanation-protocol "$RUN_ROOT/32_explanation_baselines/EXPLANATION_BASELINE_PROTOCOL.md" \
  --output-dir "$RUN_ROOT/29_reviewer_defense_report"

cat >> "$RUN_ROOT/RUN_MANIFEST.md" <<EOF

Completed: $(date -Iseconds)
Clinical CSV: $CLIN_CSV
Clinical JSONL: $CLIN_JSONL
Prediction artifacts: $RUN_ROOT/26_clinical_metrics/prediction_artifacts
Reviewer bundle: $RUN_ROOT/29_reviewer_defense_report/REVIEWER_DEFENSE_BUNDLE.md
EOF

echo
cat "$RUN_ROOT/LATEST_ARTIFACTS.tsv"
echo "RUN_ROOT=$RUN_ROOT"
echo "STATUS=$STATUS"
echo "LOG=$LOG"
echo "[done] $RUN_ID $(date -Iseconds)"
