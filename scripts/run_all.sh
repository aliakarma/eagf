#!/usr/bin/env bash
# ============================================================
# scripts/run_all.sh
# EAGF Master Experiment Script
#
# Reproduces ALL results, tables, and figures from the paper:
#   "Ethical AI Governance for Cybersecurity in RE-IoT Systems"
#
# Usage:
#   bash scripts/run_all.sh [OPTIONS]
#
# Options:
#   --seeds     SEED_LIST   Space-separated random seeds (default: "42 123 456")
#   --config    CONFIG_DIR  Path to config directory (default: configs/)
#   --output    OUTPUT_DIR  Path to results directory (default: results/)
#   --device    DEVICE      pytorch device string (default: cuda:0 or cpu)
#   --skip-data             Skip dataset download/generation
#   --skip-ablation         Skip ablation study (saves ~18 h)
#   --dry-run               Print commands without executing
#
# Example:
#   bash scripts/run_all.sh --seeds "42 123 456" --output results/ --device cuda:0
# ============================================================

set -euo pipefail

# ── Default arguments ─────────────────────────────────────
SEEDS="42 123 456"
CONFIG_DIR="configs"
OUTPUT_DIR="results"
DEVICE="cuda:0"
SKIP_DATA=false
SKIP_ABLATION=false
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)        SEEDS="$2";      shift 2 ;;
    --config)       CONFIG_DIR="$2"; shift 2 ;;
    --output)       OUTPUT_DIR="$2"; shift 2 ;;
    --device)       DEVICE="$2";     shift 2 ;;
    --skip-data)    SKIP_DATA=true;  shift   ;;
    --skip-ablation) SKIP_ABLATION=true; shift ;;
    --dry-run)      DRY_RUN=true;   shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helper: run or print ──────────────────────────────────
run() {
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    eval "$*"
  fi
}

# ── Logging setup ─────────────────────────────────────────
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "========================================================"
echo "  EAGF: Full Experiment Pipeline"
echo "  Started: $(date)"
echo "  Seeds:   ${SEEDS}"
echo "  Device:  ${DEVICE}"
echo "  Output:  ${OUTPUT_DIR}"
echo "========================================================"

# ── Step 0: Environment check ─────────────────────────────
echo ""
echo "[Step 0] Checking environment..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import opacus; print(f'  Opacus {opacus.__version__}')"
python -c "import shap; print(f'  SHAP {shap.__version__}')"

# ── Step 1: Data preparation ──────────────────────────────
if [ "$SKIP_DATA" = false ]; then
  echo ""
  echo "[Step 1] Preparing datasets..."

  echo "  [1a] Downloading / preparing EFR biometric dataset..."
  run "python -m src.utils.data_loader \
        --dataset efr \
        --output  ${OUTPUT_DIR}/../data/biometric/ \
        --seed    42"

  echo "  [1b] Generating RE-IoT synthetic telemetry..."
  run "python -m src.utils.data_loader \
        --dataset     reiot \
        --output      ${OUTPUT_DIR}/../data/reiot/ \
        --nodes       120 \
        --attack-ratio 0.05 \
        --seed        42"
else
  echo "[Step 1] Skipping data preparation (--skip-data set)."
fi

# ── Step 2: Biometric case study — full EAGF (M5) ─────────
echo ""
echo "[Step 2] Running EAGF biometric case study (M5, 3 seeds)..."

for SEED in ${SEEDS}; do
  echo "  [Seed ${SEED}] EAGF biometric..."
  run "python -m src.training.eagf_trainer \
        --config  ${CONFIG_DIR}/biometric_default.yaml \
        --model   eagf \
        --seed    ${SEED} \
        --device  ${DEVICE} \
        --output  ${OUTPUT_DIR}/biometric/seed_${SEED}/ \
        2>&1 | tee ${LOG_DIR}/biometric_eagf_seed${SEED}.log"
done

# ── Step 3: Ablation study (M0–M5) ───────────────────────
if [ "$SKIP_ABLATION" = false ]; then
  echo ""
  echo "[Step 3] Running ablation study (M0–M5, 3 seeds)..."
  run "bash scripts/run_ablation.sh \
        --config  ${CONFIG_DIR}/biometric_default.yaml \
        --seeds   \"${SEEDS}\" \
        --device  ${DEVICE} \
        --output  ${OUTPUT_DIR}/biometric/ablation/ \
        2>&1 | tee ${LOG_DIR}/ablation.log"
else
  echo "[Step 3] Skipping ablation study (--skip-ablation set)."
fi

# ── Step 4: Pareto-front hyperparameter search ────────────
echo ""
echo "[Step 4] Running Pareto-front search (5×5 grid, seed 42)..."
run "python -m src.training.pareto_trainer \
      --config           ${CONFIG_DIR}/biometric_default.yaml \
      --lambda-rp-range  1e-3 1e0 --lambda-rp-steps 5 \
      --lambda-c-range   1e-3 1e0 --lambda-c-steps  5 \
      --seed             42 \
      --device           ${DEVICE} \
      --output           ${OUTPUT_DIR}/biometric/pareto/ \
      2>&1 | tee ${LOG_DIR}/pareto_search.log"

# ── Step 5: RE-IoT case study ─────────────────────────────
echo ""
echo "[Step 5] Running RE-IoT anomaly detection case study..."
run "bash scripts/run_reiot.sh \
      --config  ${CONFIG_DIR}/reiot_default.yaml \
      --seeds   \"${SEEDS}\" \
      --device  cpu \
      --output  ${OUTPUT_DIR}/reiot/ \
      2>&1 | tee ${LOG_DIR}/reiot.log"

# ── Step 6: Trust Index computation ──────────────────────
echo ""
echo "[Step 6] Computing Trust Index across all models and seeds..."
run "python -m src.metrics.trust_index \
      --results-dir ${OUTPUT_DIR}/biometric/ \
      --weights     equal \
      --output      ${OUTPUT_DIR}/biometric/trust_index_summary.json \
      2>&1 | tee ${LOG_DIR}/trust_index.log"

# ── Step 7: Statistical tests ─────────────────────────────
echo ""
echo "[Step 7] Running statistical significance tests..."
run "python -m src.evaluation.statistics \
      --baseline-dir  ${OUTPUT_DIR}/biometric/baseline/ \
      --eagf-dir      ${OUTPUT_DIR}/biometric/eagf/ \
      --output        ${OUTPUT_DIR}/biometric/statistical_tests.json \
      2>&1 | tee ${LOG_DIR}/statistics.log"

# ── Step 8: Figure generation ─────────────────────────────
echo ""
echo "[Step 8] Generating paper figures..."

echo "  [8a] Figure 3: Ablation bar chart..."
run "python src/utils/visualisation.py \
      --results ${OUTPUT_DIR}/biometric/ablation/ablation_summary.csv \
      --type    ablation_bar \
      --output  figures/figure3.png"

echo "  [8b] Pareto-front plot..."
run "python src/utils/visualisation.py \
      --results ${OUTPUT_DIR}/biometric/pareto/pareto_results.csv \
      --type    pareto_front \
      --output  figures/pareto_front.png"

echo "  [8c] RE-IoT FPRP comparison..."
run "python src/utils/visualisation.py \
      --results ${OUTPUT_DIR}/reiot/node_class_results.csv \
      --type    fprp_bar \
      --output  figures/reiot_fprp.png"

# ── Step 9: Summary report ────────────────────────────────
echo ""
echo "[Step 9] Generating summary report..."
run "python -m src.evaluation.report_generator \
      --biometric-results ${OUTPUT_DIR}/biometric/ \
      --reiot-results     ${OUTPUT_DIR}/reiot/ \
      --output            ${OUTPUT_DIR}/summary_report.md"

# ── Done ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  EAGF: Full Pipeline Complete"
echo "  Finished: $(date)"
echo ""
echo "  Key output files:"
echo "    Tables  → ${OUTPUT_DIR}/biometric/ablation_summary.csv"
echo "            → ${OUTPUT_DIR}/biometric/main_results.csv"
echo "            → ${OUTPUT_DIR}/reiot/node_class_results.csv"
echo "    Figures → figures/figure3.png"
echo "            → figures/pareto_front.png"
echo "            → figures/reiot_fprp.png"
echo "    Report  → ${OUTPUT_DIR}/summary_report.md"
echo "    Logs    → ${LOG_DIR}/"
echo "========================================================"
