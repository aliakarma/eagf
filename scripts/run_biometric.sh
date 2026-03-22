#!/usr/bin/env bash
# ============================================================
# scripts/run_biometric.sh
# EAGF Biometric Case Study Runner (Case Study 1)
#
# Trains and evaluates the full EAGF model (M5) on the EFR
# facial-recognition dataset with three random seeds.
#
# Usage:
#   bash scripts/run_biometric.sh [--seeds "42 123 456"] [--device cuda:0]
# ============================================================

set -euo pipefail

SEEDS="42 123 456"
DEVICE="cuda:0"
CONFIG="configs/biometric_default.yaml"
OUTPUT_DIR="results/biometric"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)  SEEDS="$2";      shift 2 ;;
    --device) DEVICE="$2";     shift 2 ;;
    --config) CONFIG="$2";     shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
echo "EAGF Biometric Case Study | Config: ${CONFIG} | Device: ${DEVICE}"

for SEED in ${SEEDS}; do
  echo ""
  echo "── Seed ${SEED} ──"
  python -m src.training.eagf_trainer \
    --config  "${CONFIG}" \
    --model   eagf \
    --seed    "${SEED}" \
    --device  "${DEVICE}" \
    --output  "${OUTPUT_DIR}/seed_${SEED}/" \
    2>&1 | tee "${OUTPUT_DIR}/seed_${SEED}/train.log"
done

# Aggregate and compute final metrics with confidence intervals
python -m src.evaluation.baseline \
  --results-dir "${OUTPUT_DIR}" \
  --seeds       ${SEEDS} \
  --output      "${OUTPUT_DIR}/main_results.csv"

echo ""
echo "Biometric case study complete. Results → ${OUTPUT_DIR}/main_results.csv"
