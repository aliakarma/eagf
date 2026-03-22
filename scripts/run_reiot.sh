#!/usr/bin/env bash
# ============================================================
# scripts/run_reiot.sh
# EAGF RE-IoT Anomaly Detection Case Study Runner (Case Study 2)
#
# Generates synthetic 5G RE-IoT telemetry and trains/evaluates
# the EAGF framework for node-class fairness (FPRP) and anomaly
# detection across urban, peri-urban, and rural node classes.
#
# Usage:
#   bash scripts/run_reiot.sh [--seeds "42 123 456"] [--device cpu]
# ============================================================

set -euo pipefail

SEEDS="42 123 456"
DEVICE="cpu"
CONFIG="configs/reiot_default.yaml"
OUTPUT_DIR="results/reiot"
DATA_DIR="data/reiot"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)  SEEDS="$2";      shift 2 ;;
    --device) DEVICE="$2";     shift 2 ;;
    --config) CONFIG="$2";     shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}" "${DATA_DIR}"
echo "EAGF RE-IoT Case Study | Config: ${CONFIG} | Device: ${DEVICE}"

# Generate synthetic telemetry (deterministic given seed)
echo ""
echo "Generating RE-IoT synthetic telemetry..."
python -m src.utils.data_loader \
  --dataset      reiot \
  --output       "${DATA_DIR}/" \
  --nodes        120 \
  --attack-ratio 0.05 \
  --seed         42

for SEED in ${SEEDS}; do
  echo ""
  echo "── Seed ${SEED} ──"

  # Baseline model
  python -m src.training.eagf_trainer \
    --config  "${CONFIG}" \
    --model   baseline \
    --seed    "${SEED}" \
    --device  "${DEVICE}" \
    --output  "${OUTPUT_DIR}/baseline/seed_${SEED}/" \
    2>&1 | tee "${OUTPUT_DIR}/baseline/seed_${SEED}/train.log"

  # Full EAGF model
  python -m src.training.eagf_trainer \
    --config  "${CONFIG}" \
    --model   eagf \
    --seed    "${SEED}" \
    --device  "${DEVICE}" \
    --output  "${OUTPUT_DIR}/eagf/seed_${SEED}/" \
    2>&1 | tee "${OUTPUT_DIR}/eagf/seed_${SEED}/train.log"
done

# Compute node-class FPRP and aggregate results
python -m src.evaluation.baseline \
  --results-dir "${OUTPUT_DIR}" \
  --seeds       ${SEEDS} \
  --fairness-criterion fprp \
  --output      "${OUTPUT_DIR}/node_class_results.csv"

echo ""
echo "RE-IoT case study complete. Results → ${OUTPUT_DIR}/node_class_results.csv"
