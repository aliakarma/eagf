#!/usr/bin/env bash
# ============================================================
# scripts/run_pareto_search.sh
# EAGF Pareto-Front Hyperparameter Search
#
# Sweeps a 5×5 logarithmic grid of (lambda_RP, lambda_C) values,
# trains one model per grid point, records all four pillar scores,
# and identifies the Pareto-optimal solution with highest TI.
#
# This corresponds to Section 3.6 (MOO Formulation) in the paper.
#
# Usage:
#   bash scripts/run_pareto_search.sh [OPTIONS]
#
# Options:
#   --config CONFIG   Path to YAML config file
#   --seed   SEED     Random seed for training runs (default: 42)
#   --device DEVICE   PyTorch device string (default: cuda:0)
#   --output DIR      Output directory
# ============================================================

set -euo pipefail

CONFIG="configs/biometric_default.yaml"
SEED=42
DEVICE="cuda:0"
OUTPUT_DIR="results/biometric/pareto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2";     shift 2 ;;
    --seed)   SEED="$2";       shift 2 ;;
    --device) DEVICE="$2";     shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
echo "EAGF Pareto-Front Search"
echo "  Grid: 5×5 (lambda_RP × lambda_C), range [1e-3, 1e0]"
echo "  Seed: ${SEED} | Device: ${DEVICE}"
echo "  Total training runs: 25"

python -m src.training.pareto_trainer \
  --config           "${CONFIG}" \
  --lambda-rp-range  1e-3 1e0 \
  --lambda-rp-steps  5 \
  --lambda-c-range   1e-3 1e0 \
  --lambda-c-steps   5 \
  --seed             "${SEED}" \
  --device           "${DEVICE}" \
  --output           "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/pareto_search.log"

echo ""
echo "Pareto-front search complete."
echo "  Results     → ${OUTPUT_DIR}/pareto_results.csv"
echo "  Best model  → ${OUTPUT_DIR}/best_model.pt"
echo "  Pareto plot → ${OUTPUT_DIR}/pareto_front.png"
