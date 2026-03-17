#!/usr/bin/env bash
# ============================================================
# scripts/run_ablation.sh
# EAGF Six-Variant Ablation Study Runner
#
# Runs model variants M0 through M5 across all specified seeds
# and writes per-seed and aggregated results to the output dir.
#
# Usage:
#   bash scripts/run_ablation.sh [OPTIONS]
#
# Options:
#   --config CONFIG   Path to YAML config file
#   --seeds  SEEDS    Space-separated seeds (default: "42 123 456")
#   --device DEVICE   PyTorch device string (default: cpu)
#   --output DIR      Output directory (default: results/ablation/)
# ============================================================

set -euo pipefail

CONFIG="configs/biometric_default.yaml"
SEEDS="42 123 456"
DEVICE="cpu"
OUTPUT_DIR="results/biometric/ablation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2";     shift 2 ;;
    --seeds)  SEEDS="$2";      shift 2 ;;
    --device) DEVICE="$2";     shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

# Ablation variants: key matches --model argument in eagf_trainer.py
MODELS=("baseline" "transparency" "fairness" "privacy" "accountability" "eagf")
LABELS=("M0: Baseline" "M1: +Transparency" "M2: +Fairness" "M3: +Privacy" "M4: +Accountability" "M5: EAGF (Full)")

echo "========================================================"
echo "  EAGF Ablation Study"
echo "  Models: ${#MODELS[@]} variants × ${#SEEDS[@]} seeds"
echo "========================================================"

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  LABEL="${LABELS[$i]}"
  echo ""
  echo "── ${LABEL} ──"

  for SEED in ${SEEDS}; do
    OUT="${OUTPUT_DIR}/${MODEL}/seed_${SEED}"
    mkdir -p "${OUT}"
    echo "  Running seed=${SEED}..."
    python -m src.training.eagf_trainer \
      --config  "${CONFIG}" \
      --model   "${MODEL}" \
      --seed    "${SEED}" \
      --device  "${DEVICE}" \
      --output  "${OUT}" \
      2>&1 | tee "${OUT}/train.log"
    echo "  Done. Results → ${OUT}/"
  done
done

# Aggregate results across seeds into summary CSV
echo ""
echo "Aggregating results..."
python -m src.evaluation.ablation \
  --results-dir "${OUTPUT_DIR}" \
  --models      "${MODELS[@]}" \
  --seeds       ${SEEDS} \
  --output      "${OUTPUT_DIR}/ablation_summary.csv"

echo ""
echo "Ablation study complete. Summary → ${OUTPUT_DIR}/ablation_summary.csv"
