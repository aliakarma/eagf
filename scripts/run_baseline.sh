#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/biometric_default.yaml"
SEEDS="42"
DEVICE="cpu"
OUTPUT="results/baseline"
DEMO_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --demo) DEMO_FLAG="--demo"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT"

for SEED in $SEEDS; do
  echo "[Baseline] seed=${SEED}"
  python -m src.baselines.aif360_dp_pipeline \
    --config "$CONFIG" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --output "$OUTPUT/seed_${SEED}" \
    $DEMO_FLAG
done

echo "Baseline runs complete. Outputs in $OUTPUT"
