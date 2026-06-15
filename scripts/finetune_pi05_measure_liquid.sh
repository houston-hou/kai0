#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-pi05_aloha_measure_liquid_full}"
EXP_NAME="${EXP_NAME:-smoke_measure_liquid}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_STEPS}"
MAX_FRAMES="${MAX_FRAMES:-128}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/weights_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/hdy/emchem_pi05/training_data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

NORM_ARGS=(--config-name "$CONFIG_NAME")
if [[ -n "$MAX_FRAMES" ]]; then
  NORM_ARGS+=(--max-frames "$MAX_FRAMES")
fi

echo "Computing norm stats for $CONFIG_NAME ..."
uv run python scripts/compute_norm_stats.py "${NORM_ARGS[@]}"

echo "Running finetune smoke test: $TRAIN_STEPS step(s) ..."
uv run python scripts/train.py "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  --overwrite \
  --num-train-steps="$TRAIN_STEPS" \
  --save-interval="$SAVE_INTERVAL" \
  --log-interval=1

CKPT_DIR="$PROJECT_ROOT/checkpoints/$CONFIG_NAME/$EXP_NAME"
LATEST_STEP="$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+$/ {print}' | sort -n | tail -1)"
if [[ -z "$LATEST_STEP" ]]; then
  echo "No numeric checkpoint step found under $CKPT_DIR" >&2
  exit 1
fi

EXPORT_DIR="$PROJECT_ROOT/exports/$CONFIG_NAME/$EXP_NAME/step_$LATEST_STEP"
mkdir -p "$EXPORT_DIR"
cp -a "$CKPT_DIR/$LATEST_STEP/." "$EXPORT_DIR/"

echo "Exported checkpoint result to: $EXPORT_DIR"
