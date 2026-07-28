#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-qwen3-32b-lora}"
OUTPUT_ROOT="${2:-/workspace/timing/${SCENARIO}}"
THEORETICAL_TFLOPS_PER_DEVICE="${THEORETICAL_TFLOPS_PER_DEVICE:-232.6528}"

case "${SCENARIO}" in
  qwen3-32b-lora)
    MODEL=/models/Qwen3-32B
    MODEL_SLUG=Qwen3-32B-LoRA
    TRAIN_MODE=lora
    MICRO_BATCH="${MICRO_BATCH:-8}"
    ACTIVE_PARAMETERS_B=32.8
    DEEPSPEED_CONFIG=none
    ;;
  qwen3-30b-a3b-lora)
    MODEL=/models/Qwen3-30B-A3B
    MODEL_SLUG=Qwen3-30B-A3B-LoRA
    TRAIN_MODE=lora
    MICRO_BATCH="${MICRO_BATCH:-8}"
    ACTIVE_PARAMETERS_B=3.3
    DEEPSPEED_CONFIG=none
    ;;
  qwen3-32b-full)
    MODEL=/models/Qwen3-32B
    MODEL_SLUG=Qwen3-32B-full
    TRAIN_MODE=full
    MICRO_BATCH="${MICRO_BATCH:-2}"
    ACTIVE_PARAMETERS_B=32.8
    DEEPSPEED_CONFIG=configs/zero3_bf16.json
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}" >&2
    echo "Available: qwen3-32b-lora, qwen3-30b-a3b-lora, qwen3-32b-full" >&2
    exit 2
    ;;
esac

echo "8-GPU shortcut: ${SCENARIO} -> run_scaling.sh" >&2
exec env \
  GPU_COUNTS=8 \
  MODELS="${MODEL}" \
  MODEL_SLUG="${MODEL_SLUG}" \
  TRAIN_MODE="${TRAIN_MODE}" \
  MICRO_BATCH="${MICRO_BATCH}" \
  ACTIVE_PARAMETERS_B="${ACTIVE_PARAMETERS_B}" \
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG}" \
  THEORETICAL_TFLOPS_PER_DEVICE="${THEORETICAL_TFLOPS_PER_DEVICE}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash /workspace/run_scaling.sh
