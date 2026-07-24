#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MAX_STEPS=200
export WARMUP_STEPS=20
export MEASURE_WINDOW=100
export LOCAL_BATCH=32
export THEORETICAL_TFLOPS_PER_DEVICE="${THEORETICAL_TFLOPS_PER_DEVICE:-232.6528}"

ROOT="${ROOT:-/workspace/timing/qwen-scenario1-scenario3-200steps}"
LOG="${ROOT}/console.log"
STATUS="${ROOT}/status.txt"
mkdir -p "${ROOT}"
: > "${LOG}"

on_error() {
  rc=$?
  printf 'FAILED rc=%s at %s\n' "${rc}" "$(date -Is)" > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

printf 'RUNNING scenario1 started=%s\n' "$(date -Is)" > "${STATUS}"
env \
  GPU_COUNTS="1 2 4 8" \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  OUTPUT_ROOT="${ROOT}/scenario1" \
  PORT_BASE=29800 \
  bash /workspace/run_scaling.sh >> "${LOG}" 2>&1

printf 'RUNNING scenario3 started=%s\n' "$(date -Is)" > "${STATUS}"
env \
  GPU_COUNTS="4 8" \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-full \
  TRAIN_MODE=full \
  MICRO_BATCH=2 \
  DEEPSPEED_CONFIG=configs/zero3_bf16.json \
  OUTPUT_ROOT="${ROOT}/scenario3" \
  PORT_BASE=29830 \
  bash /workspace/run_scaling.sh >> "${LOG}" 2>&1

printf 'COMPLETE finished=%s\n' "$(date -Is)" > "${STATUS}"
printf '\n===== PLAN COMPLETE %s =====\n' "$(date -Is)" >> "${LOG}"
