#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export MAX_STEPS=2000
export WARMUP_STEPS=100
export MEASURE_WINDOW=1900
export SEQ_LEN=2048
export LOCAL_BATCH=32
export SAVE_FINAL_MODEL=1
export GPU_COUNTS=8
export SKIP_COMPLETED=1
export THEORETICAL_TFLOPS_PER_DEVICE="${THEORETICAL_TFLOPS_PER_DEVICE:-232.6528}"

ROOT="${ROOT:-/workspace/timing/qwen-three-scenarios-8gpu-2000steps}"
LOG="${ROOT}/console.log"
STATUS="${ROOT}/status.txt"
mkdir -p "${ROOT}"
touch "${LOG}"

on_error() {
  rc=$?
  printf 'FAILED rc=%s at %s\n' "${rc}" "$(date -Is)" > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

run_scenario() {
  local label="$1"
  shift
  printf 'RUNNING %s started=%s\n' "${label}" "$(date -Is)" > "${STATUS}"
  printf '\n===== START %s %s =====\n' "${label}" "$(date -Is)" >> "${LOG}"
  env "$@" bash /workspace/run_scaling.sh >> "${LOG}" 2>&1
}

run_scenario scenario1-qwen3-32b-lora-8gpu \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-LoRA \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  OUTPUT_ROOT="${ROOT}/scenario1" \
  PORT_BASE=30010

run_scenario scenario2-qwen3-30b-a3b-lora-8gpu \
  MODELS=/models/Qwen3-30B-A3B \
  MODEL_SLUG=Qwen3-30B-A3B-LoRA \
  ACTIVE_PARAMETERS_B=3.3 \
  TRAIN_MODE=lora \
  MICRO_BATCH=8 \
  OUTPUT_ROOT="${ROOT}/scenario2" \
  PORT_BASE=30020

run_scenario scenario3-qwen3-32b-full-8gpu \
  MODELS=/models/Qwen3-32B \
  MODEL_SLUG=Qwen3-32B-full \
  ACTIVE_PARAMETERS_B=32.8 \
  TRAIN_MODE=full \
  MICRO_BATCH=2 \
  DEEPSPEED_CONFIG=configs/zero3_bf16.json \
  OUTPUT_ROOT="${ROOT}/scenario3" \
  PORT_BASE=30030

python summarize.py \
  --runs-dir "${ROOT}" \
  --output "${ROOT}/summary.csv" \
  --table-output "${ROOT}/summary_table.csv" \
  --theoretical-tflops-per-device "${THEORETICAL_TFLOPS_PER_DEVICE}" >> "${LOG}" 2>&1

printf 'COMPLETE all-scenarios finished=%s\n' "$(date -Is)" > "${STATUS}"
printf '\n===== ALL COMPLETE %s =====\n' "$(date -Is)" >> "${LOG}"
chmod -R a+rX "${ROOT}"
