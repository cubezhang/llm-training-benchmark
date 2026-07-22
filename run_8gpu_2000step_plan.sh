#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export MAX_STEPS=2000
export WARMUP_STEPS=100
export MEASURE_WINDOW=1900

ROOT="${ROOT:-/workspace/timing/qwen-three-scenarios-8gpu-2000steps}"
export LOG_FILE="${ROOT}/console.log"
export STATUS_FILE="${ROOT}/status.txt"
mkdir -p "${ROOT}"

run_if_needed() {
  output_dir="$3"
  if [[ -f "${output_dir}/metrics.json" ]] && \
     grep -q '"final_step": 2000' "${output_dir}/metrics.json"; then
    printf 'SKIP completed %s\n' "$1" >> "${LOG_FILE}"
    return
  fi
  bash /workspace/run_8gpu_case.sh "$@"
}

run_if_needed \
  scenario1-qwen3-32b-lora-8gpu \
  /models/Qwen3-32B \
  "${ROOT}/scenario1/Qwen3-32B-LoRA/8gpu" \
  lora 8 none 32.8 30018

run_if_needed \
  scenario2-qwen3-30b-a3b-lora-8gpu \
  /models/Qwen3-30B-A3B \
  "${ROOT}/scenario2/Qwen3-30B-A3B-LoRA/8gpu" \
  lora 8 none 3.3 30028

run_if_needed \
  scenario3-qwen3-32b-full-8gpu \
  /models/Qwen3-32B \
  "${ROOT}/scenario3/Qwen3-32B-full/8gpu" \
  full 2 configs/zero3_bf16.json 32.8 30038

python summarize.py \
  --runs-dir "${ROOT}" \
  --output "${ROOT}/summary.csv" \
  --table-output "${ROOT}/summary_table.csv" \
  --theoretical-tflops-per-device 232.6528 >> "${LOG_FILE}" 2>&1

printf 'COMPLETE all-scenarios finished=%s\n' "$(date -Is)" > "${STATUS_FILE}"
printf '\n===== ALL COMPLETE %s =====\n' "$(date -Is)" >> "${LOG_FILE}"
