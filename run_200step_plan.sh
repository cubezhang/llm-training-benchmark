#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/hf-cache/datasets}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

run_case() {
  local label="$1"
  local devices="$2"
  local nproc="$3"
  local mode="$4"
  local micro_batch="$5"
  local deepspeed="$6"
  local port="$7"
  local output_dir="$8"
  local samples="${output_dir}/gpu_samples.csv"

  printf 'RUNNING %s started=%s\n' "${label}" "$(date -Is)" > "${STATUS}"
  printf '\n===== START %s %s =====\n' "${label}" "$(date -Is)" >> "${LOG}"
  local cmd=(
    torchrun
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --master_port="${port}"
    --nproc_per_node="${nproc}"
    train_qwen.py
    --model /models/Qwen3-32B
    --output-dir "${output_dir}"
    --seq-len 2048
    --micro-batch "${micro_batch}"
    --local-batch 32
    --max-steps 200
    --warmup-steps 20
    --measure-window 100
    --train-mode "${mode}"
    --deepspeed "${deepspeed}"
    --active-parameters-billion 32.8
  )
  mkdir -p "${output_dir}"
  HIP_VISIBLE_DEVICES="${devices}" "${cmd[@]}" >> "${LOG}" 2>&1 &
  local train_pid=$!
  (
    printf 'timestamp,device,gpu_use_percent,vram_percent,gpu_memory_activity_percent,memory_activity\n' > "${samples}"
    while kill -0 "${train_pid}" 2>/dev/null; do
      local timestamp
      timestamp="$(date +%s)"
      /opt/rocm/bin/rocm-smi --showuse --showmemuse --csv |
        tail -n +2 |
        awk -v timestamp="${timestamp}" '{print timestamp "," $0}' >> "${samples}"
      sleep 10
    done
  ) &
  local sampler_pid=$!

  set +e
  wait "${train_pid}"
  local train_rc=$?
  set -e
  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
  if (( train_rc != 0 )); then
    return "${train_rc}"
  fi
  python merge_gpu_samples.py --metrics "${output_dir}/metrics.json" --samples "${samples}" --devices "${devices}" >> "${LOG}" 2>&1
  printf '===== DONE %s %s =====\n' "${label}" "$(date -Is)" >> "${LOG}"
}

SCENE1="${ROOT}/scenario1"
run_case "scenario1-lora-1gpu" "0" 1 lora 8 none 29801 "${SCENE1}/Qwen3-32B/1gpu"
run_case "scenario1-lora-2gpu" "0,1" 2 lora 8 none 29802 "${SCENE1}/Qwen3-32B/2gpu"
run_case "scenario1-lora-4gpu" "0,1,2,3" 4 lora 8 none 29804 "${SCENE1}/Qwen3-32B/4gpu"
run_case "scenario1-lora-8gpu" "0,1,2,3,4,5,6,7" 8 lora 8 none 29808 "${SCENE1}/Qwen3-32B/8gpu"

python summarize.py --runs-dir "${SCENE1}" --output "${SCENE1}/summary.csv" --table-output "${SCENE1}/summary_table.csv" --model-name Qwen3-32B-LoRA --theoretical-tflops-per-device 232.6528 >> "${LOG}" 2>&1

SCENE3="${ROOT}/scenario3"
run_case "scenario3-full-4gpu" "0,1,2,3" 4 full 2 configs/zero3_bf16.json 29834 "${SCENE3}/Qwen3-32B-full/4gpu"
run_case "scenario3-full-8gpu" "0,1,2,3,4,5,6,7" 8 full 2 configs/zero3_bf16.json 29838 "${SCENE3}/Qwen3-32B-full/8gpu"

python summarize.py --runs-dir "${SCENE3}" --output "${SCENE3}/summary.csv" --table-output "${SCENE3}/summary_table.csv" --model-name Qwen3-32B-full --theoretical-tflops-per-device 232.6528 >> "${LOG}" 2>&1

printf 'COMPLETE finished=%s\n' "$(date -Is)" > "${STATUS}"
printf '\n===== PLAN COMPLETE %s =====\n' "$(date -Is)" >> "${LOG}"
