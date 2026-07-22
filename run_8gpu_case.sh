#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 8 )); then
  echo "Usage: $0 LABEL MODEL OUTPUT_DIR TRAIN_MODE MICRO_BATCH DEEPSPEED ACTIVE_PARAMS_B PORT" >&2
  exit 2
fi

LABEL="$1"
MODEL="$2"
OUTPUT_DIR="$3"
TRAIN_MODE="$4"
MICRO_BATCH="$5"
DEEPSPEED="$6"
ACTIVE_PARAMETERS="$7"
PORT="$8"

cd /workspace

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/hf-cache/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MEASURE_WINDOW="${MEASURE_WINDOW:-$((MAX_STEPS - WARMUP_STEPS))}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-1}"
DEVICES="0,1,2,3,4,5,6,7"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/console.log}"
STATUS_FILE="${STATUS_FILE:-${OUTPUT_DIR}/status.txt}"
SAMPLES="${OUTPUT_DIR}/gpu_samples.csv"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${STATUS_FILE}")"
touch "${LOG_FILE}"
printf 'RUNNING %s started=%s\n' "${LABEL}" "$(date -Is)" > "${STATUS_FILE}"
printf '\n===== START %s %s =====\n' "${LABEL}" "$(date -Is)" >> "${LOG_FILE}"

on_error() {
  rc=$?
  printf 'FAILED %s rc=%s at %s\n' "${LABEL}" "${rc}" "$(date -Is)" > "${STATUS_FILE}"
  exit "${rc}"
}
trap on_error ERR

cmd=(
  torchrun
  --nnodes=1
  --node_rank=0
  --master_addr=127.0.0.1
  --master_port="${PORT}"
  --nproc_per_node=8
  train_qwen.py
  --model "${MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --seq-len 2048
  --micro-batch "${MICRO_BATCH}"
  --local-batch 32
  --max-steps "${MAX_STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --measure-window "${MEASURE_WINDOW}"
  --train-mode "${TRAIN_MODE}"
  --deepspeed "${DEEPSPEED}"
  --active-parameters-billion "${ACTIVE_PARAMETERS}"
)

if [[ "${SAVE_FINAL_MODEL}" == "0" ]]; then
  cmd+=(--no-save-final-model)
else
  cmd+=(--save-final-model)
fi

HIP_VISIBLE_DEVICES="${DEVICES}" "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
train_pid=$!

(
  printf 'timestamp,device,gpu_use_percent,vram_percent,gpu_memory_activity_percent,memory_activity\n' > "${SAMPLES}"
  while kill -0 "${train_pid}" 2>/dev/null; do
    timestamp="$(date +%s)"
    /opt/rocm/bin/rocm-smi --showuse --showmemuse --csv |
      tail -n +2 |
      awk -v timestamp="${timestamp}" 'NF {print timestamp "," $0}' >> "${SAMPLES}"
    sleep 10
  done
) &
sampler_pid=$!

set +e
wait "${train_pid}"
train_rc=$?
set -e
kill "${sampler_pid}" 2>/dev/null || true
wait "${sampler_pid}" 2>/dev/null || true

if (( train_rc != 0 )); then
  printf 'FAILED %s rc=%s at %s\n' "${LABEL}" "${train_rc}" "$(date -Is)" > "${STATUS_FILE}"
  exit "${train_rc}"
fi

python merge_gpu_samples.py \
  --metrics "${OUTPUT_DIR}/metrics.json" \
  --samples "${SAMPLES}" \
  --devices "${DEVICES}" >> "${LOG_FILE}" 2>&1

printf 'COMPLETE %s finished=%s\n' "${LABEL}" "$(date -Is)" > "${STATUS_FILE}"
printf '===== DONE %s %s =====\n' "${LABEL}" "$(date -Is)" >> "${LOG_FILE}"
