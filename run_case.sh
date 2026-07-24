#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 9 )); then
  echo "Usage: $0 LABEL MODEL OUTPUT_DIR TRAIN_MODE GPU_COUNT MICRO_BATCH DEEPSPEED ACTIVE_PARAMS_B PORT" >&2
  exit 2
fi

LABEL="$1"
MODEL="$2"
OUTPUT_DIR="$3"
TRAIN_MODE="$4"
GPU_COUNT="$5"
MICRO_BATCH="$6"
DEEPSPEED="$7"
ACTIVE_PARAMETERS="$8"
PORT="$9"

cd /workspace

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MEASURE_WINDOW="${MEASURE_WINDOW:-$((MAX_STEPS - WARMUP_STEPS))}"
SEQ_LEN="${SEQ_LEN:-2048}"
LOCAL_BATCH="${LOCAL_BATCH:-32}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-1}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"
THEORETICAL_TFLOPS_PER_DEVICE="${THEORETICAL_TFLOPS_PER_DEVICE:-${PEAK_TFLOPS:-}}"
DATASET_DIR="${DATASET_DIR:-}"
if [[ -z "${DATASET_DIR}" && -d /workspace/datasets/wikitext-103-raw-v1 ]]; then
  DATASET_DIR=/workspace/datasets/wikitext-103-raw-v1
fi

if (( GPU_COUNT < 1 )); then
  echo "GPU_COUNT must be at least 1" >&2
  exit 2
fi
if (( MAX_STEPS < 1 || MAX_STEPS > 2000 )); then
  echo "MAX_STEPS must be between 1 and 2000" >&2
  exit 2
fi
if (( LOCAL_BATCH % MICRO_BATCH != 0 )); then
  echo "LOCAL_BATCH must be divisible by MICRO_BATCH" >&2
  exit 2
fi

if [[ -n "${GPU_DEVICES:-}" ]]; then
  DEVICES="${GPU_DEVICES}"
else
  DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
fi

LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/console.log}"
STATUS_FILE="${STATUS_FILE:-${OUTPUT_DIR}/status.txt}"
SAMPLES="${OUTPUT_DIR}/gpu_samples.csv"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${STATUS_FILE}")"
touch "${LOG_FILE}"
printf 'RUNNING %s gpus=%s devices=%s started=%s\n' \
  "${LABEL}" "${GPU_COUNT}" "${DEVICES}" "$(date -Is)" > "${STATUS_FILE}"
printf '\n===== START %s gpus=%s devices=%s %s =====\n' \
  "${LABEL}" "${GPU_COUNT}" "${DEVICES}" "$(date -Is)" >> "${LOG_FILE}"

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
  --nproc_per_node="${GPU_COUNT}"
  train_qwen.py
  --model "${MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --seq-len "${SEQ_LEN}"
  --micro-batch "${MICRO_BATCH}"
  --local-batch "${LOCAL_BATCH}"
  --max-steps "${MAX_STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --measure-window "${MEASURE_WINDOW}"
  --train-mode "${TRAIN_MODE}"
  --deepspeed "${DEEPSPEED}"
  --active-parameters-billion "${ACTIVE_PARAMETERS}"
)

if [[ -n "${THEORETICAL_TFLOPS_PER_DEVICE}" ]]; then
  cmd+=(--theoretical-tflops-per-device "${THEORETICAL_TFLOPS_PER_DEVICE}")
fi
if [[ -n "${DATASET_DIR}" ]]; then
  cmd+=(--dataset-dir "${DATASET_DIR}")
fi
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
    sleep "${GPU_SAMPLE_INTERVAL}"
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
