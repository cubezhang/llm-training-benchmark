#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Fixed effective batch per accelerator. Therefore global batch is 32/64/128/256.
LOCAL_BATCH="${LOCAL_BATCH:-32}"
MICRO_BATCH="${MICRO_BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MEASURE_WINDOW="${MEASURE_WINDOW:-400}"
TRAIN_MODE="${TRAIN_MODE:-lora}"
PEAK_TFLOPS="${PEAK_TFLOPS:-}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
MODELS="${MODELS:-Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-none}"

if (( MAX_STEPS < 1 || MAX_STEPS > 2000 )); then
  echo "MAX_STEPS must be between 1 and 2000" >&2
  exit 2
fi
if (( LOCAL_BATCH % MICRO_BATCH != 0 )); then
  echo "LOCAL_BATCH must be divisible by MICRO_BATCH" >&2
  exit 2
fi

for model in ${MODELS}; do
  slug="$(basename "${model}")"
  active_params="32.8"
  if [[ "${slug}" == "Qwen3-30B-A3B" ]]; then active_params="3.3"; fi
  for nproc in ${GPU_COUNTS}; do
    out="runs/${TRAIN_MODE}/${slug}/${nproc}gpu"
    extra=()
    if [[ -n "${PEAK_TFLOPS}" ]]; then
      extra+=(--theoretical-tflops-per-device "${PEAK_TFLOPS}")
    fi
    ds="${DEEPSPEED_CONFIG}"
    if [[ "${TRAIN_MODE}" == "full" && "${ds}" == "none" ]]; then
      ds="configs/zero3_bf16.json"
    fi
    echo "Running ${model}: ${nproc} devices, global batch $((LOCAL_BATCH * nproc))"
    torchrun --standalone --nproc_per_node="${nproc}" train_qwen.py \
      --model "${model}" \
      --output-dir "${out}" \
      --seq-len "${SEQ_LEN}" \
      --micro-batch "${MICRO_BATCH}" \
      --local-batch "${LOCAL_BATCH}" \
      --max-steps "${MAX_STEPS}" \
      --warmup-steps "${WARMUP_STEPS}" \
      --measure-window "${MEASURE_WINDOW}" \
      --train-mode "${TRAIN_MODE}" \
      --deepspeed "${ds}" \
      --active-parameters-billion "${active_params}" \
      "${extra[@]}"
  done
done

python summarize.py --runs-dir "runs/${TRAIN_MODE}" \
  --output "runs/${TRAIN_MODE}/summary.csv"
