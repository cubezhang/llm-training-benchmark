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
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
MODELS="${MODELS:-Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-none}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/${TRAIN_MODE}}"
PORT_BASE="${PORT_BASE:-29500}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

if (( MAX_STEPS < 1 || MAX_STEPS > 2000 )); then
  echo "MAX_STEPS must be between 1 and 2000" >&2
  exit 2
fi
if (( LOCAL_BATCH % MICRO_BATCH != 0 )); then
  echo "LOCAL_BATCH must be divisible by MICRO_BATCH" >&2
  exit 2
fi

model_index=0
for model in ${MODELS}; do
  slug="${MODEL_SLUG:-$(basename "${model}")}"
  active_params="${ACTIVE_PARAMETERS_B:-32.8}"
  if [[ -z "${ACTIVE_PARAMETERS_B:-}" && "${slug}" == *"Qwen3-30B-A3B"* ]]; then
    active_params="3.3"
  fi
  for nproc in ${GPU_COUNTS}; do
    if (( nproc < 1 )); then
      echo "Every GPU_COUNTS value must be at least 1" >&2
      exit 2
    fi
    out="${OUTPUT_ROOT}/${slug}/${nproc}gpu"
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${out}/metrics.json" ]] && \
       grep -q "\"final_step\": ${MAX_STEPS}" "${out}/metrics.json"; then
      echo "Skipping completed ${model}: ${nproc} GPUs"
      continue
    fi
    ds="${DEEPSPEED_CONFIG}"
    if [[ "${TRAIN_MODE}" == "full" && "${ds}" == "none" ]]; then
      ds="configs/zero3_bf16.json"
    fi
    port="$((PORT_BASE + model_index * 100 + nproc))"
    echo "Running ${model}: ${nproc} GPUs, global batch $((LOCAL_BATCH * nproc))"
    bash /workspace/run_case.sh \
      "${slug}-${TRAIN_MODE}-${nproc}gpu" \
      "${model}" "${out}" "${TRAIN_MODE}" "${nproc}" "${MICRO_BATCH}" \
      "${ds}" "${active_params}" "${port}"
  done
  model_index="$((model_index + 1))"
done

summary_args=(
  python summarize.py
  --runs-dir "${OUTPUT_ROOT}"
  --output "${OUTPUT_ROOT}/summary.csv"
  --table-output "${OUTPUT_ROOT}/summary_table.csv"
)
if [[ -n "${THEORETICAL_TFLOPS_PER_DEVICE:-${PEAK_TFLOPS:-}}" ]]; then
  summary_args+=(--theoretical-tflops-per-device "${THEORETICAL_TFLOPS_PER_DEVICE:-${PEAK_TFLOPS}}")
fi
"${summary_args[@]}"
