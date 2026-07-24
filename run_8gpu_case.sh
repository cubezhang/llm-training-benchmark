#!/usr/bin/env bash
set -euo pipefail

if (( $# != 8 )); then
  echo "Usage: $0 LABEL MODEL OUTPUT_DIR TRAIN_MODE MICRO_BATCH DEEPSPEED ACTIVE_PARAMS_B PORT" >&2
  exit 2
fi

echo "run_8gpu_case.sh is kept for compatibility; new commands should use run_scaling.sh." >&2
exec bash /workspace/run_case.sh "$1" "$2" "$3" "$4" 8 "$5" "$6" "$7" "$8"
