#!/usr/bin/env bash
set -euo pipefail

NAME="${CONTAINER_NAME:-llm-training-rocm}"
IMAGE="${ROCM_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
PROJECT="${PROJECT_DIR:-/volumes/oss5/models/qwen-scaling}"

if ! docker inspect "${NAME}" >/dev/null 2>&1; then
  docker run -d \
    --name "${NAME}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --network=host \
    --shm-size=256g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /volumes/oss0/models:/models \
    -v "${PROJECT}":/workspace \
    -w /workspace \
    "${IMAGE}" sleep infinity
else
  docker start "${NAME}" >/dev/null
fi

docker exec "${NAME}" python3 -m pip install -r /workspace/requirements-full.txt
docker exec "${NAME}" python3 -c \
  "import torch,transformers,datasets,accelerate,deepspeed,peft; print(torch.__version__, transformers.__version__, torch.cuda.device_count())"

echo "Container ${NAME} is ready"
