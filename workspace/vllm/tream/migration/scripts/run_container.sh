#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-vllm-tream-portable:latest}"
CONTAINER="${CONTAINER:-vllm-tream-portable}"
SHM_SIZE="${SHM_SIZE:-64g}"

# Optional mounts (recommended on target machine)
HOST_DATASET_DIR="${HOST_DATASET_DIR:-}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-}"
HOST_HF_CACHE_DIR="${HOST_HF_CACHE_DIR:-}"

MOUNTS=()
if [[ -n "${HOST_DATASET_DIR}" ]]; then
  MOUNTS+=(-v "${HOST_DATASET_DIR}:/workspace/vllm/tream/data/streaming-lvm-dataset")
fi
if [[ -n "${HOST_MODEL_DIR}" ]]; then
  MOUNTS+=(-v "${HOST_MODEL_DIR}:/workspace/vllm/tream/lvm-llama2-7b")
fi
if [[ -n "${HOST_HF_CACHE_DIR}" ]]; then
  MOUNTS+=(-v "${HOST_HF_CACHE_DIR}:/workspace/vllm/hf_cache")
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "Container already exists: ${CONTAINER}"
  exit 1
fi

docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  --shm-size "${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${MOUNTS[@]}" \
  "${IMAGE}" \
  sleep infinity

echo "Started container: ${CONTAINER}"
