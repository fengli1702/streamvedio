#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${ROOT}/workspace/vllm"
TREAM_ROOT="${REPO}/tream"

ENV_METHOD="${ENV_METHOD:-conda}"     # conda | venv
ENV_NAME="${ENV_NAME:-tream}"
DATASET_SUBDIR="${DATASET_SUBDIR:-Ego4d}"

bash "${ROOT}/tools/no_docker/setup.sh" \
  --method "${ENV_METHOD}" \
  --env-name "${ENV_NAME}" \
  --repo-root "${REPO}"

if [[ "${ENV_METHOD}" == "conda" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
else
  source "${HOME}/.venvs/${ENV_NAME}/bin/activate"
fi

TREAM_ROOT="${TREAM_ROOT}" \
RUN_MODE=static \
RUN_SPEC=1 \
DATASET_SUBDIR="${DATASET_SUBDIR}" \
GPU_LIST=0,1,2,3,4,5,6,7 \
CONCURRENCY=8 \
bash "${ROOT}/tools/launch_ab48_8gpu.sh"
