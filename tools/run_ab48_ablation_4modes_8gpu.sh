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

run_mode() {
  local mode="$1"
  echo "[ablation] START ${mode}"
  case "${mode}" in
    no_feature)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=static RUN_SPEC=0 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    spec_only)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=static RUN_SPEC=1 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    dyn_nojsd)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=dynamic RUN_SPEC=1 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      SHIFT_SHOCK_DISABLE_DRIFT=1 \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    dyn_jsd)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=dynamic RUN_SPEC=1 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0 \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    *)
      echo "unknown mode ${mode}" >&2; exit 2 ;;
  esac
  echo "[ablation] DONE ${mode}"
}

run_mode no_feature
run_mode spec_only
run_mode dyn_nojsd
run_mode dyn_jsd
