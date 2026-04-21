#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRIPT_A="${SCRIPT_A:-${SCRIPT_DIR}/exp_doh_spec_static_grid_a.sh}"
SCRIPT_B="${SCRIPT_B:-${SCRIPT_DIR}/exp_doh_spec_static_grid_b.sh}"

GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"

RAY_PORT_A="${RAY_PORT_A:-6379}"
RAY_PORT_B="${RAY_PORT_B:-6380}"

RAY_TEMP_DIR_A="${RAY_TEMP_DIR_A:-/tmp/ray_tream_spec_a}"
RAY_TEMP_DIR_B="${RAY_TEMP_DIR_B:-/tmp/ray_tream_spec_b}"

RUNNER="${RUNNER:-python}"
MAX_FRAMES="${MAX_FRAMES:-4000}"

LOG_A="${LOG_A:-${REPO_ROOT}/inference_logs/doh_spec_static_a.log}"
LOG_B="${LOG_B:-${REPO_ROOT}/inference_logs/doh_spec_static_b.log}"

if [[ ! -x "${SCRIPT_A}" ]]; then
  echo "[ERROR] Script A not found or not executable: ${SCRIPT_A}" >&2
  exit 1
fi
if [[ ! -x "${SCRIPT_B}" ]]; then
  echo "[ERROR] Script B not found or not executable: ${SCRIPT_B}" >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/inference_logs"

echo "[INFO] Starting Script A on GPU ${GPU_A} (ray port ${RAY_PORT_A})"
CUDA_VISIBLE_DEVICES="${GPU_A}" \
RAY_PORT="${RAY_PORT_A}" \
RAY_TEMP_DIR="${RAY_TEMP_DIR_A}" \
RUNNER="${RUNNER}" \
MAX_FRAMES="${MAX_FRAMES}" \
"${SCRIPT_A}" \
> "${LOG_A}" 2>&1 &
PID_A=$!

echo "[INFO] Starting Script B on GPU ${GPU_B} (ray port ${RAY_PORT_B})"
CUDA_VISIBLE_DEVICES="${GPU_B}" \
RAY_PORT="${RAY_PORT_B}" \
RAY_TEMP_DIR="${RAY_TEMP_DIR_B}" \
RUNNER="${RUNNER}" \
MAX_FRAMES="${MAX_FRAMES}" \
"${SCRIPT_B}" \
> "${LOG_B}" 2>&1 &
PID_B=$!

echo "[INFO] PIDs: A=${PID_A}, B=${PID_B}"

wait "${PID_A}"
RC_A=$?
wait "${PID_B}"
RC_B=$?

echo "[INFO] Done. exit codes: A=${RC_A}, B=${RC_B}"
exit $((RC_A != 0 || RC_B != 0))
