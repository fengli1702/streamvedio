#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to repo root to ensure relative paths work correctly
cd "${REPO_ROOT}"

# ---- User-tunable knobs (export VAR=... to override) ----
RUNNER="${RUNNER:-python}"
RUN_TAG="${RUN_TAG:-1.22}"
RUN_PREFIX="${RUN_PREFIX:-doh_ibctx_did}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/streaming-lvm-dataset/DOH}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/lvm-llama2-7b}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/inference_logs}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_FRAMES="${MAX_FRAMES:-1600}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_LORAS="${MAX_LORAS:-1}"
USE_TRAINING="${USE_TRAINING:-1}"

INF_LENGTH="${INF_LENGTH:-4}"
TB_FIXED="${TB_FIXED:-8}"
LR_FIXED="${LR_FIXED:-2e-5}"

CTX_LEVELS_CSV="${CTX_LEVELS_CSV:-2,8}"
IB_LEVELS_CSV="${IB_LEVELS_CSV:-2,16}"
REPEATS="${REPEATS:-2}"
# ---- End user-tunable knobs ----

read -r -a RUNNER_CMD <<< "${RUNNER}"
IFS=',' read -r -a CTX_LEVELS <<< "${CTX_LEVELS_CSV}"
IFS=',' read -r -a IB_LEVELS <<< "${IB_LEVELS_CSV}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERROR] Dataset dir not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

run_one() {
  local ctx="$1"
  local ib="$2"
  local rep="$3"

  local run_name="${RUN_PREFIX}_${RUN_TAG}_c${ctx}_i${INF_LENGTH}_ib${ib}_tb${TB_FIXED}_lr${LR_FIXED}_rep${rep}"
  local log_path="${LOG_DIR}/${run_name}.jsonl"
  local cmd=(
    "${RUNNER_CMD[@]}" "${REPO_ROOT}/tream.py"
    --input_frames_path "${DATASET_DIR}"
    --context_length "${ctx}"
    --inference_length "${INF_LENGTH}"
    --inference_batch_size "${ib}"
    --training_batch_size "${TB_FIXED}"
    --learning_rate "${LR_FIXED}"
    --model_name "${MODEL_PATH}"
    --num_workers "${NUM_WORKERS}"
    --max_frames "${MAX_FRAMES}"
    --gpu_memory_utilization "${GPU_MEM_UTIL}"
    --max_loras "${MAX_LORAS}"
    --wandb_run_name "${run_name}"
    --disable_dynamic_scheduling
    -gc
  )

  if [[ "${USE_TRAINING}" == "1" ]]; then
    cmd+=(-train)
  fi

  if [[ -f "${log_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "[SKIP] ${run_name} (exists: ${log_path})"
    return 0
  fi

  echo "[RUN] ${run_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  "${cmd[@]}"
}

for ctx in "${CTX_LEVELS[@]}"; do
  for ib in "${IB_LEVELS[@]}"; do
    for ((rep=1; rep<=REPEATS; rep++)); do
      run_one "${ctx}" "${ib}" "${rep}"
    done
  done
done

# ---- New data points: ctx=4 with ib=8,32 (no repeat) ----
echo "[INFO] Running new data points with ctx=4, ib=8,32"
for ib in 8 32; do
  run_one 4 "${ib}" 1
done
