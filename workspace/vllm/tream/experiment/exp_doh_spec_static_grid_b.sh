#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${RUNNER:-python}"
RUN_TAG="${RUN_TAG:-2.05}"
RUN_PREFIX="${RUN_PREFIX:-doh_spec_static_b}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/streaming-lvm-dataset/DOH}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/lvm-llama2-7b}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/inference_logs}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_FRAMES="${MAX_FRAMES:-4000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_LORAS="${MAX_LORAS:-1}"

RAY_PORT="${RAY_PORT:-6380}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_tream_spec_b}"
RAY_NAMESPACE="${RAY_NAMESPACE:-}"

SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475}"
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
SPEC_DISABLE_MQA_SCORER="${SPEC_DISABLE_MQA_SCORER:-1}"

LENS=(1 2 4 6 8)
IB_LIST=(8 16 32)
TB_LIST=(8 16 32)

read -r -a RUNNER_CMD <<< "${RUNNER}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERROR] Dataset dir not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -z "${SPEC_DRAFT_MODEL}" ]]; then
  echo "[ERROR] SPEC_DRAFT_MODEL is required (spec decoding enabled)." >&2
  exit 1
fi
if [[ -z "${SPEC_VOCAB_MAPPING_PATH}" ]]; then
  echo "[ERROR] SPEC_VOCAB_MAPPING_PATH is required (spec decoding enabled)." >&2
  exit 1
fi

run_one() {
  local ctx="$1"
  local inf="$2"
  local ib="$3"
  local tb="$4"

  local run_name="${RUN_PREFIX}_${RUN_TAG}_c${ctx}_i${inf}_ib${ib}_tb${tb}"
  local log_path="${LOG_DIR}/${run_name}.jsonl"

  local cmd=(
    "${RUNNER_CMD[@]}" "${REPO_ROOT}/tream.py"
    --input_frames_path "${DATASET_DIR}"
    --context_length "${ctx}"
    --inference_length "${inf}"
    --inference_batch_size "${ib}"
    --training_batch_size "${tb}"
    --model_name "${MODEL_PATH}"
    --num_workers "${NUM_WORKERS}"
    --max_frames "${MAX_FRAMES}"
    --gpu_memory_utilization "${GPU_MEM_UTIL}"
    --max_loras "${MAX_LORAS}"
    --wandb_run_name "${run_name}"
    --inference_logs_dir "${LOG_DIR}"
    --disable_dynamic_scheduling
    --use_speculative_decoding
    --spec_draft_model "${SPEC_DRAFT_MODEL}"
    --spec_vocab_mapping_path "${SPEC_VOCAB_MAPPING_PATH}"
    --num_spec_tokens "${SPEC_NUM_TOKENS}"
    -gc
    -train
  )

  if [[ "${SPEC_DISABLE_MQA_SCORER}" == "1" ]]; then
    cmd+=(--spec_disable_mqa_scorer)
  fi
  if [[ -n "${RAY_PORT}" && "${RAY_PORT}" != "0" ]]; then
    cmd+=(--ray_port "${RAY_PORT}")
  fi
  if [[ -n "${RAY_TEMP_DIR}" ]]; then
    cmd+=(--ray_temp_dir "${RAY_TEMP_DIR}")
  fi
  if [[ -n "${RAY_NAMESPACE}" ]]; then
    cmd+=(--ray_namespace "${RAY_NAMESPACE}")
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

  if [[ -n "${RAY_PORT}" && "${RAY_PORT}" != "0" ]]; then
    RAY_PORT="${RAY_PORT}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

count=0
for ctx in "${LENS[@]}"; do
  for inf in "${LENS[@]}"; do
    if (( ctx + inf <= 8 )); then
      continue
    fi
    if (( ctx + inf > 12 )); then
      continue
    fi
    for ib in "${IB_LIST[@]}"; do
      for tb in "${TB_LIST[@]}"; do
        count=$((count + 1))
      done
    done
  done
done

echo "[INFO] Total runs in this script: ${count}"

for ctx in "${LENS[@]}"; do
  for inf in "${LENS[@]}"; do
    if (( ctx + inf <= 8 )); then
      continue
    fi
    if (( ctx + inf > 12 )); then
      continue
    fi
    for ib in "${IB_LIST[@]}"; do
      for tb in "${TB_LIST[@]}"; do
        run_one "${ctx}" "${inf}" "${ib}" "${tb}"
      done
    done
  done
done
