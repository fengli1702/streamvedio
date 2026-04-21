#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- User-tunable knobs (export VAR=... to override) ----
RUNNER="${RUNNER:-python}"
RUN_TAG="${RUN_TAG:-1.22}"
RUN_PREFIX="${RUN_PREFIX:-doh_cod}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/streaming-lvm-dataset/DOH}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/lvm-llama2-7b}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/inference_logs}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_FRAMES="${MAX_FRAMES:-1600}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_LORAS="${MAX_LORAS:-1}"

# Workloads encoded as "context:inference", comma-separated.
WORKLOADS_CSV="${WORKLOADS_CSV:-4:4}"

# Infer-only sweep lists
IB_LIST_CSV="${IB_LIST_CSV:-2,4,8,16,32}"
TB_LIST_CSV="${TB_LIST_CSV:-8}"
LR="${LR:-2e-5}"
SPEC_LIST_CSV="${SPEC_LIST_CSV:-off,on}"

# Spec decoding config (required when spec=on).
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-}"
DEFAULT_VOCAB_MAPPING=""
if [[ -d "${REPO_ROOT}/vocab_mapping" ]]; then
  DEFAULT_VOCAB_MAPPING="$(ls "${REPO_ROOT}/vocab_mapping"/*.pt 2>/dev/null | head -n1 || true)"
fi
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-${DEFAULT_VOCAB_MAPPING}}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
# ---- End knobs ----

read -r -a RUNNER_CMD <<< "${RUNNER}"
IFS=',' read -r -a WORKLOADS <<< "${WORKLOADS_CSV}"
IFS=',' read -r -a IB_LIST <<< "${IB_LIST_CSV}"
IFS=',' read -r -a TB_LIST <<< "${TB_LIST_CSV}"
IFS=',' read -r -a SPEC_LIST <<< "${SPEC_LIST_CSV}"

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
  local inf="$2"
  local ib="$3"
  local tb="$4"
  local spec="$5"

  local run_name="${RUN_PREFIX}_${RUN_TAG}_infer_only_c${ctx}_i${inf}_ib${ib}_tb${tb}_lr${LR}_spec${spec}"
  local log_path="${LOG_DIR}/${run_name}.jsonl"

  local cmd=(
    "${RUNNER_CMD[@]}" "${REPO_ROOT}/tream.py"
    --input_frames_path "${DATASET_DIR}"
    --context_length "${ctx}"
    --inference_length "${inf}"
    --inference_batch_size "${ib}"
    --training_batch_size "${tb}"
    --learning_rate "${LR}"
    --model_name "${MODEL_PATH}"
    --num_workers "${NUM_WORKERS}"
    --max_frames "${MAX_FRAMES}"
    --gpu_memory_utilization "${GPU_MEM_UTIL}"
    --max_loras "${MAX_LORAS}"
    --wandb_run_name "${run_name}"
    --disable_dynamic_scheduling
    -gc
  )

  if [[ "${spec}" == "on" ]]; then
    if [[ -z "${SPEC_DRAFT_MODEL}" ]]; then
      echo "[ERROR] SPEC_DRAFT_MODEL is required when spec=on." >&2
      exit 1
    fi
    if [[ -z "${SPEC_VOCAB_MAPPING_PATH}" ]]; then
      echo "[ERROR] SPEC_VOCAB_MAPPING_PATH is required when spec=on." >&2
      exit 1
    fi
    cmd+=(
      --use_speculative_decoding
      --spec_draft_model "${SPEC_DRAFT_MODEL}"
      --spec_vocab_mapping_path "${SPEC_VOCAB_MAPPING_PATH}"
      --num_spec_tokens "${SPEC_NUM_TOKENS}"
      --spec_disable_mqa_scorer
    )
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

  if [[ "${spec}" == "on" ]]; then
    VLLM_MAX_SPEC_TOKENS="${SPEC_NUM_TOKENS}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

for workload in "${WORKLOADS[@]}"; do
  IFS=':' read -r ctx inf <<< "${workload}"
  if [[ -z "${ctx}" || -z "${inf}" ]]; then
    echo "[ERROR] Invalid WORKLOADS_CSV entry: ${workload}" >&2
    exit 1
  fi

  for tb in "${TB_LIST[@]}"; do
    for ib in "${IB_LIST[@]}"; do
      for spec in "${SPEC_LIST[@]}"; do
        run_one "${ctx}" "${inf}" "${ib}" "${tb}" "${spec}"
      done
    done
  done
 done
