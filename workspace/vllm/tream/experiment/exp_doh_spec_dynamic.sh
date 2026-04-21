#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${RUNNER:-python}"
RUN_TAG="${RUN_TAG:-2.0}"
RUN_PREFIX="${RUN_PREFIX:-doh_spec_dynamic}"
RUN_ID="${RUN_ID:-a}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/streaming-lvm-dataset/DOH}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/lvm-llama2-7b}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/inference_logs}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_FRAMES="${MAX_FRAMES:-4000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_LORAS="${MAX_LORAS:-1}"

CTX_INIT="${CTX_INIT:-4}"
INF_INIT="${INF_INIT:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-32}"

OPTIMIZATION_PRIORITY="${OPTIMIZATION_PRIORITY:-latency}"
SCHED_TARGET_LATENCY="${SCHED_TARGET_LATENCY:-2.0}"
SCHED_QUALITY_MIN="${SCHED_QUALITY_MIN:-0.3}"
SCHED_WINDOW_SIZE="${SCHED_WINDOW_SIZE:-3}"
SCHED_Q_BATCH="${SCHED_Q_BATCH:-2}"
SCHED_MIX_LAMBDA="${SCHED_MIX_LAMBDA:-0.7}"
SCHED_LATENCY_MARGIN="${SCHED_LATENCY_MARGIN:-0.1}"
SCHED_CYCLE_FRAMES="${SCHED_CYCLE_FRAMES:-20}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_tream_dyn_${RUN_ID}}"
RAY_NAMESPACE="${RAY_NAMESPACE:-}"

SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475}"
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
SPEC_DISABLE_MQA_SCORER="${SPEC_DISABLE_MQA_SCORER:-1}"

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

run_name="${RUN_PREFIX}_${RUN_TAG}_c${CTX_INIT}_i${INF_INIT}_ib${INFERENCE_BATCH_SIZE}_tb${TRAINING_BATCH_SIZE}_${RUN_ID}"
log_path="${LOG_DIR}/${run_name}.jsonl"

cmd=(
  "${RUNNER_CMD[@]}" "${REPO_ROOT}/tream.py"
  --input_frames_path "${DATASET_DIR}"
  --context_length "${CTX_INIT}"
  --inference_length "${INF_INIT}"
  --inference_batch_size "${INFERENCE_BATCH_SIZE}"
  --training_batch_size "${TRAINING_BATCH_SIZE}"
  --model_name "${MODEL_PATH}"
  --num_workers "${NUM_WORKERS}"
  --max_frames "${MAX_FRAMES}"
  --gpu_memory_utilization "${GPU_MEM_UTIL}"
  --max_loras "${MAX_LORAS}"
  --wandb_run_name "${run_name}"
  --inference_logs_dir "${LOG_DIR}"
  --optimization_priority "${OPTIMIZATION_PRIORITY}"
  --scheduler_target_latency "${SCHED_TARGET_LATENCY}"
  --scheduler_quality_min "${SCHED_QUALITY_MIN}"
  --scheduler_window_size "${SCHED_WINDOW_SIZE}"
  --scheduler_q_batch "${SCHED_Q_BATCH}"
  --scheduler_mix_lambda "${SCHED_MIX_LAMBDA}"
  --scheduler_latency_margin "${SCHED_LATENCY_MARGIN}"
  --scheduling_cycle_frames "${SCHED_CYCLE_FRAMES}"
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

mkdir -p "${LOG_DIR}"

if [[ -f "${log_path}" && "${OVERWRITE}" != "1" ]]; then
  echo "[SKIP] ${run_name} (exists: ${log_path})"
  exit 0
fi

echo "[RUN] ${run_name}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

if [[ -n "${RAY_PORT}" && "${RAY_PORT}" != "0" ]]; then
  RAY_PORT="${RAY_PORT}" "${cmd[@]}"
else
  "${cmd[@]}"
fi
