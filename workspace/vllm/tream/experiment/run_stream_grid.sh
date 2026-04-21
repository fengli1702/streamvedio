#!/usr/bin/env bash
set -u

TREAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREAM_SCRIPT="${TREAM_DIR}/tream.py"
LOG_DIR="${TREAM_DIR}/inference_logs"

INPUT_FRAMES_PATH="${INPUT_FRAMES_PATH:-${TREAM_DIR}/data/streaming-lvm-dataset/motsynth}"
MODEL_NAME="${MODEL_NAME:-${TREAM_DIR}/lvm-llama2-7b}"
DATASET_TAG="${DATASET_TAG:-$(basename "${INPUT_FRAMES_PATH}")}"
DATASET_TAG="${DATASET_TAG// /-}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475}"
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt}"

SPEC_METHOD="${SPEC_METHOD:-eagle3}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
SPEC_DISABLE_MQA_SCORER="${SPEC_DISABLE_MQA_SCORER:-0}"

INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-32}"
MAX_FRAMES="${MAX_FRAMES:-1600}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_LORAS="${MAX_LORAS:-1}"
OPTIMIZATION_PRIORITY="${OPTIMIZATION_PRIORITY:-latency}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-12}"
RUN_PREFIX="${RUN_PREFIX:-1-6}"
RUN_SPEC="${RUN_SPEC:-0}"

SCHED_WINDOW_SIZE="${SCHED_WINDOW_SIZE:-3}"
SCHED_Q_BATCH="${SCHED_Q_BATCH:-2}"
SCHED_MIX_LAMBDA="${SCHED_MIX_LAMBDA:-0.7}"
LATENCY_MARGIN="${LATENCY_MARGIN:-0.1}"
DISABLE_SPEC_METRICS_LOG="${DISABLE_SPEC_METRICS_LOG:-1}"
TARGET_LATENCY="${TARGET_LATENCY:-2.0}"
QUALITY_MIN="${QUALITY_MIN:-0.3}"
DISABLE_DYNAMIC_SCHED="${DISABLE_DYNAMIC_SCHED:-0}"

CTX_VALUES="${CTX_VALUES:-1 2 4 6 8}"
INF_VALUES="${INF_VALUES:-1 2 4 6 8}"

NO_TRAIN="${NO_TRAIN:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

compute_sla_defaults() {
  return 0
}

run_one() {
  local ctx="$1"
  local inf="$2"
  local spec_tag="$3"

  local run_name="${RUN_PREFIX}_ctx${ctx}_inf${inf}_${spec_tag}_dyn_data-${DATASET_TAG}"
  local log_path="${LOG_DIR}/${run_name}.jsonl"

  if [[ -f "${log_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "[SKIP] ${run_name} (exists: ${log_path})"
    return 0
  fi

  local cmd=(
    python "${TREAM_SCRIPT}"
    --input_frames_path "${INPUT_FRAMES_PATH}"
    --context_length "${ctx}"
    --model_name "${MODEL_NAME}"
    --inference_length "${inf}"
    --inference_batch_size "${INFERENCE_BATCH_SIZE}"
    --wandb_run_name "${run_name}"
    --max_frames "${MAX_FRAMES}"
    --num_workers "${NUM_WORKERS}"
    --max_loras "${MAX_LORAS}"
    --optimization_priority "${OPTIMIZATION_PRIORITY}"
    --scheduler_window_size "${SCHED_WINDOW_SIZE}"
    --scheduler_q_batch "${SCHED_Q_BATCH}"
    --scheduler_mix_lambda "${SCHED_MIX_LAMBDA}"
    --scheduler_target_latency "${TARGET_LATENCY}"
    --scheduler_latency_margin "${LATENCY_MARGIN}"
  )
  if [[ "${DISABLE_DYNAMIC_SCHED}" == "1" ]]; then
    cmd+=(--disable_dynamic_scheduling)
  fi

  if [[ "${NO_TRAIN}" != "1" ]]; then
    cmd+=(-train)
  fi
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    cmd+=(-gc)
  fi
  if [[ "${spec_tag}" == "spec" ]]; then
    cmd+=(
      --use_speculative_decoding
      --spec_method "${SPEC_METHOD}"
      --spec_draft_model "${SPEC_DRAFT_MODEL}"
      --spec_vocab_mapping_path "${SPEC_VOCAB_MAPPING_PATH}"
      --num_spec_tokens "${NUM_SPEC_TOKENS}"
    )
    if [[ "${SPEC_DISABLE_MQA_SCORER}" == "1" ]]; then
      cmd+=(--spec_disable_mqa_scorer)
    fi
    if [[ "${DISABLE_SPEC_METRICS_LOG}" == "1" ]]; then
      cmd+=(--disable_spec_metrics_log)
    fi
  fi
  if [[ -n "${QUALITY_MIN:-}" ]]; then
    cmd+=(--scheduler_quality_min "${QUALITY_MIN}")
  fi

  echo "[RUN] ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${run_name} (exit=${rc})"
    return 0
  fi
  return 0
}

compute_sla_defaults

for ctx in ${CTX_VALUES}; do
  for inf in ${INF_VALUES}; do
    if (( ctx + inf > MAX_TOTAL_LEN )); then
      continue
    fi
    run_one "${ctx}" "${inf}" "nospec"
    if [[ "${RUN_SPEC}" == "1" ]]; then
      run_one "${ctx}" "${inf}" "spec"
    fi
  done
done
