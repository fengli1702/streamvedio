#!/usr/bin/env bash
set -u

TREAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREAM_SCRIPT="${TREAM_DIR}/tream.py"

INPUT_FRAMES_PATH="${INPUT_FRAMES_PATH:-${TREAM_DIR}/data/streaming-lvm-dataset/motsynth}"
MODEL_NAME="${MODEL_NAME:-${TREAM_DIR}/lvm-llama2-7b}"
DATASET_TAG="${DATASET_TAG:-$(basename "${INPUT_FRAMES_PATH}")}"
DATASET_TAG="${DATASET_TAG// /-}"

CTX_LEN="${CTX_LEN:-4}"
INF_LEN="${INF_LEN:-4}"
INF_BS_BASE="${INF_BS_BASE:-8}"
TRAIN_BS_BASE="${TRAIN_BS_BASE:-8}"

TRAIN_BS_LIST="${TRAIN_BS_LIST:-2 4 8 16}"
LR_LIST="${LR_LIST:-1e-5 2e-5 5e-5}"
INFER_BS_LIST_A4="${INFER_BS_LIST_A4:-4 8 16}"
TRAIN_BS_LIST_A4="${TRAIN_BS_LIST_A4:-4 8 16}"

RUN_PREFIX="${RUN_PREFIX:-stepA}"
MAX_FRAMES="${MAX_FRAMES:-1600}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_LORAS="${MAX_LORAS:-1}"
OPTIMIZATION_PRIORITY="${OPTIMIZATION_PRIORITY:-latency}"
DISABLE_DYNAMIC_SCHEDULING="${DISABLE_DYNAMIC_SCHEDULING:-1}"

SCHED_WINDOW_SIZE="${SCHED_WINDOW_SIZE:-3}"
SCHED_Q_BATCH="${SCHED_Q_BATCH:-2}"
SCHED_MIX_LAMBDA="${SCHED_MIX_LAMBDA:-0.7}"
TARGET_LATENCY="${TARGET_LATENCY:-999}"
LATENCY_MARGIN="${LATENCY_MARGIN:-0.1}"
QUALITY_MIN="${QUALITY_MIN:-0}"

USE_SPEC="${USE_SPEC:-0}"
SPEC_METHOD="${SPEC_METHOD:-eagle3}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-${TREAM_DIR}/../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475}"
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-${TREAM_DIR}/vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt}"
SPEC_DISABLE_MQA_SCORER="${SPEC_DISABLE_MQA_SCORER:-1}"
DISABLE_SPEC_METRICS_LOG="${DISABLE_SPEC_METRICS_LOG:-1}"

NO_TRAIN="${NO_TRAIN:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

RUN_A2="${RUN_A2:-1}"
RUN_A3="${RUN_A3:-1}"
RUN_A4="${RUN_A4:-1}"

spec_tag="nospec"
if [[ "${USE_SPEC}" == "1" ]]; then
  spec_tag="spec"
fi
sched_tag="dyn"
if [[ "${DISABLE_DYNAMIC_SCHEDULING}" == "1" ]]; then
  sched_tag="static"
fi

run_one() {
  local phase="$1"
  local infer_bs="$2"
  local train_bs="$3"
  local lr="$4"

  local run_name="${RUN_PREFIX}_${phase}_ctx${CTX_LEN}_inf${INF_LEN}_ib${infer_bs}_tb${train_bs}_lr${lr}_${spec_tag}_${sched_tag}_data-${DATASET_TAG}"

  cmd=(
    python "${TREAM_SCRIPT}"
    --input_frames_path "${INPUT_FRAMES_PATH}"
    --context_length "${CTX_LEN}"
    --model_name "${MODEL_NAME}"
    --inference_length "${INF_LEN}"
    --inference_batch_size "${infer_bs}"
    --training_batch_size "${train_bs}"
    --learning_rate "${lr}"
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
    --scheduler_quality_min "${QUALITY_MIN}"
  )

  if [[ "${NO_TRAIN}" != "1" ]]; then
    cmd+=(-train)
  fi
  if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    cmd+=(-gc)
  fi
  if [[ "${DISABLE_DYNAMIC_SCHEDULING}" == "1" ]]; then
    cmd+=(--disable_dynamic_scheduling)
  fi

  if [[ "${USE_SPEC}" == "1" ]]; then
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

  echo "[RUN] ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${run_name} (exit=${rc})"
  fi
}

if [[ "${RUN_A2}" == "1" ]]; then
  echo "[PHASE] A2: sweep training_batch_size (fixed infer_bs=${INF_BS_BASE}, lr=${TRAIN_LR:-${LR_LIST%% *}})"
  base_lr="${TRAIN_LR:-${LR_LIST%% *}}"
  for train_bs in ${TRAIN_BS_LIST}; do
    run_one "a2" "${INF_BS_BASE}" "${train_bs}" "${base_lr}"
  done
fi

if [[ "${RUN_A3}" == "1" ]]; then
  echo "[PHASE] A3: sweep learning_rate (fixed infer_bs=${INF_BS_BASE}, train_bs=${TRAIN_BS_BASE})"
  for lr in ${LR_LIST}; do
    run_one "a3" "${INF_BS_BASE}" "${TRAIN_BS_BASE}" "${lr}"
  done
fi

if [[ "${RUN_A4}" == "1" ]]; then
  echo "[PHASE] A4: grid infer_bs x train_bs"
  for infer_bs in ${INFER_BS_LIST_A4}; do
    for train_bs in ${TRAIN_BS_LIST_A4}; do
      run_one "a4" "${infer_bs}" "${train_bs}" "${TRAIN_LR:-${LR_LIST%% *}}"
    done
  done
fi
