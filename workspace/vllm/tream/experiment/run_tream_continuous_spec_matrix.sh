#!/usr/bin/env bash
# Run four tream.py experiments to compare:
#   1) Continuous LoRA updates ON  / Spec OFF
#   2) Continuous LoRA updates ON  / Spec ON
#   3) Continuous LoRA updates OFF / Spec OFF
#   4) Continuous LoRA updates OFF / Spec ON
# Each run trains (-train), enables gradient checkpointing (-gc),
# processes 800 frames, uploads metrics to W&B, and keeps the vLLM
# optimizations aligned with the streaming bench scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_DEVICES="${CUDA_DEVICES:-2}"
DATASET_ROOT="${DATASET_ROOT:-/app/data/streaming-lvm-dataset/DOH}"
TEACHER_MODEL="${TEACHER_MODEL:-/app/saved_models/lvm-llama2-7b}"
DRAFT_MODEL="${DRAFT_MODEL:-/app/SpecForge/checkpoints/lvm_eagle3_v1_8292/epoch_0}"
VOCAB_MAPPING="${VOCAB_MAPPING:-/app/SpecForge/cache/lvm_eagle3_v1_8292/vocab_mapping/027d603b856cb0a6c76f074c0414b23b.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/tream_continuous_matrix}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_MAX_SPEC_TOKENS="${VLLM_MAX_SPEC_TOKENS:-2}"
export VLLM_LORA_SPEC_CPU_BCAST="${VLLM_LORA_SPEC_CPU_BCAST:-0}"
export VLLM_LORA_SPEC_REINDEX="${VLLM_LORA_SPEC_REINDEX:-0}"

COMMON_ARGS=(
  -gc
  -train
  --num_workers 1
  --input_frames_path "${DATASET_ROOT}"
  --model_name "${TEACHER_MODEL}"
  --context_length 4
  --inference_length 4
  --inference_batch_size 32
  --gpu_memory_utilization 0.7
  --max_frames 800
  --max_loras 1
  --lora_rank 8
  --disable_dynamic_scheduling
  --use_wandb
)

declare -a CASES=(
  "cont_on_spec_off:0:0"
  "cont_on_spec_on:0:1"
  "cont_off_spec_off:1:0"
  "cont_off_spec_on:1:1"
)

timestamp=$(date +"%Y%m%d_%H%M%S")

for entry in "${CASES[@]}"; do
  IFS=":" read -r CASE_NAME DISABLE_CONT SPEC_ON <<<"${entry}"
  if [[ "${DISABLE_CONT}" == "1" ]]; then
    CONT_LABEL="off"
  else
    CONT_LABEL="on"
  fi
  if [[ "${SPEC_ON}" == "1" ]]; then
    SPEC_LABEL="on"
  else
    SPEC_LABEL="off"
  fi
  RUN_DIR="${OUTPUT_ROOT}/${CASE_NAME}"
  mkdir -p "${RUN_DIR}/inference" "${RUN_DIR}/training"

  RUN_NAME="${CASE_NAME}_${timestamp}"
  LOG_FILE="${RUN_DIR}/${CASE_NAME}.log"

  CMD=(
    "${PYTHON_BIN}" tream.py
    "${COMMON_ARGS[@]}"
    --wandb_run_name "${RUN_NAME}"
    --inference_logs_dir "${RUN_DIR}/inference"
    --training_logs_dir "${RUN_DIR}/training"
  )

  if [[ "${DISABLE_CONT}" == "1" ]]; then
    CMD+=(--disable_continuous_lora_update)
  fi

  if [[ "${SPEC_ON}" == "1" ]]; then
    CMD+=(
      --use_speculative_decoding
      --spec_method eagle3
      --spec_draft_model "${DRAFT_MODEL}"
      --spec_vocab_mapping_path "${VOCAB_MAPPING}"
      --num_spec_tokens 2
      --spec_disable_mqa_scorer
    )
  fi

  echo "==== Running ${CASE_NAME} (continuous=${CONT_LABEL}, spec=${SPEC_LABEL}) ===="
  echo "Logs: ${LOG_FILE}"
  "${CMD[@]}" | tee "${LOG_FILE}"
  echo "==== Completed ${CASE_NAME} ===="
done
