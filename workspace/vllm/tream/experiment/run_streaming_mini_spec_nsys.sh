#!/usr/bin/env bash
#
# Streaming mini_spec benchmark matrix (LoRA/Spec combinations) with Nsight capture.
# Generates 4 Nsight traces + per-case logs under ${OUTPUT_ROOT}/<case_name>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
DATASET_ROOT="${DATASET_ROOT:-/app/data/streaming-lvm-dataset/DOH}"
TEACHER_MODEL="${TEACHER_MODEL:-/app/saved_models/lvm-llama2-7b}"
DRAFT_MODEL="${DRAFT_MODEL:-/app/SpecForge/checkpoints/lvm_eagle3_v1_8292/epoch_0}"
VOCAB_MAPPING="${VOCAB_MAPPING:-/app/SpecForge/cache/lvm_eagle3_v1_8292/vocab_mapping/027d603b856cb0a6c76f074c0414b23b.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/mini_logs_stream_matrix_lora_on_test_nolog}"
MAX_FRAMES="${MAX_FRAMES:-800}"
NUM_PROMPTS="${NUM_PROMPTS:-800}"
NSYS_BIN="${NSYS_BIN:-nsys}"
ENABLE_NSYS="${ENABLE_NSYS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LORA_ADAPTER_DIR="${LORA_ADAPTER_DIR:-/app/inference_lora_adapters_34829/step_6}"
ENABLE_SPEC_METRICS="${ENABLE_SPEC_METRICS:-1}"

mkdir -p "${OUTPUT_ROOT}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export VLLM_USE_V1=1
export VLLM_MAX_SPEC_TOKENS=2
export VLLM_LORA_SPEC_CPU_BCAST=0
export VLLM_LORA_SPEC_REINDEX=0

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] VLLM_USE_V1=${VLLM_USE_V1}"
echo "[env] VLLM_MAX_SPEC_TOKENS=${VLLM_MAX_SPEC_TOKENS}"
echo "[env] VLLM_LORA_SPEC_CPU_BCAST=${VLLM_LORA_SPEC_CPU_BCAST}"
echo "[env] VLLM_LORA_SPEC_REINDEX=${VLLM_LORA_SPEC_REINDEX}"
echo "[env] LORA_ADAPTER_DIR=${LORA_ADAPTER_DIR:-<unset>}"
echo "[env] ENABLE_SPEC_METRICS=${ENABLE_SPEC_METRICS}"

if [[ -n "${LORA_ADAPTER_DIR}" && ! -d "${LORA_ADAPTER_DIR}" ]]; then
  echo "[ERR] LORA_ADAPTER_DIR does not exist: ${LORA_ADAPTER_DIR}" >&2
  exit 1
fi

NSYS_ARGS_COMMON=(
  profile
  --trace=cuda,osrt,nvtx
  --cuda-graph-trace=node
  --wait=all
  --sample=none
  --trace-fork-before-exec=true
  --force-overwrite=true
)

COMMON_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --teacher-model "${TEACHER_MODEL}"
  --context-length 4
  --inference-length 4
  --inference-batch-size 32
  --num-prompts "${NUM_PROMPTS}"
  --vqgan-batch 32
  --gpu-mem-util 0.75
  --dtype bf16
  --streaming
  --sleep-seconds 0
)
if [[ -n "${MAX_FRAMES}" ]]; then
  COMMON_ARGS+=(--max-frames "${MAX_FRAMES}")
fi
#"stream_lora_on_nospec:baseline:lora"
#  "stream_lora_on_spec:spec:lora"
#  "stream_lora_off_spec:spec:nolora"
#"stream_lora_off_nospec:baseline:nolora"
declare -a CASES=(
"stream_lora_on_nospec:baseline:lora"
  "stream_lora_on_spec:spec:lora"
  "stream_lora_off_spec:spec:nolora"
"stream_lora_off_nospec:baseline:nolora"
)

for entry in "${CASES[@]}"; do
  IFS=":" read -r CASE_NAME RUN_MODE LORA_MODE <<<"${entry}"
  CASE_DIR="${OUTPUT_ROOT}/${CASE_NAME}"
  mkdir -p "${CASE_DIR}"
  LOG_FILE="${CASE_DIR}/${CASE_NAME}.log"
  OUT_DIR="${CASE_DIR}/mini_spec_logs"
  mkdir -p "${OUT_DIR}"

  CMD=("${PYTHON_BIN}" experiments/mini_spec_benchmark.py "${COMMON_ARGS[@]}"
       --run-mode "${RUN_MODE}"
       --output-dir "${OUT_DIR}"
       --log-file "${LOG_FILE}")

  if [[ "${LORA_MODE}" == "lora" ]]; then
    CMD+=(--max-lora-rank 16 --max-loras 4)
    if [[ -n "${LORA_ADAPTER_DIR}" ]]; then
      CMD+=(--lora-adapter-dir "${LORA_ADAPTER_DIR}")
    else
      echo "[WARN] LORA_MODE=lora but LORA_ADAPTER_DIR is unset; running without external adapter payload."
    fi
  else
    CMD+=(--disable-lora)
  fi

  if [[ "${RUN_MODE}" == "spec" ]]; then
    CMD+=(--draft-model "${DRAFT_MODEL}"
          --spec-method eagle3
          --num-spec-tokens 2
          --spec-disable-mqa-scorer
          --vocab-mapping-path "${VOCAB_MAPPING}")
    if [[ "${ENABLE_SPEC_METRICS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
      SPEC_METRICS_PATH="${CASE_DIR}/spec_metrics.jsonl"
    else
      SPEC_METRICS_PATH=""
    fi
  else
    SPEC_METRICS_PATH=""
  fi

  NSYS_OUT="${CASE_DIR}/${CASE_NAME}"
  echo "==== Running ${CASE_NAME} (run_mode=${RUN_MODE}, lora=${LORA_MODE}) ===="
  echo "Logs: ${LOG_FILE}"
  echo "Nsight: ${NSYS_OUT}.nsys-rep"

  if [[ -n "${SPEC_METRICS_PATH}" ]]; then
    export VLLM_SPEC_METRICS_PATH="${SPEC_METRICS_PATH}"
  else
    unset VLLM_SPEC_METRICS_PATH || true
  fi

  if [[ "${ENABLE_NSYS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    rm -f "${NSYS_OUT}.nsys-rep" "${NSYS_OUT}.qdstrm" 2>/dev/null || true
    "${NSYS_BIN}" "${NSYS_ARGS_COMMON[@]}" --output "${NSYS_OUT}" \
      "${CMD[@]}"
  else
    echo "[WARN] ENABLE_NSYS=${ENABLE_NSYS}; skipping Nsight capture."
    "${CMD[@]}"
  fi

  echo "==== Completed ${CASE_NAME} ===="
done
