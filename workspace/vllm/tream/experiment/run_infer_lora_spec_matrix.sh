#!/usr/bin/env bash
#
# Run infer-only benchmarks covering four LoRA/Spec combinations.
# The script watches GPU 3,4, waits until both are idle, and retries
# automatically if a run fails due to CUDA OOM.

set -euo pipefail

CUDA_DEVICES="${CUDA_DEVICES:-3,4}"
GPU_LIST="${CUDA_DEVICES//,/ }"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

BASE_CMD=(
  python3 /app/tream.py
  --input_frames_path /app/data/streaming-lvm-dataset/DOH
  --context_length 4
  --model_name /app/saved_models/lvm-llama2-7b
  --inference_length 4
  --inference_batch_size 32
  --max_frames 1600
  --num_workers 1
  --disable_dynamic_scheduling
)

gpu_idle() {
  local gpu out
  for gpu in ${GPU_LIST}; do
    out=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [[ -n "${out}" && ! "${out}" =~ No\ running\ processes ]]; then
      return 1
    fi
  done
  return 0
}

wait_for_idle() {
  until gpu_idle; do
    echo "[scheduler] GPUs ${CUDA_DEVICES} busy; sleeping 30s..."
    sleep 30
  done
}

run_case() {
  local name="$1"; shift
  local env_vars=()
  local cli_args=()
  for arg in "$@"; do
    if [[ "${arg}" == *=* && "${arg}" != --* ]]; then
      env_vars+=("${arg}")
    else
      cli_args+=("${arg}")
    fi
  done

  local logfile="${LOG_DIR}/${name}.log"
  local attempt=1000

  while true; do
    wait_for_idle
    echo "===== Running: ${name} (attempt ${attempt}) ====="
    set +e
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" VLLM_USE_V1=1 \
      "${env_vars[@]}" \
      "${BASE_CMD[@]}" --wandb_run_name "${name}" "${cli_args[@]}" \
      2>&1 | tee "${logfile}"
    local status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
      echo "===== Finished: ${name} ====="
      break
    fi

    if grep -qi "out of memory" "${logfile}"; then
      echo "[${name}] Detected CUDA OOM. Waiting for GPU to free up before retry..."
      attempt=$((attempt + 1))
      sleep 30
      continue
    fi

    echo "[${name}] Failed with status ${status} (non-OOM). Aborting." >&2
    exit ${status}
  done
}

run_case "infer_lora_on_nospec" \
  --max_loras 1 \
  --lora_rank 8 \
  --gpu_memory_utilization 0.75

run_case "infer_lora_on_spec" \
  VLLM_MAX_SPEC_TOKENS=2 \
  VLLM_LORA_SPEC_CPU_BCAST=0 \
  VLLM_LORA_SPEC_REINDEX=0 \
  --max_loras 1 \
  --lora_rank 8 \
  --gpu_memory_utilization 0.75 \
  --use_speculative_decoding \
  --spec_method eagle3 \
  --spec_draft_model /app/SpecForge/checkpoints/lvm_eagle3_v1_8292/epoch_0 \
  --spec_vocab_mapping_path /app/SpecForge/cache/lvm_eagle3_v1_8292/vocab_mapping/027d603b856cb0a6c76f074c0414b23b.pt \
  --num_spec_tokens 2 \
  --spec_disable_mqa_scorer

run_case "infer_lora_off_spec" \
  VLLM_MAX_SPEC_TOKENS=2 \
  --max_loras 0 \
  --use_speculative_decoding \
  --spec_method eagle3 \
  --spec_draft_model /app/SpecForge/checkpoints/lvm_eagle3_v1_8292/epoch_0 \
  --spec_vocab_mapping_path /app/SpecForge/cache/lvm_eagle3_v1_8292/vocab_mapping/027d603b856cb0a6c76f074c0414b23b.pt \
  --num_spec_tokens 2 \
  --spec_disable_mqa_scorer

run_case "infer_lora_off_nospec" \
  --max_loras 0
