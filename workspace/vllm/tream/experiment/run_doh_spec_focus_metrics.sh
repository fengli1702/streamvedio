#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${RUNNER:-python}"
RUN_TAG="${RUN_TAG:-2.06}"
RUN_PREFIX="${RUN_PREFIX:-doh_spec_focus}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

GPU_LIST="${GPU_LIST:-0,1}"   # e.g. "0,1" or "0,1,2,3"
RAY_PORT_BASE="${RAY_PORT_BASE:-6510}"
RAY_TEMP_ROOT="${RAY_TEMP_ROOT:-/tmp/ray_tream_focus_metrics}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/streaming-lvm-dataset/DOH}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/lvm-llama2-7b}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/inference_logs}"
RUNLOG_DIR="${RUNLOG_DIR:-${LOG_DIR}/rerun_logs}"

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_FRAMES="${MAX_FRAMES:-4000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_LORAS="${MAX_LORAS:-1}"

SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475}"
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
SPEC_DISABLE_MQA_SCORER="${SPEC_DISABLE_MQA_SCORER:-1}"

IFS=' ' read -r -a RUNNER_CMD <<< "${RUNNER}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERROR] Dataset dir not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -z "${SPEC_DRAFT_MODEL}" ]]; then
  echo "[ERROR] SPEC_DRAFT_MODEL is required." >&2
  exit 1
fi
if [[ -z "${SPEC_VOCAB_MAPPING_PATH}" ]]; then
  echo "[ERROR] SPEC_VOCAB_MAPPING_PATH is required." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${RUNLOG_DIR}"

# Points extracted from the current best-set figures:
# 1) fixed (ib,tb)=(32,16), vary (ctx,inf): {ctx1_inf1, ctx1_inf2, ctx2_inf1, ctx4_inf1, ctx6_inf1}
# 2) fixed (ctx,inf)=(4,2), vary (ib,tb): {ib32_tb32, ib32_tb8, ib8_tb32}
# + anchor point (ctx,inf,ib,tb)=(4,2,32,16)
TASKS=(
  "1 1 32 16"
  "1 2 32 16"
  "2 1 32 16"
  "4 1 32 16"
  "6 1 32 16"
  "4 2 32 32"
  "4 2 32 8"
  "4 2 8 32"
  "4 2 32 16"
)

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty." >&2
  exit 1
fi

run_one() {
  local ctx="$1"
  local inf="$2"
  local ib="$3"
  local tb="$4"
  local gpu="$5"
  local ray_port="$6"

  local run_name="${RUN_PREFIX}_${RUN_TAG}_c${ctx}_i${inf}_ib${ib}_tb${tb}"
  local infer_log="${LOG_DIR}/${run_name}.jsonl"
  local stdout_log="${RUNLOG_DIR}/${run_name}.focus.out"
  local ray_tmp="${RAY_TEMP_ROOT}_${run_name}"

  if [[ -f "${infer_log}" && "${OVERWRITE}" != "1" ]]; then
    echo "[SKIP][GPU ${gpu}] ${run_name} (exists)"
    return 0
  fi

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
    --ray_port "${ray_port}"
    --ray_temp_dir "${ray_tmp}"
    -gc
    -train
  )
  if [[ "${SPEC_DISABLE_MQA_SCORER}" == "1" ]]; then
    cmd+=(--spec_disable_mqa_scorer)
  fi

  echo "[RUN][GPU ${gpu}] ${run_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${stdout_log}" 2>&1
}

worker_loop() {
  local worker_idx="$1"
  local gpu="$2"
  local n_gpus="$3"
  local n_tasks="$4"
  local i
  for ((i=worker_idx; i<n_tasks; i+=n_gpus)); do
    local ctx inf ib tb
    IFS=' ' read -r ctx inf ib tb <<< "${TASKS[$i]}"
    local ray_port=$((RAY_PORT_BASE + i))
    run_one "${ctx}" "${inf}" "${ib}" "${tb}" "${gpu}" "${ray_port}"
  done
}

echo "[INFO] GPUs: ${GPU_LIST}"
echo "[INFO] Total focus tasks: ${#TASKS[@]}"

pids=()
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  worker_loop "${idx}" "${gpu}" "${#GPUS[@]}" "${#TASKS[@]}" &
  pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do
  if ! wait "${p}"; then
    rc=1
  fi
done

if [[ "${rc}" -ne 0 ]]; then
  echo "[ERROR] One or more focus runs failed." >&2
  exit 1
fi
echo "[OK] Focus rerun set completed."
