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

# Knob grids (comma-separated lists).
IB_LIST_CSV="${IB_LIST_CSV:-2,8,16}"
TB_LIST_CSV="${TB_LIST_CSV:-2,8,16}"
LR_LIST_CSV="${LR_LIST_CSV:-2e-5}"
SPEC_LIST_CSV="${SPEC_LIST_CSV:-off,on}"

# Baselines for single-axis sweeps.
BASE_IB="${BASE_IB:-8}"
BASE_TB="${BASE_TB:-8}"
BASE_LR="${BASE_LR:-2e-5}"
BASE_SPEC="${BASE_SPEC:-off}"

# Spec decoding config (required when spec=on).
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-}"
DEFAULT_VOCAB_MAPPING=""
if [[ -d "${REPO_ROOT}/vocab_mapping" ]]; then
  DEFAULT_VOCAB_MAPPING="$(ls "${REPO_ROOT}/vocab_mapping"/*.pt 2>/dev/null | head -n1 || true)"
fi
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-${DEFAULT_VOCAB_MAPPING}}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
# Minimal spec-on probe set for cross-end coupling (0=off,1=on).
SPECON_EXTRA="${SPECON_EXTRA:-0}"
# Specon probe lists (comma-separated) for minimal 6-point extension.
SPECON_EXTRA_IB_LIST="${SPECON_EXTRA_IB_LIST:-2,8,16}"
SPECON_EXTRA_TB_LIST="${SPECON_EXTRA_TB_LIST:-2,8,16}"
# Skip the base grid entirely (0=off,1=on). Use with SPECON_EXTRA=1 to only run 6 specon points.
SKIP_BASE_GRID="${SKIP_BASE_GRID:-0}"

# ---- End user-tunable knobs ----

read -r -a RUNNER_CMD <<< "${RUNNER}"
IFS=',' read -r -a WORKLOADS <<< "${WORKLOADS_CSV}"
IFS=',' read -r -a IB_LIST <<< "${IB_LIST_CSV}"
IFS=',' read -r -a TB_LIST <<< "${TB_LIST_CSV}"
IFS=',' read -r -a LR_LIST <<< "${LR_LIST_CSV}"
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
  local mode="$1"
  local ctx="$2"
  local inf="$3"
  local ib="$4"
  local tb="$5"
  local lr="$6"
  local spec="$7"

  local run_name="${RUN_PREFIX}_${RUN_TAG}_${mode}_c${ctx}_i${inf}_ib${ib}_tb${tb}_lr${lr}_spec${spec}"
  local log_path="${LOG_DIR}/${run_name}.jsonl"
  local cmd=(
    "${RUNNER_CMD[@]}" "${REPO_ROOT}/tream.py"
    --input_frames_path "${DATASET_DIR}"
    --context_length "${ctx}"
    --inference_length "${inf}"
    --inference_batch_size "${ib}"
    --training_batch_size "${tb}"
    --learning_rate "${lr}"
    --model_name "${MODEL_PATH}"
    --num_workers "${NUM_WORKERS}"
    --max_frames "${MAX_FRAMES}"
    --gpu_memory_utilization "${GPU_MEM_UTIL}"
    --max_loras "${MAX_LORAS}"
    --wandb_run_name "${run_name}"
    --disable_dynamic_scheduling
    -gc
  )

  if [[ "${mode}" != "infer_only" ]]; then
    cmd+=(-train)
  fi

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

if [[ "${SKIP_BASE_GRID}" != "1" ]]; then
  for workload in "${WORKLOADS[@]}"; do
    IFS=':' read -r ctx inf <<< "${workload}"
    if [[ -z "${ctx}" || -z "${inf}" ]]; then
      echo "[ERROR] Invalid WORKLOADS_CSV entry: ${workload}" >&2
      exit 1
    fi

    # Infer-only: sweep ib x spec, fix training knobs.
    for ib in "${IB_LIST[@]}"; do
      for spec in "${SPEC_LIST[@]}"; do
        run_one "infer_only" "${ctx}" "${inf}" "${ib}" "${BASE_TB}" "${BASE_LR}" "${spec}"
      done
    done

    # Train-only: sweep tb x lr, fix inference knobs.
    for tb in "${TB_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        run_one "train_only" "${ctx}" "${inf}" "${BASE_IB}" "${tb}" "${lr}" "${BASE_SPEC}"
      done
    done

    # Joint: full grid across all knobs.
    for ib in "${IB_LIST[@]}"; do
      for tb in "${TB_LIST[@]}"; do
        for lr in "${LR_LIST[@]}"; do
          for spec in "${SPEC_LIST[@]}"; do
            run_one "joint" "${ctx}" "${inf}" "${ib}" "${tb}" "${lr}" "${spec}"
          done
        done
      done
    done
  done
fi

# ---- Minimal specon probes (6 points) ----
# A) fix ib=8, sweep tb to test spec×tb coupling
# B) fix tb=8, sweep ib to test spec×ib shape
if [[ "${SPECON_EXTRA}" == "1" ]]; then
  for workload in "${WORKLOADS[@]}"; do
    IFS=':' read -r ctx inf <<< "${workload}"
    if [[ -z "${ctx}" || -z "${inf}" ]]; then
      echo "[ERROR] Invalid WORKLOADS_CSV entry: ${workload}" >&2
      exit 1
    fi

    IFS=',' read -r -a SPECON_TB_LIST <<< "${SPECON_EXTRA_TB_LIST}"
    IFS=',' read -r -a SPECON_IB_LIST <<< "${SPECON_EXTRA_IB_LIST}"

    # A. joint, ib=8, tb in SPECON_EXTRA_TB_LIST, spec=on
    for tb in "${SPECON_TB_LIST[@]}"; do
      run_one "joint" "${ctx}" "${inf}" 8 "${tb}" "${BASE_LR}" "on"
    done

    # B. infer_only, tb=8, ib in SPECON_EXTRA_IB_LIST, spec=on
    for ib in "${SPECON_IB_LIST[@]}"; do
      run_one "infer_only" "${ctx}" "${inf}" "${ib}" 8 "${BASE_LR}" "on"
    done
  done
fi

# ---- New data points: ib=32, tb=32 ----
echo "[INFO] Running new data points with ib=32, tb=32"
for workload in "${WORKLOADS[@]}"; do
  IFS=':' read -r ctx inf <<< "${workload}"
  if [[ -z "${ctx}" || -z "${inf}" ]]; then
    echo "[ERROR] Invalid WORKLOADS_CSV entry: ${workload}" >&2
    exit 1
  fi

  # Test all three modes with ib=32, tb=32
  # infer_only: ib=32, spec=off
  run_one "infer_only" "${ctx}" "${inf}" 32 "${BASE_TB}" "${BASE_LR}" "off"
  # infer_only: ib=32, spec=on
  run_one "infer_only" "${ctx}" "${inf}" 32 "${BASE_TB}" "${BASE_LR}" "on"
  
  # train_only: tb=32
  run_one "train_only" "${ctx}" "${inf}" "${BASE_IB}" 32 "${BASE_LR}" "${BASE_SPEC}"
  
  # joint: ib=32, tb=32, spec=off
  run_one "joint" "${ctx}" "${inf}" 32 32 "${BASE_LR}" "off"
  # joint: ib=32, tb=32, spec=on
  run_one "joint" "${ctx}" "${inf}" 32 32 "${BASE_LR}" "on"
done
