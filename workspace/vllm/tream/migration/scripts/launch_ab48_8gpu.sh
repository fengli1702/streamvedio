#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREAM_ROOT="${TREAM_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${TREAM_ROOT}"

RUN_MODE="${RUN_MODE:-static}"      # static | dynamic
RUN_SPEC="${RUN_SPEC:-1}"           # 1=spec on, 0=spec off
DATASET_SUBDIR="${DATASET_SUBDIR:-DOH}"
DATASET_DIR="${DATASET_DIR:-${TREAM_ROOT}/data/streaming-lvm-dataset/${DATASET_SUBDIR}}"
MODEL_PATH="${MODEL_PATH:-${TREAM_ROOT}/lvm-llama2-7b}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475}"
SPEC_VOCAB_MAPPING_PATH="${SPEC_VOCAB_MAPPING_PATH:-vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt}"

MAX_FRAMES="${MAX_FRAMES:-4000}"
IB="${IB:-32}"
TB="${TB:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
SCHEDULER_QUALITY_MIN="${SCHEDULER_QUALITY_MIN:-0.2}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CONCURRENCY="${CONCURRENCY:-8}"
CASE_FILTER="${CASE_FILTER:-}"      # e.g. A01,B04,C08
LOG_DIR="${LOG_DIR:-${TREAM_ROOT}/inference_logs}"

mkdir -p "${LOG_DIR}"
START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ab48x8_${RUN_MODE}_spec${RUN_SPEC}_${DATASET_SUBDIR}_${START_TS}"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"

echo "run_name\tlane\tcase_id\tgpu\tctx\tinf\tib\ttb\tdriver_log\tstatus" > "${MANIFEST}"

if [[ "${RUN_MODE}" != "static" && "${RUN_MODE}" != "dynamic" ]]; then
  echo "RUN_MODE must be static|dynamic, got ${RUN_MODE}" >&2
  exit 1
fi
if [[ "${RUN_SPEC}" != "0" && "${RUN_SPEC}" != "1" ]]; then
  echo "RUN_SPEC must be 0|1, got ${RUN_SPEC}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
  echo "No GPUs in GPU_LIST" >&2
  exit 1
fi
if (( CONCURRENCY < 1 )); then
  echo "CONCURRENCY must be >=1" >&2
  exit 1
fi
if (( CONCURRENCY > ${#GPUS[@]} )); then
  CONCURRENCY="${#GPUS[@]}"
fi

CASES_A=("1,1" "1,2" "2,1" "2,2" "2,3" "3,2" "3,3" "2,4")
CASES_B=("3,4" "4,3" "4,4" "3,5" "5,3" "4,5" "5,4" "5,5")
CASES_C=("1,8" "8,1" "2,8" "8,2" "3,8" "8,3" "4,8" "8,4")
CASES_D=("6,1" "1,6" "6,2" "2,6" "6,3" "3,6" "6,4" "4,6")
CASES_E=("6,5" "5,6" "6,6" "7,1" "1,7" "7,2" "2,7" "7,3")
CASES_F=("3,7" "7,4" "4,7" "7,5" "5,7" "7,6" "6,7" "7,7")

JOBS=()
for i in "${!CASES_A[@]}"; do JOBS+=("A$(printf '%02d' $((i+1))),${CASES_A[$i]}"); done
for i in "${!CASES_B[@]}"; do JOBS+=("B$(printf '%02d' $((i+1))),${CASES_B[$i]}"); done
for i in "${!CASES_C[@]}"; do JOBS+=("C$(printf '%02d' $((i+1))),${CASES_C[$i]}"); done
for i in "${!CASES_D[@]}"; do JOBS+=("D$(printf '%02d' $((i+1))),${CASES_D[$i]}"); done
for i in "${!CASES_E[@]}"; do JOBS+=("E$(printf '%02d' $((i+1))),${CASES_E[$i]}"); done
for i in "${!CASES_F[@]}"; do JOBS+=("F$(printf '%02d' $((i+1))),${CASES_F[$i]}"); done

if [[ -n "${CASE_FILTER}" ]]; then
  declare -A WANT=()
  IFS=',' read -r -a CASES <<< "${CASE_FILTER}"
  for c in "${CASES[@]}"; do WANT["${c}"]=1; done
  FILTERED=()
  for spec in "${JOBS[@]}"; do
    IFS=',' read -r cid _ctx _inf <<< "${spec}"
    if [[ -n "${WANT[${cid}]:-}" ]]; then
      FILTERED+=("${spec}")
    fi
  done
  JOBS=("${FILTERED[@]}")
fi

TOTAL="${#JOBS[@]}"
echo "[$(date '+%F %T')] START ${RUN_ID} total_cases=${TOTAL} concurrency=${CONCURRENCY} gpus=${GPU_LIST}" | tee -a "${STATUS_LOG}"
[[ -n "${CASE_FILTER}" ]] && echo "[$(date '+%F %T')] FILTER ${CASE_FILTER}" | tee -a "${STATUS_LOG}"

if (( TOTAL == 0 )); then
  echo "No cases selected." | tee -a "${STATUS_LOG}"
  exit 0
fi

QUEUE_DIR="/tmp/${RUN_ID}"
mkdir -p "${QUEUE_DIR}"
for ((i=0;i<CONCURRENCY;i++)); do : > "${QUEUE_DIR}/lane_${i}.jobs"; done

for idx in "${!JOBS[@]}"; do
  lane=$((idx % CONCURRENCY))
  echo "${JOBS[$idx]}" >> "${QUEUE_DIR}/lane_${lane}.jobs"
done

run_one() {
  local lane="$1" gpu="$2" case_id="$3" ctx="$4" inf="$5"
  local mode_tag
  if [[ "${RUN_MODE}" == "dynamic" ]]; then
    mode_tag="dynamic"
  else
    mode_tag="static_nodyn"
  fi
  if [[ "${RUN_SPEC}" == "1" ]]; then
    mode_tag="${mode_tag}_spec"
  else
    mode_tag="${mode_tag}_nospec"
  fi

  local run_name="doh_shift_ab48x8_${mode_tag}_${case_id}_${START_TS}_g${gpu}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"
  echo "[$(date '+%F %T')] RUN lane=${lane} gpu=${gpu} case=${case_id} ctx=${ctx} inf=${inf}" | tee -a "${STATUS_LOG}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${lane}" "${case_id}" "${gpu}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  local -a cmd
  cmd=(python "${TREAM_ROOT}/tream.py"
    --input_frames_path "${DATASET_DIR}"
    --context_length "${ctx}"
    --inference_length "${inf}"
    --inference_batch_size "${IB}"
    --training_batch_size "${TB}"
    --model_name "${MODEL_PATH}"
    --num_workers 1
    --max_frames "${MAX_FRAMES}"
    --gpu_memory_utilization "${GPU_MEM_UTIL}"
    --max_loras 1
    --scheduler_quality_min "${SCHEDULER_QUALITY_MIN}"
    --wandb_run_name "${run_name}"
    --inference_logs_dir "${LOG_DIR}"
    -gc
    -train
  )
  if [[ "${RUN_SPEC}" == "1" ]]; then
    cmd+=(
      --use_speculative_decoding
      --spec_draft_model "${SPEC_DRAFT_MODEL}"
      --spec_vocab_mapping_path "${SPEC_VOCAB_MAPPING_PATH}"
      --num_spec_tokens 3
      --spec_disable_mqa_scorer
    )
  fi
  if [[ "${RUN_MODE}" == "static" ]]; then
    cmd+=(--disable_dynamic_scheduling)
  fi

  local rc=0
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH="/workspace/vllm:${PYTHONPATH:-}" \
  "${cmd[@]}" > "${driver_log}" 2>&1 || rc=$?

  if (( rc == 0 )); then
    echo "[$(date '+%F %T')] DONE case=${case_id} lane=${lane} gpu=${gpu}" | tee -a "${STATUS_LOG}"
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL case=${case_id} lane=${lane} gpu=${gpu} rc=${rc}" | tee -a "${STATUS_LOG}"
  return "${rc}"
}

lane_worker() {
  local lane="$1"
  local gpu="${GPUS[$lane]}"
  local fail=0
  while IFS=',' read -r case_id ctx inf; do
    [[ -z "${case_id:-}" ]] && continue
    run_one "${lane}" "${gpu}" "${case_id}" "${ctx}" "${inf}" || fail=$((fail+1))
  done < "${QUEUE_DIR}/lane_${lane}.jobs"
  return "${fail}"
}

PIDS=()
for ((lane=0; lane<CONCURRENCY; lane++)); do
  lane_worker "${lane}" & PIDS+=("$!")
done

FAIL_TOTAL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL_TOTAL=$((FAIL_TOTAL+1))
  fi
done

echo "[$(date '+%F %T')] FINISH ${RUN_ID} lane_failures=${FAIL_TOTAL}" | tee -a "${STATUS_LOG}"
echo "manifest=${MANIFEST}" | tee -a "${STATUS_LOG}"
echo "status_log=${STATUS_LOG}" | tee -a "${STATUS_LOG}"
