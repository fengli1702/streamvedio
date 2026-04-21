#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cd /workspace/tream

START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ab16x2_dualgpu_shift_${START_TS}"
LOG_DIR="/workspace/tream/inference_logs"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_ab16x2_dualgpu_shift_run.txt"

mkdir -p "${LOG_DIR}"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tqueue\tgpu\tray_addr\tctx\tinf\tib\ttb\tdriver_log\tstatus\n" > "${MANIFEST}"
echo "[$(date '+%F %T')] START ${RUN_ID}" | tee -a "${STATUS_LOG}"

DATASET_DIR="/workspace/tream/data/streaming-lvm-dataset/DOH"
MODEL_PATH="/workspace/tream/lvm-llama2-7b"
SPEC_DRAFT_MODEL="../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475"
SPEC_VOCAB_MAPPING_PATH="vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt"
MAX_FRAMES="${MAX_FRAMES:-4000}"
IB="${IB:-32}"
TB="${TB:-16}"
GPU_A="${GPU_A:-5}"
GPU_B="${GPU_B:-7}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

RAY_PORT_A="${RAY_PORT_A:-28121}"
RAY_PORT_B="${RAY_PORT_B:-28122}"
RAY_TEMP_A="/tmp/ray_ab16x2_a_${START_TS}"
RAY_TEMP_B="/tmp/ray_ab16x2_b_${START_TS}"

# Keep the relaxed scheduler settings used in the better-exploration run.
SHIFT_COLD_START_WINDOWS="${SHIFT_COLD_START_WINDOWS:-12}"
SHIFT_COLD_PROBE_EVERY="${SHIFT_COLD_PROBE_EVERY:-2}"
SHIFT_ACCEPT_SHOCK_DELTA="${SHIFT_ACCEPT_SHOCK_DELTA:-0.12}"
SHIFT_ACCEPT_SHOCK_COOLDOWN="${SHIFT_ACCEPT_SHOCK_COOLDOWN:-1}"
SHIFT_COLD_AXIS_ROTATION="${SHIFT_COLD_AXIS_ROTATION:-1}"
SHIFT_DEFAULT_QUALITY_MIN="${SHIFT_DEFAULT_QUALITY_MIN:-0.2}"
SHIFT_COLD_RELAX_SAFETY="${SHIFT_COLD_RELAX_SAFETY:-1}"
SHIFT_MIN_COUNT_FOR_TRUST="${SHIFT_MIN_COUNT_FOR_TRUST:-1}"
SHIFT_ADAPT_PROBE_WINDOWS="${SHIFT_ADAPT_PROBE_WINDOWS:-8}"
SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET:-12}"
SCHEDULER_QUALITY_MIN="${SCHEDULER_QUALITY_MIN:-0.2}"

run_one() {
  local queue="$1"
  local tag="$2"
  local gpu="$3"
  local ray_port="$4"
  local ray_temp="$5"
  local ctx="$6"
  local inf="$7"

  local run_name="doh_shift_ab16x2_${queue}${tag}_${START_TS}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${queue}" "${gpu}" "local:${ray_port}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} gpu=${gpu} ray=local:${ray_port} temp=${ray_temp}" | tee -a "${STATUS_LOG}"

  if env -u RAY_ADDRESS -u RAY_PORT -u RAY_GCS_SERVER_PORT \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      TREAM_SHIFT_COLD_START_WINDOWS="${SHIFT_COLD_START_WINDOWS}" \
      TREAM_SHIFT_COLD_PROBE_EVERY="${SHIFT_COLD_PROBE_EVERY}" \
      TREAM_SHIFT_ACCEPT_SHOCK_DELTA="${SHIFT_ACCEPT_SHOCK_DELTA}" \
      TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN="${SHIFT_ACCEPT_SHOCK_COOLDOWN}" \
      TREAM_SHIFT_COLD_AXIS_ROTATION="${SHIFT_COLD_AXIS_ROTATION}" \
      TREAM_SHIFT_DEFAULT_QUALITY_MIN="${SHIFT_DEFAULT_QUALITY_MIN}" \
      TREAM_SHIFT_COLD_RELAX_SAFETY="${SHIFT_COLD_RELAX_SAFETY}" \
      TREAM_SHIFT_MIN_COUNT_FOR_TRUST="${SHIFT_MIN_COUNT_FOR_TRUST}" \
      TREAM_SHIFT_ADAPT_PROBE_WINDOWS="${SHIFT_ADAPT_PROBE_WINDOWS}" \
      TREAM_SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET}" \
      python /workspace/tream/tream.py \
        --input_frames_path "${DATASET_DIR}" \
        --context_length "${ctx}" \
        --inference_length "${inf}" \
        --inference_batch_size "${IB}" \
        --training_batch_size "${TB}" \
        --model_name "${MODEL_PATH}" \
        --num_workers 1 \
        --max_frames "${MAX_FRAMES}" \
        --gpu_memory_utilization "${GPU_MEM_UTIL}" \
        --max_loras 1 \
        --scheduler_quality_min "${SCHEDULER_QUALITY_MIN}" \
        --wandb_run_name "${run_name}" \
        --inference_logs_dir "${LOG_DIR}" \
        --ray_port "${ray_port}" \
        --ray_temp_dir "${ray_temp}" \
        --ray_namespace "${run_name}" \
        --use_speculative_decoding \
        --spec_draft_model "${SPEC_DRAFT_MODEL}" \
        --spec_vocab_mapping_path "${SPEC_VOCAB_MAPPING_PATH}" \
        --num_spec_tokens 3 \
        --spec_disable_mqa_scorer \
        -gc \
        -train \
        > "${driver_log}" 2>&1; then
    echo "[$(date '+%F %T')] DONE ${run_name} rc=0" | tee -a "${STATUS_LOG}"
    return 0
  else
    local rc=$?
    echo "[$(date '+%F %T')] FAIL ${run_name} rc=${rc}" | tee -a "${STATUS_LOG}"
    return "${rc}"
  fi
}

# Small/medium/large mixed set, covering diagonal and cross extremes.
# A queue: balanced + diagonal progression.
CASES_A=(
  "1,1" "1,2" "2,1" "2,2"
  "2,4" "3,3" "4,2" "4,4"
  "3,6" "6,3" "4,6" "6,4"
  "5,5" "6,6" "7,7" "8,8"
)

# B queue: stronger cross-shape coverage and asymmetric workloads.
CASES_B=(
  "1,8" "8,1" "2,8" "8,2"
  "3,8" "8,3" "4,8" "8,4"
  "2,6" "6,2" "3,6" "6,3"
  "4,6" "6,4" "5,7" "7,5"
)

run_case_list() {
  local queue="$1"
  local gpu="$2"
  local ray_port="$3"
  local ray_temp="$4"
  shift 4
  local cases=("$@")

  local failures=0
  local idx=0
  for spec in "${cases[@]}"; do
    idx=$((idx + 1))
    IFS=',' read -r ctx inf <<< "${spec}"
    local tag
    tag="$(printf "%02d" "${idx}")"
    run_one "${queue}" "${tag}" "${gpu}" "${ray_port}" "${ray_temp}" "${ctx}" "${inf}" || failures=$((failures + 1))
  done
  return "${failures}"
}

queue_a() {
  run_case_list "A" "${GPU_A}" "${RAY_PORT_A}" "${RAY_TEMP_A}" "${CASES_A[@]}"
}

queue_b() {
  echo "[$(date '+%F %T')] queue_b warmup sleep 30s" | tee -a "${STATUS_LOG}"
  sleep 30
  run_case_list "B" "${GPU_B}" "${RAY_PORT_B}" "${RAY_TEMP_B}" "${CASES_B[@]}"
}

cleanup() {
  echo "[$(date '+%F %T')] RAY_STOP_ALL" | tee -a "${STATUS_LOG}"
  ray stop --force >> "${STATUS_LOG}" 2>&1 || true
}
trap cleanup EXIT

echo "[$(date '+%F %T')] RAY_MODE local_auto (--ray_port, no --ray_address)" | tee -a "${STATUS_LOG}"
ray stop --force >> "${STATUS_LOG}" 2>&1 || true

queue_a &
PID_A=$!
queue_b &
PID_B=$!

wait "${PID_A}" || RC_A=$?
RC_A=${RC_A:-0}
wait "${PID_B}" || RC_B=$?
RC_B=${RC_B:-0}

TOTAL_FAIL=$((RC_A + RC_B))
if (( TOTAL_FAIL == 0 )); then
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} all_success" | tee -a "${STATUS_LOG}"
else
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} failures=${TOTAL_FAIL} (A=${RC_A}, B=${RC_B})" | tee -a "${STATUS_LOG}"
fi
