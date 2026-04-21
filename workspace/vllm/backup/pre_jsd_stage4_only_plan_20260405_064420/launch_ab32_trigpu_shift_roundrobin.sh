#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cd /workspace/tream

START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ab32x3_trigpu_shift_${START_TS}"
LOG_DIR="/workspace/tream/inference_logs"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_ab32x3_trigpu_shift_run.txt"

mkdir -p "${LOG_DIR}"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tlane\tcase_id\tgpu\tray_addr\tctx\tinf\tib\ttb\tdriver_log\tstatus\n" > "${MANIFEST}"
echo "[$(date '+%F %T')] START ${RUN_ID}" | tee -a "${STATUS_LOG}"

DATASET_DIR="/workspace/tream/data/streaming-lvm-dataset/DOH"
MODEL_PATH="/workspace/tream/lvm-llama2-7b"
SPEC_DRAFT_MODEL="../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475"
SPEC_VOCAB_MAPPING_PATH="vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt"
MAX_FRAMES="${MAX_FRAMES:-4000}"
IB="${IB:-32}"
TB="${TB:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"

# Default to free cards on this node (GPU6 is typically occupied by another service).
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-5}"
GPU_C="${GPU_C:-7}"

RAY_PORT_A="${RAY_PORT_A:-29121}"
RAY_PORT_B="${RAY_PORT_B:-29122}"
RAY_PORT_C="${RAY_PORT_C:-29123}"
RAY_ADDR_A="127.0.0.1:${RAY_PORT_A}"
RAY_ADDR_B="127.0.0.1:${RAY_PORT_B}"
RAY_ADDR_C="127.0.0.1:${RAY_PORT_C}"
RAY_WORKER_MIN_A="${RAY_WORKER_MIN_A:-11000}"
RAY_WORKER_MAX_A="${RAY_WORKER_MAX_A:-11999}"
RAY_WORKER_MIN_B="${RAY_WORKER_MIN_B:-12000}"
RAY_WORKER_MAX_B="${RAY_WORKER_MAX_B:-12999}"
RAY_WORKER_MIN_C="${RAY_WORKER_MIN_C:-13000}"
RAY_WORKER_MAX_C="${RAY_WORKER_MAX_C:-13999}"
RAY_TEMP_A="/tmp/ray_ab32x3_a_${START_TS}"
RAY_TEMP_B="/tmp/ray_ab32x3_b_${START_TS}"
RAY_TEMP_C="/tmp/ray_ab32x3_c_${START_TS}"

# Keep scheduler knobs aligned with the previous relaxed-cross AB16x2 run.
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

start_ray_head() {
  local lane="$1"
  local gpu="$2"
  local ray_port="$3"
  local dash_agent_port="$4"
  local metrics_port="$5"
  local runtime_env_port="$6"
  local dash_port="$7"
  local temp_dir="$8"
  local min_worker_port="$9"
  local max_worker_port="${10}"

  echo "[$(date '+%F %T')] RAY_START lane=${lane} gpu=${gpu} port=${ray_port}" | tee -a "${STATUS_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" ray start --head \
    --port "${ray_port}" \
    --num-cpus "${RAY_NUM_CPUS}" \
    --num-gpus 1 \
    --min-worker-port "${min_worker_port}" \
    --max-worker-port "${max_worker_port}" \
    --temp-dir "${temp_dir}" \
    --include-dashboard=False \
    --dashboard-port "${dash_port}" \
    --dashboard-agent-listen-port "${dash_agent_port}" \
    --metrics-export-port "${metrics_port}" \
    --runtime-env-agent-port "${runtime_env_port}" \
    --disable-usage-stats \
    >> "${STATUS_LOG}" 2>&1
}

run_one() {
  local lane="$1"
  local case_id="$2"
  local gpu="$3"
  local ray_addr="$4"
  local ctx="$5"
  local inf="$6"

  local run_name="doh_shift_ab32x3_${case_id}_${START_TS}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${lane}" "${case_id}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} lane=${lane} gpu=${gpu} ray=${ray_addr}" | tee -a "${STATUS_LOG}"

  if env -u RAY_ADDRESS \
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
        --ray_address "${ray_addr}" \
        --ray_namespace "${run_name}" \
        --ray_num_cpus "${RAY_NUM_CPUS}" \
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

cleanup() {
  echo "[$(date '+%F %T')] RAY_STOP_ALL" | tee -a "${STATUS_LOG}"
  ray stop --force >> "${STATUS_LOG}" 2>&1 || true
}
trap cleanup EXIT

echo "[$(date '+%F %T')] RAY_MODE explicit_heads (--ray_address)" | tee -a "${STATUS_LOG}"
ray stop --force >> "${STATUS_LOG}" 2>&1 || true

start_ray_head "A" "${GPU_A}" "${RAY_PORT_A}" 52365 60121 57121 8265 "${RAY_TEMP_A}" "${RAY_WORKER_MIN_A}" "${RAY_WORKER_MAX_A}"
start_ray_head "B" "${GPU_B}" "${RAY_PORT_B}" 52366 60122 57122 8266 "${RAY_TEMP_B}" "${RAY_WORKER_MIN_B}" "${RAY_WORKER_MAX_B}"
start_ray_head "C" "${GPU_C}" "${RAY_PORT_C}" 52367 60123 57123 8267 "${RAY_TEMP_C}" "${RAY_WORKER_MIN_C}" "${RAY_WORKER_MAX_C}"

JOBS=()
for i in "${!CASES_A[@]}"; do
  tag="$(printf "%02d" "$((i + 1))")"
  JOBS+=("A${tag},${CASES_A[$i]}")
  JOBS+=("B${tag},${CASES_B[$i]}")
done

TOTAL_CASES="${#JOBS[@]}"
TOTAL_ROUNDS="$(((TOTAL_CASES + 2) / 3))"
echo "[$(date '+%F %T')] PLAN total_cases=${TOTAL_CASES} total_rounds=${TOTAL_ROUNDS} dispatch=round_robin_3gpu" | tee -a "${STATUS_LOG}"

LANE_NAMES=("A" "B" "C")
LANE_GPUS=("${GPU_A}" "${GPU_B}" "${GPU_C}")
LANE_RAYS=("${RAY_ADDR_A}" "${RAY_ADDR_B}" "${RAY_ADDR_C}")

failures=0
cursor=0
round=0

while (( cursor < TOTAL_CASES )); do
  round=$((round + 1))
  echo "[$(date '+%F %T')] ROUND_START ${round}/${TOTAL_ROUNDS} cursor=${cursor}" | tee -a "${STATUS_LOG}"

  pids=()
  for lane_idx in 0 1 2; do
    if (( cursor >= TOTAL_CASES )); then
      break
    fi

    IFS=',' read -r case_id ctx inf <<< "${JOBS[$cursor]}"
    lane_name="${LANE_NAMES[$lane_idx]}"
    lane_gpu="${LANE_GPUS[$lane_idx]}"
    lane_ray="${LANE_RAYS[$lane_idx]}"

    run_one "${lane_name}" "${case_id}" "${lane_gpu}" "${lane_ray}" "${ctx}" "${inf}" &
    pids+=("$!")
    cursor=$((cursor + 1))
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done

  echo "[$(date '+%F %T')] ROUND_DONE ${round}/${TOTAL_ROUNDS} cumulative_failures=${failures}" | tee -a "${STATUS_LOG}"
done

if (( failures == 0 )); then
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} all_success" | tee -a "${STATUS_LOG}"
else
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} failures=${failures}" | tee -a "${STATUS_LOG}"
fi
