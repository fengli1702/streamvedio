#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cd /workspace/tream

START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ab3_trigpu_shift_focus_${START_TS}"
LOG_DIR="/workspace/tream/inference_logs"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_ab3_trigpu_shift_focus_run.txt"

mkdir -p "${LOG_DIR}"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tlane\tgpu\tray_addr\tctx\tinf\tib\ttb\tdriver_log\tstatus\n" > "${MANIFEST}"
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

# Focused trio from previous AB discussion:
# A lane: A10 (6,3)
# B lane: B01 (1,8)
# C lane: B05 (3,8)
CASE_A_CTX="${CASE_A_CTX:-6}"
CASE_A_INF="${CASE_A_INF:-3}"
CASE_B_CTX="${CASE_B_CTX:-1}"
CASE_B_INF="${CASE_B_INF:-8}"
CASE_C_CTX="${CASE_C_CTX:-3}"
CASE_C_INF="${CASE_C_INF:-8}"

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
RAY_TEMP_A="/tmp/ray_ab3_a_${START_TS}"
RAY_TEMP_B="/tmp/ray_ab3_b_${START_TS}"
RAY_TEMP_C="/tmp/ray_ab3_c_${START_TS}"

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
  local case_tag="$2"
  local gpu="$3"
  local ray_addr="$4"
  local ctx="$5"
  local inf="$6"

  local run_name="doh_shift_ab3_${lane}${case_tag}_${START_TS}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${lane}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} gpu=${gpu} ray=${ray_addr}" | tee -a "${STATUS_LOG}"

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

run_one "A" "10" "${GPU_A}" "${RAY_ADDR_A}" "${CASE_A_CTX}" "${CASE_A_INF}" &
PID_A=$!
run_one "B" "01" "${GPU_B}" "${RAY_ADDR_B}" "${CASE_B_CTX}" "${CASE_B_INF}" &
PID_B=$!
run_one "C" "05" "${GPU_C}" "${RAY_ADDR_C}" "${CASE_C_CTX}" "${CASE_C_INF}" &
PID_C=$!

wait "${PID_A}" || RC_A=$?
RC_A=${RC_A:-1}
wait "${PID_B}" || RC_B=$?
RC_B=${RC_B:-1}
wait "${PID_C}" || RC_C=$?
RC_C=${RC_C:-1}

TOTAL_FAIL=$((RC_A + RC_B + RC_C))
if (( TOTAL_FAIL == 0 )); then
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} all_success" | tee -a "${STATUS_LOG}"
else
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} failures=${TOTAL_FAIL} (A=${RC_A}, B=${RC_B}, C=${RC_C})" | tee -a "${STATUS_LOG}"
fi
