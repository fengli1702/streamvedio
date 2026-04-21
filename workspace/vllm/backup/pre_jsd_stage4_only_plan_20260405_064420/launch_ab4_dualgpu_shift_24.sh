#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cd /workspace/tream

START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ab4_dualgpu_shift_${START_TS}"
LOG_DIR="/workspace/tream/inference_logs"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_ab4_dualgpu_shift_run.txt"

mkdir -p "${LOG_DIR}"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tqueue\tgpu\tray_addr\tctx\tinf\tib\ttb\tdriver_log\tstatus\n" > "${MANIFEST}"

echo "[$(date '+%F %T')] START ${RUN_ID}" | tee -a "${STATUS_LOG}"

DATASET_DIR="/workspace/tream/data/streaming-lvm-dataset/DOH"
MODEL_PATH="/workspace/tream/lvm-llama2-7b"
SPEC_DRAFT_MODEL="../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475"
SPEC_VOCAB_MAPPING_PATH="vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt"
MAX_FRAMES=4000
IB=32
TB=16

RAY_ADDR_A="127.0.0.1:28121"
RAY_ADDR_B="127.0.0.1:28122"
RAY_TEMP_A="/tmp/ray_ab4_a_${START_TS}"
RAY_TEMP_B="/tmp/ray_ab4_b_${START_TS}"

start_ray_head() {
  local queue="$1"
  local gpu="$2"
  local ray_port="$3"
  local dash_agent_port="$4"
  local metrics_port="$5"
  local runtime_env_port="$6"
  local dash_port="$7"
  local temp_dir="$8"

  echo "[$(date '+%F %T')] RAY_START queue=${queue} gpu=${gpu} port=${ray_port} dash_agent=${dash_agent_port}" | tee -a "${STATUS_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" ray start --head \
    --port "${ray_port}" \
    --num-cpus 16 \
    --num-gpus 1 \
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
  local queue="$1"
  local tag="$2"
  local ray_addr="$3"
  local ctx="$4"
  local inf="$5"

  local run_name="doh_shift_ab4_${queue}${tag}_${START_TS}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${queue}" "-" "${ray_addr}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} ray=${ray_addr}" | tee -a "${STATUS_LOG}"

  if TREAM_SHIFT_COLD_START_WINDOWS=8 \
      TREAM_SHIFT_COLD_PROBE_EVERY=2 \
      TREAM_SHIFT_ACCEPT_SHOCK_DELTA=0.12 \
      TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN=1 \
      TREAM_SHIFT_COLD_AXIS_ROTATION=1 \
      TREAM_SHIFT_DEFAULT_QUALITY_MIN=0.5 \
      python /workspace/tream/tream.py \
        --input_frames_path "${DATASET_DIR}" \
        --context_length "${ctx}" \
        --inference_length "${inf}" \
        --inference_batch_size "${IB}" \
        --training_batch_size "${TB}" \
        --model_name "${MODEL_PATH}" \
        --num_workers 1 \
        --max_frames "${MAX_FRAMES}" \
        --gpu_memory_utilization 0.95 \
        --max_loras 1 \
        --scheduler_quality_min 0.5 \
        --wandb_run_name "${run_name}" \
        --inference_logs_dir "${LOG_DIR}" \
        --ray_address "${ray_addr}" \
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

queue_a() {
  local failures=0
  run_one "A" "1" "${RAY_ADDR_A}" 1 1 || failures=$((failures + 1))
  run_one "A" "2" "${RAY_ADDR_A}" 2 2 || failures=$((failures + 1))
  run_one "A" "3" "${RAY_ADDR_A}" 2 4 || failures=$((failures + 1))
  run_one "A" "4" "${RAY_ADDR_A}" 4 4 || failures=$((failures + 1))
  return "${failures}"
}

queue_b() {
  echo "[$(date '+%F %T')] queue_b warmup sleep 30s" | tee -a "${STATUS_LOG}"
  sleep 30
  local failures=0
  run_one "B" "1" "${RAY_ADDR_B}" 1 8 || failures=$((failures + 1))
  run_one "B" "2" "${RAY_ADDR_B}" 2 8 || failures=$((failures + 1))
  run_one "B" "3" "${RAY_ADDR_B}" 4 6 || failures=$((failures + 1))
  run_one "B" "4" "${RAY_ADDR_B}" 6 6 || failures=$((failures + 1))
  return "${failures}"
}

cleanup() {
  echo "[$(date '+%F %T')] RAY_STOP_ALL" | tee -a "${STATUS_LOG}"
  ray stop --force >> "${STATUS_LOG}" 2>&1 || true
}
trap cleanup EXIT

start_ray_head "A" 2 28121 52365 60121 57121 8265 "${RAY_TEMP_A}"
start_ray_head "B" 4 28122 52366 60122 57122 8266 "${RAY_TEMP_B}"

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
