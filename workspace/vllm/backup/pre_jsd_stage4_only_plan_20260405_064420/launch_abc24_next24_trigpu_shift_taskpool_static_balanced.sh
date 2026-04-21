#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREAM_ROOT="${TREAM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="$(cd "${TREAM_ROOT}/.." && pwd)"
cd "${TREAM_ROOT}"

START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="abc24_next24x3_trigpu_shift_taskpool_static_balanced_${START_TS}"
LOG_DIR="${TREAM_ROOT}/inference_logs"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_abc24_next24x3_trigpu_shift_taskpool_static_balanced_run.txt"

mkdir -p "${LOG_DIR}"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tworker_lane\tcase_id\tgpu\tray_addr\tctx\tinf\tib\ttb\tdriver_log\tstatus\n" > "${MANIFEST}"
echo "[$(date '+%F %T')] START ${RUN_ID}" | tee -a "${STATUS_LOG}"

DATASET_DIR="${TREAM_ROOT}/data/streaming-lvm-dataset/DOH"
MODEL_PATH="${TREAM_ROOT}/lvm-llama2-7b"
SPEC_DRAFT_MODEL="../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475"
SPEC_VOCAB_MAPPING_PATH="vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt"
MAX_FRAMES="${MAX_FRAMES:-4000}"
IB="${IB:-32}"
TB="${TB:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"

# Default to currently idle cards.
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-3}"

RAY_PORT_A="${RAY_PORT_A:-29321}"
RAY_PORT_B="${RAY_PORT_B:-29322}"
RAY_PORT_C="${RAY_PORT_C:-29323}"
RAY_ADDR_A="127.0.0.1:${RAY_PORT_A}"
RAY_ADDR_B="127.0.0.1:${RAY_PORT_B}"
RAY_ADDR_C="127.0.0.1:${RAY_PORT_C}"
RAY_WORKER_MIN_A="${RAY_WORKER_MIN_A:-31000}"
RAY_WORKER_MAX_A="${RAY_WORKER_MAX_A:-31999}"
RAY_WORKER_MIN_B="${RAY_WORKER_MIN_B:-32000}"
RAY_WORKER_MAX_B="${RAY_WORKER_MAX_B:-32999}"
RAY_WORKER_MIN_C="${RAY_WORKER_MIN_C:-33000}"
RAY_WORKER_MAX_C="${RAY_WORKER_MAX_C:-33999}"

# Keep scheduler knobs aligned with relaxed-cross runs.
SHIFT_COLD_START_WINDOWS="${SHIFT_COLD_START_WINDOWS:-12}"
SHIFT_COLD_START_MAX_WINDOWS="${SHIFT_COLD_START_MAX_WINDOWS:-24}"
SHIFT_COLD_PROBE_EVERY="${SHIFT_COLD_PROBE_EVERY:-1}"
SHIFT_COLD_PROBE_EPS_FAST="${SHIFT_COLD_PROBE_EPS_FAST:-2}"
SHIFT_COLD_PROBE_EPS_SLOW="${SHIFT_COLD_PROBE_EPS_SLOW:-0}"
SHIFT_COLD_AVOID_TWO_CYCLE="${SHIFT_COLD_AVOID_TWO_CYCLE:-1}"
SHIFT_COLD_PATIENCE_DIRECTIONS="${SHIFT_COLD_PATIENCE_DIRECTIONS:-8}"
SHIFT_ACCEPT_SHOCK_DELTA="${SHIFT_ACCEPT_SHOCK_DELTA:-0.12}"
SHIFT_ACCEPT_SHOCK_COOLDOWN="${SHIFT_ACCEPT_SHOCK_COOLDOWN:-1}"
SHIFT_COLD_AXIS_ROTATION="${SHIFT_COLD_AXIS_ROTATION:-1}"
SHIFT_DEFAULT_QUALITY_MIN="${SHIFT_DEFAULT_QUALITY_MIN:-0.2}"
SHIFT_COLD_RELAX_SAFETY="${SHIFT_COLD_RELAX_SAFETY:-1}"
SHIFT_MIN_COUNT_FOR_TRUST="${SHIFT_MIN_COUNT_FOR_TRUST:-1}"
SHIFT_ADAPT_PROBE_WINDOWS="${SHIFT_ADAPT_PROBE_WINDOWS:-8}"
SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET:-12}"
SCHEDULER_QUALITY_MIN="${SCHEDULER_QUALITY_MIN:-0.2}"

echo "[$(date '+%F %T')] SHIFT_KNOBS cold_windows=${SHIFT_COLD_START_WINDOWS} cold_max_windows=${SHIFT_COLD_START_MAX_WINDOWS} cold_probe_every=${SHIFT_COLD_PROBE_EVERY} cold_probe_eps_fast=${SHIFT_COLD_PROBE_EPS_FAST} cold_probe_eps_slow=${SHIFT_COLD_PROBE_EPS_SLOW} cold_avoid_two_cycle=${SHIFT_COLD_AVOID_TWO_CYCLE} cold_patience_directions=${SHIFT_COLD_PATIENCE_DIRECTIONS} cold_axis_rotation=${SHIFT_COLD_AXIS_ROTATION} cold_relax_safety=${SHIFT_COLD_RELAX_SAFETY} cold_whitelist_budget=${SHIFT_COLD_WHITELIST_BUDGET} min_count_for_trust=${SHIFT_MIN_COUNT_FOR_TRUST} adapt_probe_windows=${SHIFT_ADAPT_PROBE_WINDOWS} default_quality_min=${SHIFT_DEFAULT_QUALITY_MIN} scheduler_quality_min=${SCHEDULER_QUALITY_MIN}" | tee -a "${STATUS_LOG}"
echo "[$(date '+%F %T')] PYTHONPATH_ROOT ${REPO_ROOT}" | tee -a "${STATUS_LOG}"

# A queue (remaining low-mid points; non-overlap with existing abc24 set)
CASES_A=(
  "1,3" "1,4" "1,5" "2,5"
  "3,1" "4,1" "4,2" "5,1"
)

# B queue (remaining medium points)
CASES_B=(
  "2,6" "3,6" "4,6" "5,6"
  "6,2" "6,3" "6,4" "6,5"
)

# C queue (remaining cross/high points)
CASES_C=(
  "1,7" "7,1" "2,7" "7,2"
  "3,7" "7,3" "4,7" "7,4"
)

start_ray_head() {
  local lane="$1"
  local gpu="$2"
  local ray_port="$3"
  local min_worker="$4"
  local max_worker="$5"
  local dash_agent_port="$6"
  local metrics_port="$7"
  local runtime_env_port="$8"
  local dash_port="$9"
  # Keep this short; Ray appends session/sockets suffixes and can exceed AF_UNIX length limit.
  local temp_dir="/tmp/rn24_${lane}_${START_TS}"

  echo "[$(date '+%F %T')] RAY_START lane=${lane} gpu=${gpu} port=${ray_port}" | tee -a "${STATUS_LOG}"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  CUDA_VISIBLE_DEVICES="${gpu}" ray start --head \
    --port "${ray_port}" \
    --num-cpus "${RAY_NUM_CPUS}" \
    --num-gpus 1 \
    --min-worker-port "${min_worker}" \
    --max-worker-port "${max_worker}" \
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
  local worker_lane="$1"
  local case_id="$2"
  local gpu="$3"
  local ray_addr="$4"
  local ctx="$5"
  local inf="$6"

  local run_name="doh_shift_abc24_next24x3_${case_id}_${START_TS}_w${worker_lane}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${worker_lane}" "${case_id}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} worker=${worker_lane} case=${case_id} gpu=${gpu} ray=${ray_addr}" | tee -a "${STATUS_LOG}"

  if env -u RAY_ADDRESS \
      PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      TREAM_SHIFT_COLD_START_WINDOWS="${SHIFT_COLD_START_WINDOWS}" \
      TREAM_SHIFT_COLD_START_MAX_WINDOWS="${SHIFT_COLD_START_MAX_WINDOWS}" \
      TREAM_SHIFT_COLD_PROBE_EVERY="${SHIFT_COLD_PROBE_EVERY}" \
      TREAM_SHIFT_COLD_PROBE_EPS_FAST="${SHIFT_COLD_PROBE_EPS_FAST}" \
      TREAM_SHIFT_COLD_PROBE_EPS_SLOW="${SHIFT_COLD_PROBE_EPS_SLOW}" \
      TREAM_SHIFT_COLD_AVOID_TWO_CYCLE="${SHIFT_COLD_AVOID_TWO_CYCLE}" \
      TREAM_SHIFT_COLD_PATIENCE_DIRECTIONS="${SHIFT_COLD_PATIENCE_DIRECTIONS}" \
      TREAM_SHIFT_ACCEPT_SHOCK_DELTA="${SHIFT_ACCEPT_SHOCK_DELTA}" \
      TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN="${SHIFT_ACCEPT_SHOCK_COOLDOWN}" \
      TREAM_SHIFT_COLD_AXIS_ROTATION="${SHIFT_COLD_AXIS_ROTATION}" \
      TREAM_SHIFT_DEFAULT_QUALITY_MIN="${SHIFT_DEFAULT_QUALITY_MIN}" \
      TREAM_SHIFT_COLD_RELAX_SAFETY="${SHIFT_COLD_RELAX_SAFETY}" \
      TREAM_SHIFT_MIN_COUNT_FOR_TRUST="${SHIFT_MIN_COUNT_FOR_TRUST}" \
      TREAM_SHIFT_ADAPT_PROBE_WINDOWS="${SHIFT_ADAPT_PROBE_WINDOWS}" \
      TREAM_SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET}" \
      python "${TREAM_ROOT}/tream.py" \
        --input_frames_path "${DATASET_DIR}" \
        --context_length "${ctx}" \
        --inference_length "${inf}" \
        --disable_dynamic_scheduling \
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

# Fixed balanced assignment:
# - 8 cases per lane
# - C (large/cross) workload split 3/2/3
# - sum(ctx*inf) balanced to 116/117/117
LANE_A_SPECS=(
  "A03,1,5" "A05,3,1" "B03,4,6" "B06,6,3"
  "B07,6,4" "C01,1,7" "C04,7,2" "C05,3,7"
)
LANE_B_SPECS=(
  "A01,1,3" "A02,1,4" "A07,4,2" "B02,3,6"
  "B05,6,2" "B08,6,5" "C03,2,7" "C07,4,7"
)
LANE_C_SPECS=(
  "A04,2,5" "A06,4,1" "A08,5,1" "B01,2,6"
  "B04,5,6" "C02,7,1" "C06,7,3" "C08,7,4"
)
TOTAL_CASES=$(( ${#LANE_A_SPECS[@]} + ${#LANE_B_SPECS[@]} + ${#LANE_C_SPECS[@]} ))

lane_worker() {
  local lane="$1"
  local gpu="$2"
  local ray_addr="$3"
  shift 3
  local specs=("$@")
  local failures=0
  local picked=0
  local spec case_id ctx inf

  for spec in "${specs[@]}"; do
    IFS=',' read -r case_id ctx inf <<< "${spec}"
    picked=$((picked + 1))
    echo "[$(date '+%F %T')] WORKER_PICK lane=${lane} case=${case_id} ctx=${ctx} inf=${inf}" | tee -a "${STATUS_LOG}"

    if ! run_one "${lane}" "${case_id}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}"; then
      failures=$((failures + 1))
    fi
  done

  echo "[$(date '+%F %T')] WORKER_IDLE lane=${lane} picked=${picked} failures=${failures}" | tee -a "${STATUS_LOG}"

  return "${failures}"
}

cleanup() {
  echo "[$(date '+%F %T')] RAY_STOP_ALL" | tee -a "${STATUS_LOG}"
  ray stop --force >> "${STATUS_LOG}" 2>&1 || true
}
trap cleanup EXIT

echo "[$(date '+%F %T')] PLAN total_cases=${TOTAL_CASES} dispatch=fixed_balanced workers=3 dynamic=off" | tee -a "${STATUS_LOG}"
echo "[$(date '+%F %T')] BALANCE laneA=${#LANE_A_SPECS[@]} laneB=${#LANE_B_SPECS[@]} laneC=${#LANE_C_SPECS[@]} metric=ctx_x_inf_equalized" | tee -a "${STATUS_LOG}"
echo "[$(date '+%F %T')] GPU_INIT A=${GPU_A} B=${GPU_B} C=${GPU_C}" | tee -a "${STATUS_LOG}"

ray stop --force >> "${STATUS_LOG}" 2>&1 || true
start_ray_head "A" "${GPU_A}" "${RAY_PORT_A}" "${RAY_WORKER_MIN_A}" "${RAY_WORKER_MAX_A}" 54365 62121 59121 8465
start_ray_head "B" "${GPU_B}" "${RAY_PORT_B}" "${RAY_WORKER_MIN_B}" "${RAY_WORKER_MAX_B}" 54366 62122 59122 8466
start_ray_head "C" "${GPU_C}" "${RAY_PORT_C}" "${RAY_WORKER_MIN_C}" "${RAY_WORKER_MAX_C}" 54367 62123 59123 8467

lane_worker "A" "${GPU_A}" "${RAY_ADDR_A}" "${LANE_A_SPECS[@]}" &
PID_A=$!
lane_worker "B" "${GPU_B}" "${RAY_ADDR_B}" "${LANE_B_SPECS[@]}" &
PID_B=$!
lane_worker "C" "${GPU_C}" "${RAY_ADDR_C}" "${LANE_C_SPECS[@]}" &
PID_C=$!

wait "${PID_A}" || RC_A=$?
RC_A=${RC_A:-0}
wait "${PID_B}" || RC_B=$?
RC_B=${RC_B:-0}
wait "${PID_C}" || RC_C=$?
RC_C=${RC_C:-0}

TOTAL_FAIL=$((RC_A + RC_B + RC_C))
if (( TOTAL_FAIL == 0 )); then
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} all_success" | tee -a "${STATUS_LOG}"
else
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} failures=${TOTAL_FAIL} (A=${RC_A}, B=${RC_B}, C=${RC_C})" | tee -a "${STATUS_LOG}"
fi
