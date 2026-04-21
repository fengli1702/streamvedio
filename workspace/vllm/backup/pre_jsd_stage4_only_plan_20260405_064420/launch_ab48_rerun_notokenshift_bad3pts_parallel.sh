#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREAM_ROOT="${TREAM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${TREAM_ROOT}"

RUN_MODE="${RUN_MODE:-dynamic}"   # dynamic | static
if [[ "${RUN_MODE}" != "dynamic" && "${RUN_MODE}" != "static" ]]; then
  echo "RUN_MODE must be dynamic or static, got: ${RUN_MODE}" >&2
  exit 1
fi

START_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${TREAM_ROOT}/inference_logs"
mkdir -p "${LOG_DIR}"

DATASET_DIR="${TREAM_ROOT}/data/streaming-lvm-dataset/DOH"
MODEL_PATH="${TREAM_ROOT}/lvm-llama2-7b"
SPEC_DRAFT_MODEL="../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475"
SPEC_VOCAB_MAPPING_PATH="vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt"
MAX_FRAMES="${MAX_FRAMES:-4000}"
IB="${IB:-32}"
TB="${TB:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"

# 3 GPUs
GPU_A="${GPU_A:-4}"
GPU_B="${GPU_B:-5}"
GPU_C="${GPU_C:-6}"

RAY_PORT_A="${RAY_PORT_A:-29421}"
RAY_PORT_B="${RAY_PORT_B:-29422}"
RAY_PORT_C="${RAY_PORT_C:-29423}"
RAY_ADDR_A="127.0.0.1:${RAY_PORT_A}"
RAY_ADDR_B="127.0.0.1:${RAY_PORT_B}"
RAY_ADDR_C="127.0.0.1:${RAY_PORT_C}"
RAY_WORKER_MIN_A="${RAY_WORKER_MIN_A:-34000}"
RAY_WORKER_MAX_A="${RAY_WORKER_MAX_A:-34999}"
RAY_WORKER_MIN_B="${RAY_WORKER_MIN_B:-35000}"
RAY_WORKER_MAX_B="${RAY_WORKER_MAX_B:-35999}"
RAY_WORKER_MIN_C="${RAY_WORKER_MIN_C:-36000}"
RAY_WORKER_MAX_C="${RAY_WORKER_MAX_C:-36999}"

# Scheduler knobs (dynamic mode only; still logged for static naming)
SHIFT_COLD_START_WINDOWS="${SHIFT_COLD_START_WINDOWS:-15}"
SHIFT_COLD_START_MAX_WINDOWS="${SHIFT_COLD_START_MAX_WINDOWS:-15}"
SHIFT_COLD_PROBE_EVERY="${SHIFT_COLD_PROBE_EVERY:-1}"
SHIFT_COLD_PROBE_EPS_FAST="${SHIFT_COLD_PROBE_EPS_FAST:-2}"
SHIFT_COLD_PROBE_EPS_SLOW="${SHIFT_COLD_PROBE_EPS_SLOW:-0}"
SHIFT_COLD_AVOID_TWO_CYCLE="${SHIFT_COLD_AVOID_TWO_CYCLE:-1}"
SHIFT_COLD_DIRECTIONAL_ENABLE="${SHIFT_COLD_DIRECTIONAL_ENABLE:-1}"
SHIFT_COLD_PREFER_LARGE_DESCEND="${SHIFT_COLD_PREFER_LARGE_DESCEND:-1}"
SHIFT_COLD_PREFER_SMALL_ASCEND="${SHIFT_COLD_PREFER_SMALL_ASCEND:-1}"
SHIFT_COLD_LARGE_CTX_THRESHOLD="${SHIFT_COLD_LARGE_CTX_THRESHOLD:-6}"
SHIFT_COLD_LARGE_INF_THRESHOLD="${SHIFT_COLD_LARGE_INF_THRESHOLD:-6}"
SHIFT_COLD_SMALL_CTX_THRESHOLD="${SHIFT_COLD_SMALL_CTX_THRESHOLD:-1}"
SHIFT_COLD_SMALL_INF_THRESHOLD="${SHIFT_COLD_SMALL_INF_THRESHOLD:-1}"
SHIFT_COLD_TARGET_CTX="${SHIFT_COLD_TARGET_CTX:-2}"
SHIFT_COLD_TARGET_INF="${SHIFT_COLD_TARGET_INF:-2}"
SHIFT_COLD_TARGET_CTX_BAND="${SHIFT_COLD_TARGET_CTX_BAND:-1}"
SHIFT_COLD_TARGET_INF_BAND="${SHIFT_COLD_TARGET_INF_BAND:-1}"
SHIFT_COLD_SINGLE_AXIS_LOCK="${SHIFT_COLD_SINGLE_AXIS_LOCK:-1}"
SHIFT_COLD_EARLY_EXIT_ENABLE="${SHIFT_COLD_EARLY_EXIT_ENABLE:-1}"
SHIFT_COLD_TARGET_STABLE_WINDOWS="${SHIFT_COLD_TARGET_STABLE_WINDOWS:-2}"
SHIFT_COLD_PATIENCE_DIRECTIONS="${SHIFT_COLD_PATIENCE_DIRECTIONS:-8}"
SHIFT_ACCEPT_SHOCK_DELTA="${SHIFT_ACCEPT_SHOCK_DELTA:-0.12}"
SHIFT_ACCEPT_SHOCK_COOLDOWN="${SHIFT_ACCEPT_SHOCK_COOLDOWN:-1}"
SHIFT_COLD_AXIS_ROTATION="${SHIFT_COLD_AXIS_ROTATION:-1}"
SHIFT_DEFAULT_QUALITY_MIN="${SHIFT_DEFAULT_QUALITY_MIN:-0.2}"
SHIFT_COLD_RELAX_SAFETY="${SHIFT_COLD_RELAX_SAFETY:-1}"
SHIFT_COLD_RELAX_RADIUS="${SHIFT_COLD_RELAX_RADIUS:-2}"
SHIFT_COLD_RELAX_Q_SLACK="${SHIFT_COLD_RELAX_Q_SLACK:-0.25}"
SHIFT_COLD_RELAX_LAT_SLACK="${SHIFT_COLD_RELAX_LAT_SLACK:-0.15}"
SHIFT_MIN_COUNT_FOR_TRUST="${SHIFT_MIN_COUNT_FOR_TRUST:-1}"
SHIFT_ADAPT_PROBE_WINDOWS="${SHIFT_ADAPT_PROBE_WINDOWS:-8}"
SHIFT_COLD_WHITELIST_PROBE="${SHIFT_COLD_WHITELIST_PROBE:-1}"
SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET:-12}"
SHIFT_COLD_WHITELIST_LAT_SLACK="${SHIFT_COLD_WHITELIST_LAT_SLACK:-0.15}"
SHIFT_COLD_WHITELIST_Q_SLACK="${SHIFT_COLD_WHITELIST_Q_SLACK:-0.60}"
SHIFT_COLD_WHITELIST_MAX_SWITCH="${SHIFT_COLD_WHITELIST_MAX_SWITCH:-2}"
SHIFT_SHOCK_USE_JSD_ONLY="${SHIFT_SHOCK_USE_JSD_ONLY:-1}"
SHIFT_SHOCK_DISABLE_DRIFT="${SHIFT_SHOCK_DISABLE_DRIFT:-0}"
SCHEDULER_QUALITY_MIN="${SCHEDULER_QUALITY_MIN:-0.2}"

if [[ "${RUN_MODE}" == "dynamic" ]]; then
  MODE_TAG="dyn_wb${SHIFT_COLD_WHITELIST_PROBE}b${SHIFT_COLD_WHITELIST_BUDGET}j${SHIFT_COLD_WHITELIST_MAX_SWITCH}_jsd${SHIFT_SHOCK_USE_JSD_ONLY}td${SHIFT_SHOCK_DISABLE_DRIFT}_cw${SHIFT_COLD_START_WINDOWS}"
else
  MODE_TAG="static_nodyn_ib${IB}_tb${TB}"
fi

RUN_ID="ab48x3_trigpu_shift_taskpool_${MODE_TAG}_${START_TS}"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_ab48x3_trigpu_shift_taskpool_${RUN_MODE}_run.txt"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tworker_lane\tcase_id\tgpu\tray_addr\tctx\tinf\tib\ttb\tdriver_log\tstatus\n" > "${MANIFEST}"
echo "[$(date '+%F %T')] START ${RUN_ID}" | tee -a "${STATUS_LOG}"
echo "[$(date '+%F %T')] MODE ${RUN_MODE} mode_tag=${MODE_TAG}" | tee -a "${STATUS_LOG}"
echo "[$(date '+%F %T')] GPU_INIT A=${GPU_A} B=${GPU_B} C=${GPU_C}" | tee -a "${STATUS_LOG}"

echo "[$(date '+%F %T')] SHIFT_KNOBS whitelist_probe=${SHIFT_COLD_WHITELIST_PROBE} whitelist_budget=${SHIFT_COLD_WHITELIST_BUDGET} whitelist_jump=${SHIFT_COLD_WHITELIST_MAX_SWITCH} whitelist_slack_lat=${SHIFT_COLD_WHITELIST_LAT_SLACK} whitelist_slack_q=${SHIFT_COLD_WHITELIST_Q_SLACK} shock_jsd_only=${SHIFT_SHOCK_USE_JSD_ONLY} shock_disable_drift=${SHIFT_SHOCK_DISABLE_DRIFT} cold_windows=${SHIFT_COLD_START_WINDOWS}/${SHIFT_COLD_START_MAX_WINDOWS}" | tee -a "${STATUS_LOG}"

CASES_A=(
  "1,1" "1,2" "2,1" "2,2"
  "2,3" "3,2" "3,3" "2,4"
)
CASES_B=(
  "3,4" "4,3" "4,4" "3,5"
  "5,3" "4,5" "5,4" "5,5"
)
CASES_C=(
  "1,8" "8,1" "2,8" "8,2"
  "3,8" "8,3" "4,8" "8,4"
)
CASES_D=(
  "6,1" "1,6" "6,2" "2,6"
  "6,3" "3,6" "6,4" "4,6"
)
CASES_E=(
  "6,5" "5,6" "6,6" "7,1"
  "1,7" "7,2" "2,7" "7,3"
)
CASES_F=(
  "3,7" "7,4" "4,7" "7,5"
  "5,7" "7,6" "6,7" "7,7"
)

start_ray_head() {
  local lane="$1"; local gpu="$2"; local ray_port="$3"; local min_worker="$4"; local max_worker="$5"; local dash_agent_port="$6"; local metrics_port="$7"; local runtime_env_port="$8"; local dash_port="$9"
  local temp_dir="/tmp/ray_ab48x3_${lane}_${START_TS}"
  echo "[$(date '+%F %T')] RAY_START lane=${lane} gpu=${gpu} port=${ray_port}" | tee -a "${STATUS_LOG}"
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
  local worker_lane="$1"; local case_id="$2"; local gpu="$3"; local ray_addr="$4"; local ctx="$5"; local inf="$6"
  local run_name="doh_shift_ab48x3_${MODE_TAG}_${case_id}_${START_TS}_w${worker_lane}_c${ctx}_i${inf}_ib${IB}_tb${TB}"
  local driver_log="${LOG_DIR}/${run_name}.driver.log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${worker_lane}" "${case_id}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}" "${IB}" "${TB}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} worker=${worker_lane} case=${case_id} gpu=${gpu} ray=${ray_addr}" | tee -a "${STATUS_LOG}"

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
    --ray_address "${ray_addr}"
    --ray_namespace "${run_name}"
    --ray_num_cpus "${RAY_NUM_CPUS}"
    --use_speculative_decoding
    --spec_draft_model "${SPEC_DRAFT_MODEL}"
    --spec_vocab_mapping_path "${SPEC_VOCAB_MAPPING_PATH}"
    --num_spec_tokens 3
    --spec_disable_mqa_scorer
    -gc
    -train
  )

  if [[ "${RUN_MODE}" == "static" ]]; then
    cmd+=(--disable_dynamic_scheduling)
  fi

  if [[ "${RUN_MODE}" == "dynamic" ]]; then
    if env -u RAY_ADDRESS \
      PYTHONPATH="${TREAM_ROOT}/..:${PYTHONPATH:-}" \
      TREAM_SHIFT_COLD_START_WINDOWS="${SHIFT_COLD_START_WINDOWS}" \
      TREAM_SHIFT_COLD_START_MAX_WINDOWS="${SHIFT_COLD_START_MAX_WINDOWS}" \
      TREAM_SHIFT_COLD_PROBE_EVERY="${SHIFT_COLD_PROBE_EVERY}" \
      TREAM_SHIFT_COLD_PROBE_EPS_FAST="${SHIFT_COLD_PROBE_EPS_FAST}" \
      TREAM_SHIFT_COLD_PROBE_EPS_SLOW="${SHIFT_COLD_PROBE_EPS_SLOW}" \
      TREAM_SHIFT_COLD_AVOID_TWO_CYCLE="${SHIFT_COLD_AVOID_TWO_CYCLE}" \
      TREAM_SHIFT_COLD_DIRECTIONAL_ENABLE="${SHIFT_COLD_DIRECTIONAL_ENABLE}" \
      TREAM_SHIFT_COLD_PREFER_LARGE_DESCEND="${SHIFT_COLD_PREFER_LARGE_DESCEND}" \
      TREAM_SHIFT_COLD_PREFER_SMALL_ASCEND="${SHIFT_COLD_PREFER_SMALL_ASCEND}" \
      TREAM_SHIFT_COLD_LARGE_CTX_THRESHOLD="${SHIFT_COLD_LARGE_CTX_THRESHOLD}" \
      TREAM_SHIFT_COLD_LARGE_INF_THRESHOLD="${SHIFT_COLD_LARGE_INF_THRESHOLD}" \
      TREAM_SHIFT_COLD_SMALL_CTX_THRESHOLD="${SHIFT_COLD_SMALL_CTX_THRESHOLD}" \
      TREAM_SHIFT_COLD_SMALL_INF_THRESHOLD="${SHIFT_COLD_SMALL_INF_THRESHOLD}" \
      TREAM_SHIFT_COLD_TARGET_CTX="${SHIFT_COLD_TARGET_CTX}" \
      TREAM_SHIFT_COLD_TARGET_INF="${SHIFT_COLD_TARGET_INF}" \
      TREAM_SHIFT_COLD_TARGET_CTX_BAND="${SHIFT_COLD_TARGET_CTX_BAND}" \
      TREAM_SHIFT_COLD_TARGET_INF_BAND="${SHIFT_COLD_TARGET_INF_BAND}" \
      TREAM_SHIFT_COLD_SINGLE_AXIS_LOCK="${SHIFT_COLD_SINGLE_AXIS_LOCK}" \
      TREAM_SHIFT_COLD_EARLY_EXIT_ENABLE="${SHIFT_COLD_EARLY_EXIT_ENABLE}" \
      TREAM_SHIFT_COLD_TARGET_STABLE_WINDOWS="${SHIFT_COLD_TARGET_STABLE_WINDOWS}" \
      TREAM_SHIFT_COLD_PATIENCE_DIRECTIONS="${SHIFT_COLD_PATIENCE_DIRECTIONS}" \
      TREAM_SHIFT_ACCEPT_SHOCK_DELTA="${SHIFT_ACCEPT_SHOCK_DELTA}" \
      TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN="${SHIFT_ACCEPT_SHOCK_COOLDOWN}" \
      TREAM_SHIFT_COLD_AXIS_ROTATION="${SHIFT_COLD_AXIS_ROTATION}" \
      TREAM_SHIFT_DEFAULT_QUALITY_MIN="${SHIFT_DEFAULT_QUALITY_MIN}" \
      TREAM_SHIFT_COLD_RELAX_SAFETY="${SHIFT_COLD_RELAX_SAFETY}" \
      TREAM_SHIFT_COLD_RELAX_RADIUS="${SHIFT_COLD_RELAX_RADIUS}" \
      TREAM_SHIFT_COLD_RELAX_Q_SLACK="${SHIFT_COLD_RELAX_Q_SLACK}" \
      TREAM_SHIFT_COLD_RELAX_LAT_SLACK="${SHIFT_COLD_RELAX_LAT_SLACK}" \
      TREAM_SHIFT_MIN_COUNT_FOR_TRUST="${SHIFT_MIN_COUNT_FOR_TRUST}" \
      TREAM_SHIFT_ADAPT_PROBE_WINDOWS="${SHIFT_ADAPT_PROBE_WINDOWS}" \
      TREAM_SHIFT_COLD_WHITELIST_PROBE="${SHIFT_COLD_WHITELIST_PROBE}" \
      TREAM_SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET}" \
      TREAM_SHIFT_COLD_WHITELIST_LAT_SLACK="${SHIFT_COLD_WHITELIST_LAT_SLACK}" \
      TREAM_SHIFT_COLD_WHITELIST_Q_SLACK="${SHIFT_COLD_WHITELIST_Q_SLACK}" \
      TREAM_SHIFT_COLD_WHITELIST_MAX_SWITCH="${SHIFT_COLD_WHITELIST_MAX_SWITCH}" \
      TREAM_SHIFT_SHOCK_USE_JSD_ONLY="${SHIFT_SHOCK_USE_JSD_ONLY}" \
      TREAM_SHIFT_SHOCK_DISABLE_DRIFT="${SHIFT_SHOCK_DISABLE_DRIFT}" \
      "${cmd[@]}" > "${driver_log}" 2>&1; then
      echo "[$(date '+%F %T')] DONE ${run_name} rc=0" | tee -a "${STATUS_LOG}"
      return 0
    fi
  else
    if env -u RAY_ADDRESS \
      PYTHONPATH="${TREAM_ROOT}/..:${PYTHONPATH:-}" \
      "${cmd[@]}" > "${driver_log}" 2>&1; then
      echo "[$(date '+%F %T')] DONE ${run_name} rc=0" | tee -a "${STATUS_LOG}"
      return 0
    fi
  fi

  local rc=$?
  echo "[$(date '+%F %T')] FAIL ${run_name} rc=${rc}" | tee -a "${STATUS_LOG}"
  return "${rc}"
}

JOBS=("E07,2,7")
TOTAL_CASES=1
QUEUE_LOCK="/tmp/${RUN_ID}.queue.lock"
CURSOR_FILE="/tmp/${RUN_ID}.cursor"
echo 0 > "${CURSOR_FILE}"

pop_next_job() {
  local item=""; local idx
  { flock -x 9
    idx="$(cat "${CURSOR_FILE}")"
    if (( idx < TOTAL_CASES )); then
      item="${JOBS[$idx]}"
      echo $((idx + 1)) > "${CURSOR_FILE}"
    fi
  } 9>"${QUEUE_LOCK}"
  printf "%s" "${item}"
}

lane_worker() {
  local lane="$1"; local gpu="$2"; local ray_addr="$3"
  local failures=0 picked=0 spec case_id ctx inf
  while true; do
    spec="$(pop_next_job)"
    if [ -z "${spec}" ]; then
      echo "[$(date '+%F %T')] WORKER_IDLE lane=${lane} picked=${picked} failures=${failures}" | tee -a "${STATUS_LOG}"
      break
    fi
    IFS=',' read -r case_id ctx inf <<< "${spec}"
    picked=$((picked + 1))
    echo "[$(date '+%F %T')] WORKER_PICK lane=${lane} case=${case_id} ctx=${ctx} inf=${inf}" | tee -a "${STATUS_LOG}"
    if ! run_one "${lane}" "${case_id}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}"; then
      failures=$((failures + 1))
    fi
  done
  return "${failures}"
}

cleanup() {
  if [[ "${NO_GLOBAL_RAY_STOP:-0}" == "1" ]]; then
    echo "[$(date '+%F %T')] SKIP_RAY_STOP NO_GLOBAL_RAY_STOP=1" | tee -a "${STATUS_LOG}"
  else
    echo "[$(date '+%F %T')] RAY_STOP_ALL" | tee -a "${STATUS_LOG}"
    ray stop --force >> "${STATUS_LOG}" 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${NO_GLOBAL_RAY_STOP:-0}" != "1" ]]; then
  ray stop --force >> "${STATUS_LOG}" 2>&1 || true
fi

start_ray_head "A" "${GPU_A}" "${RAY_PORT_A}" "${RAY_WORKER_MIN_A}" "${RAY_WORKER_MAX_A}" 55465 63221 60221 8665
start_ray_head "B" "${GPU_B}" "${RAY_PORT_B}" "${RAY_WORKER_MIN_B}" "${RAY_WORKER_MAX_B}" 55466 63222 60222 8666
start_ray_head "C" "${GPU_C}" "${RAY_PORT_C}" "${RAY_WORKER_MIN_C}" "${RAY_WORKER_MAX_C}" 55467 63223 60223 8667

echo "[$(date '+%F %T')] PLAN total_cases=${TOTAL_CASES} dispatch=task_pool workers=3" | tee -a "${STATUS_LOG}"

lane_worker "A" "${GPU_A}" "${RAY_ADDR_A}" & PID_A=$!
lane_worker "B" "${GPU_B}" "${RAY_ADDR_B}" & PID_B=$!
lane_worker "C" "${GPU_C}" "${RAY_ADDR_C}" & PID_C=$!

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
