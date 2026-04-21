#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cd /workspace/tream

START_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ab32x3_trigpu_shift_autofix_${START_TS}"
LOG_DIR="/workspace/tream/inference_logs"
MANIFEST="${LOG_DIR}/${RUN_ID}.tsv"
STATUS_LOG="${LOG_DIR}/${RUN_ID}.status.log"
LATEST_PTR="${LOG_DIR}/latest_ab32x3_trigpu_shift_autofix_run.txt"

mkdir -p "${LOG_DIR}"

echo "${RUN_ID}" > "${LATEST_PTR}"
printf "run_name\tlane\tcase_id\tattempt\tgpu\tray_addr\tctx\tinf\tib\ttb\tgpu_mem_util\twandb_mode\tdriver_log\tstatus\n" > "${MANIFEST}"
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

# Default to currently idle GPUs.
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-3}"
GPU_POOL="${GPU_POOL:-0,1,2,3,4,5,7}"

# Auto-fix knobs.
MAX_RETRIES="${MAX_RETRIES:-2}"
MEM_UTIL_STEP="${MEM_UTIL_STEP:-0.08}"
MEM_UTIL_FLOOR="${MEM_UTIL_FLOOR:-0.70}"
WANDB_FALLBACK_OFFLINE="${WANDB_FALLBACK_OFFLINE:-1}"

RAY_PORT_A="${RAY_PORT_A:-29221}"
RAY_PORT_B="${RAY_PORT_B:-29222}"
RAY_PORT_C="${RAY_PORT_C:-29223}"
RAY_ADDR_A="127.0.0.1:${RAY_PORT_A}"
RAY_ADDR_B="127.0.0.1:${RAY_PORT_B}"
RAY_ADDR_C="127.0.0.1:${RAY_PORT_C}"
RAY_WORKER_MIN_A="${RAY_WORKER_MIN_A:-21000}"
RAY_WORKER_MAX_A="${RAY_WORKER_MAX_A:-21999}"
RAY_WORKER_MIN_B="${RAY_WORKER_MIN_B:-22000}"
RAY_WORKER_MAX_B="${RAY_WORKER_MAX_B:-22999}"
RAY_WORKER_MIN_C="${RAY_WORKER_MIN_C:-23000}"
RAY_WORKER_MAX_C="${RAY_WORKER_MAX_C:-23999}"

# Keep scheduler knobs aligned with AB16 relaxed-cross runs.
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

LANE_NAMES=("A" "B" "C")
IFS=',' read -r -a GPU_POOL_ARR <<< "${GPU_POOL}"

declare -A LANE_GPU
declare -A LANE_MEM_UTIL
declare -A LANE_WANDB_MODE
declare -A LANE_RAY_PORT
declare -A LANE_RAY_ADDR
declare -A LANE_MIN_WORKER
declare -A LANE_MAX_WORKER
declare -A LANE_DASH_AGENT
declare -A LANE_METRICS
declare -A LANE_RUNTIME
declare -A LANE_DASH

LANE_GPU["A"]="${GPU_A}"
LANE_GPU["B"]="${GPU_B}"
LANE_GPU["C"]="${GPU_C}"
LANE_MEM_UTIL["A"]="${GPU_MEM_UTIL}"
LANE_MEM_UTIL["B"]="${GPU_MEM_UTIL}"
LANE_MEM_UTIL["C"]="${GPU_MEM_UTIL}"
LANE_WANDB_MODE["A"]="online"
LANE_WANDB_MODE["B"]="online"
LANE_WANDB_MODE["C"]="online"

LANE_RAY_PORT["A"]="${RAY_PORT_A}"
LANE_RAY_PORT["B"]="${RAY_PORT_B}"
LANE_RAY_PORT["C"]="${RAY_PORT_C}"
LANE_RAY_ADDR["A"]="${RAY_ADDR_A}"
LANE_RAY_ADDR["B"]="${RAY_ADDR_B}"
LANE_RAY_ADDR["C"]="${RAY_ADDR_C}"
LANE_MIN_WORKER["A"]="${RAY_WORKER_MIN_A}"
LANE_MIN_WORKER["B"]="${RAY_WORKER_MIN_B}"
LANE_MIN_WORKER["C"]="${RAY_WORKER_MIN_C}"
LANE_MAX_WORKER["A"]="${RAY_WORKER_MAX_A}"
LANE_MAX_WORKER["B"]="${RAY_WORKER_MAX_B}"
LANE_MAX_WORKER["C"]="${RAY_WORKER_MAX_C}"
LANE_DASH_AGENT["A"]=53365
LANE_DASH_AGENT["B"]=53366
LANE_DASH_AGENT["C"]=53367
LANE_METRICS["A"]=61121
LANE_METRICS["B"]=61122
LANE_METRICS["C"]=61123
LANE_RUNTIME["A"]=58121
LANE_RUNTIME["B"]=58122
LANE_RUNTIME["C"]=58123
LANE_DASH["A"]=8365
LANE_DASH["B"]=8366
LANE_DASH["C"]=8367

build_run_name() {
  local lane="$1"
  local case_id="$2"
  local ctx="$3"
  local inf="$4"
  local attempt="$5"
  printf "doh_shift_ab32x3_%s_%s_%s_c%s_i%s_ib%s_tb%s_try%s" \
    "${lane}" "${case_id}" "${START_TS}" "${ctx}" "${inf}" "${IB}" "${TB}" "${attempt}"
}

driver_log_path() {
  local run_name="$1"
  printf "%s/%s.driver.log" "${LOG_DIR}" "${run_name}"
}

stop_all_ray() {
  ray stop --force >> "${STATUS_LOG}" 2>&1 || true
}

start_ray_head() {
  local lane="$1"
  local gpu="${LANE_GPU[${lane}]}"
  local ray_port="${LANE_RAY_PORT[${lane}]}"
  local min_worker="${LANE_MIN_WORKER[${lane}]}"
  local max_worker="${LANE_MAX_WORKER[${lane}]}"
  local dash_agent="${LANE_DASH_AGENT[${lane}]}"
  local metrics="${LANE_METRICS[${lane}]}"
  local runtime="${LANE_RUNTIME[${lane}]}"
  local dash="${LANE_DASH[${lane}]}"
  local temp_dir="/tmp/ray_ab32x3_autofix_${lane}_${START_TS}"

  echo "[$(date '+%F %T')] RAY_START lane=${lane} gpu=${gpu} port=${ray_port}" | tee -a "${STATUS_LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" ray start --head \
    --port "${ray_port}" \
    --num-cpus "${RAY_NUM_CPUS}" \
    --num-gpus 1 \
    --min-worker-port "${min_worker}" \
    --max-worker-port "${max_worker}" \
    --temp-dir "${temp_dir}" \
    --include-dashboard=False \
    --dashboard-port "${dash}" \
    --dashboard-agent-listen-port "${dash_agent}" \
    --metrics-export-port "${metrics}" \
    --runtime-env-agent-port "${runtime}" \
    --disable-usage-stats \
    >> "${STATUS_LOG}" 2>&1
}

start_all_ray() {
  stop_all_ray
  for lane in "${LANE_NAMES[@]}"; do
    start_ray_head "${lane}"
  done
}

classify_failure() {
  local log_path="$1"
  if [ ! -f "${log_path}" ]; then
    echo "missing_log"
    return 0
  fi
  if grep -q "Free memory on device" "${log_path}"; then
    echo "gpu_mem"
  elif grep -q "WandbAttachFailedError\\|Unable to attach to run" "${log_path}"; then
    echo "wandb_attach"
  elif grep -q "Engine core initialization failed" "${log_path}"; then
    echo "engine_init"
  else
    echo "unknown"
  fi
}

find_free_gpu_for_lane() {
  local lane="$1"
  local candidate
  local used_map
  local line idx mem

  used_map=""
  for other in "${LANE_NAMES[@]}"; do
    if [ "${other}" != "${lane}" ]; then
      used_map="${used_map},${LANE_GPU[${other}]},"
    fi
  done

  while IFS=',' read -r idx mem; do
    idx="$(echo "${idx}" | tr -d ' ')"
    mem="$(echo "${mem}" | tr -d ' ')"
    for candidate in "${GPU_POOL_ARR[@]}"; do
      if [ "${candidate}" != "${idx}" ]; then
        continue
      fi
      if [[ "${used_map}" == *",${candidate},"* ]]; then
        continue
      fi
      if [ "${mem}" -lt 1024 ]; then
        echo "${candidate}"
        return 0
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  return 1
}

lower_mem_util() {
  local cur="$1"
  awk -v cur="${cur}" -v step="${MEM_UTIL_STEP}" -v floor="${MEM_UTIL_FLOOR}" 'BEGIN{
    next=cur-step;
    if (next < floor) next=floor;
    printf "%.2f", next;
  }'
}

apply_repair() {
  local lane="$1"
  local reason="$2"
  local changed=0
  local actions=()

  if [ "${reason}" = "gpu_mem" ] || [ "${reason}" = "engine_init" ]; then
    local cur="${LANE_MEM_UTIL[${lane}]}"
    local next
    next="$(lower_mem_util "${cur}")"
    if [ "${next}" != "${cur}" ]; then
      LANE_MEM_UTIL["${lane}"]="${next}"
      actions+=("mem_util:${cur}->${next}")
      changed=1
    fi

    local old_gpu="${LANE_GPU[${lane}]}"
    local new_gpu
    if new_gpu="$(find_free_gpu_for_lane "${lane}" 2>/dev/null)"; then
      if [ "${new_gpu}" != "${old_gpu}" ]; then
        LANE_GPU["${lane}"]="${new_gpu}"
        actions+=("gpu:${old_gpu}->${new_gpu}")
        changed=1
      fi
    fi
  fi

  if [ "${reason}" = "wandb_attach" ] && [ "${WANDB_FALLBACK_OFFLINE}" = "1" ]; then
    if [ "${LANE_WANDB_MODE[${lane}]}" != "offline" ]; then
      LANE_WANDB_MODE["${lane}"]="offline"
      actions+=("wandb:online->offline")
      changed=1
    fi
  fi

  if [ "${changed}" -eq 1 ]; then
    echo "[$(date '+%F %T')] REPAIR lane=${lane} reason=${reason} action=$(IFS=';'; echo "${actions[*]}")" | tee -a "${STATUS_LOG}"
    return 0
  fi

  echo "[$(date '+%F %T')] REPAIR lane=${lane} reason=${reason} action=none" | tee -a "${STATUS_LOG}"
  return 1
}

run_one() {
  local lane="$1"
  local case_id="$2"
  local ctx="$3"
  local inf="$4"
  local attempt="$5"
  local run_name
  local driver_log
  local gpu="${LANE_GPU[${lane}]}"
  local ray_addr="${LANE_RAY_ADDR[${lane}]}"
  local mem_util="${LANE_MEM_UTIL[${lane}]}"
  local wandb_mode="${LANE_WANDB_MODE[${lane}]}"

  run_name="$(build_run_name "${lane}" "${case_id}" "${ctx}" "${inf}" "${attempt}")"
  driver_log="$(driver_log_path "${run_name}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_name}" "${lane}" "${case_id}" "${attempt}" "${gpu}" "${ray_addr}" "${ctx}" "${inf}" "${IB}" "${TB}" "${mem_util}" "${wandb_mode}" "${driver_log}" "started" >> "${MANIFEST}"

  echo "[$(date '+%F %T')] RUN ${run_name} lane=${lane} gpu=${gpu} ray=${ray_addr} mem_util=${mem_util} wandb=${wandb_mode}" | tee -a "${STATUS_LOG}"

  if env -u RAY_ADDRESS \
      WANDB_MODE="${wandb_mode}" \
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
        --gpu_memory_utilization "${mem_util}" \
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
  stop_all_ray
}
trap cleanup EXIT

JOBS=()
for i in "${!CASES_A[@]}"; do
  tag="$(printf "%02d" "$((i + 1))")"
  JOBS+=("A${tag},${CASES_A[$i]}")
  JOBS+=("B${tag},${CASES_B[$i]}")
done

TOTAL_CASES="${#JOBS[@]}"
TOTAL_ROUNDS="$(((TOTAL_CASES + 2) / 3))"
echo "[$(date '+%F %T')] PLAN total_cases=${TOTAL_CASES} total_rounds=${TOTAL_ROUNDS} dispatch=round_robin_3gpu_autofix retries=${MAX_RETRIES}" | tee -a "${STATUS_LOG}"
echo "[$(date '+%F %T')] GPU_INIT A=${LANE_GPU[A]} B=${LANE_GPU[B]} C=${LANE_GPU[C]} pool=${GPU_POOL}" | tee -a "${STATUS_LOG}"

cursor=0
round=0
total_failures=0

while (( cursor < TOTAL_CASES )); do
  round=$((round + 1))

  round_specs=()
  for lane_idx in 0 1 2; do
    if (( cursor >= TOTAL_CASES )); then
      break
    fi
    lane="${LANE_NAMES[$lane_idx]}"
    IFS=',' read -r case_id ctx inf <<< "${JOBS[$cursor]}"
    round_specs+=("${lane},${case_id},${ctx},${inf}")
    cursor=$((cursor + 1))
  done

  echo "[$(date '+%F %T')] ROUND_START ${round}/${TOTAL_ROUNDS} jobs=${#round_specs[@]}" | tee -a "${STATUS_LOG}"

  unresolved=("${round_specs[@]}")
  attempt=0

  while (( ${#unresolved[@]} > 0 )) && (( attempt <= MAX_RETRIES )); do
    echo "[$(date '+%F %T')] ATTEMPT_START round=${round} attempt=${attempt} unresolved=${#unresolved[@]}" | tee -a "${STATUS_LOG}"
    start_all_ray

    pids=()
    lanes=()
    case_ids=()
    ctxs=()
    infs=()

    for spec in "${unresolved[@]}"; do
      IFS=',' read -r lane case_id ctx inf <<< "${spec}"
      run_one "${lane}" "${case_id}" "${ctx}" "${inf}" "${attempt}" &
      pids+=("$!")
      lanes+=("${lane}")
      case_ids+=("${case_id}")
      ctxs+=("${ctx}")
      infs+=("${inf}")
    done

    next_unresolved=()
    for i in "${!pids[@]}"; do
      lane="${lanes[$i]}"
      case_id="${case_ids[$i]}"
      ctx="${ctxs[$i]}"
      inf="${infs[$i]}"
      if ! wait "${pids[$i]}"; then
        run_name="$(build_run_name "${lane}" "${case_id}" "${ctx}" "${inf}" "${attempt}")"
        log_path="$(driver_log_path "${run_name}")"
        reason="$(classify_failure "${log_path}")"
        echo "[$(date '+%F %T')] FAIL_REASON lane=${lane} case=${case_id} attempt=${attempt} reason=${reason}" | tee -a "${STATUS_LOG}"
        apply_repair "${lane}" "${reason}" || true
        next_unresolved+=("${lane},${case_id},${ctx},${inf}")
      fi
    done

    stop_all_ray
    unresolved=("${next_unresolved[@]}")
    attempt=$((attempt + 1))
    echo "[$(date '+%F %T')] ATTEMPT_DONE round=${round} next_unresolved=${#unresolved[@]}" | tee -a "${STATUS_LOG}"
  done

  if (( ${#unresolved[@]} > 0 )); then
    total_failures=$((total_failures + ${#unresolved[@]}))
    echo "[$(date '+%F %T')] ROUND_FAIL ${round}/${TOTAL_ROUNDS} unresolved=${#unresolved[@]}" | tee -a "${STATUS_LOG}"
  else
    echo "[$(date '+%F %T')] ROUND_DONE ${round}/${TOTAL_ROUNDS} all_success" | tee -a "${STATUS_LOG}"
  fi
done

if (( total_failures == 0 )); then
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} all_success" | tee -a "${STATUS_LOG}"
else
  echo "[$(date '+%F %T')] FINISH ${RUN_ID} unresolved_failures=${total_failures}" | tee -a "${STATUS_LOG}"
fi
