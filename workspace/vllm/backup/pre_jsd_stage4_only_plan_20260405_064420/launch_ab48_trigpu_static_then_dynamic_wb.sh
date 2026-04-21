#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/inference_logs"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/ab48x3_static_then_dynamic_${TS}.launcher.log"

{
  echo "[$(date '+%F %T')] START static_then_dynamic"
  echo "[$(date '+%F %T')] STEP static on GPUs ${GPU_A:-1},${GPU_B:-2},${GPU_C:-3}"
} | tee -a "${MASTER_LOG}"

GPU_A="${GPU_A:-1}" GPU_B="${GPU_B:-2}" GPU_C="${GPU_C:-3}" \
  "${SCRIPT_DIR}/launch_ab48_trigpu_shift_taskpool_static.sh" | tee -a "${MASTER_LOG}"

{
  echo "[$(date '+%F %T')] STEP dynamic_wb on GPUs ${GPU_DYN_A:-4},${GPU_DYN_B:-5},${GPU_DYN_C:-6}"
} | tee -a "${MASTER_LOG}"

GPU_A="${GPU_DYN_A:-4}" GPU_B="${GPU_DYN_B:-5}" GPU_C="${GPU_DYN_C:-6}" \
  "${SCRIPT_DIR}/launch_ab48_trigpu_shift_taskpool_dynamic_wb.sh" | tee -a "${MASTER_LOG}"

{
  echo "[$(date '+%F %T')] FINISH static_then_dynamic"
} | tee -a "${MASTER_LOG}"
