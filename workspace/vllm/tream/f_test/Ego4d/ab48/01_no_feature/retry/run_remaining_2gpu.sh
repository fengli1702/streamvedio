#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm/tream
RETRY_DIR=/workspace/vllm/tream/f_test/Ego4d/ab48/01_no_feature/retry
mkdir -p "$RETRY_DIR"
rm -rf /workspace/vllm/tream/tmp/* 2>/dev/null || true
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$RETRY_DIR/ego4d_ab12_nospec_remaining_2gpu_${TS}.master.log"
echo "[$(date "+%F %T")] START 2gpu-2concurrency batches=B01,B03|B08,C01|C03" | tee -a "$MASTER_LOG"
run_batch() {
  local CASES="$1"
  echo "[$(date "+%F %T")] BATCH_START cases=${CASES}" | tee -a "$MASTER_LOG"
  RUN_MODE=static \
  RUN_SPEC=0 \
  DATASET_SUBDIR=Ego4d \
  LOG_DIR="$RETRY_DIR" \
  GPU_A=0 GPU_B=1 GPU_C=1 \
  CASE_FILTER="$CASES" \
  bash /workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh >> "$MASTER_LOG" 2>&1
  echo "[$(date "+%F %T")] BATCH_DONE cases=${CASES}" | tee -a "$MASTER_LOG"
}
run_batch "B01,B03"
run_batch "B08,C01"
run_batch "C03"
echo "[$(date "+%F %T")] ALL_DONE" | tee -a "$MASTER_LOG"
