#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm/tream
mkdir -p /workspace/vllm/tream/f_test/Ego4d/ab48/00_metadata
mkdir -p /workspace/vllm/tream/f_test/Ego4d/ab48/02_spec_only/retry
mkdir -p /workspace/vllm/tream/f_test/Ego4d/ab48/01_no_feature/retry
TS=$(date +%Y%m%d_%H%M%S)
LOG=/workspace/vllm/tream/f_test/Ego4d/ab48/00_metadata/ab12_fail_retry_${TS}.launcher.log
{
  echo "[$(date '+%F %T')] START fail-retry batch (spec then nospec)"
  echo "[$(date '+%F %T')] SPEC retry cases=B01,B08,C03"
  RUN_MODE=static RUN_SPEC=1 DATASET_SUBDIR=Ego4d \
  GPU_A=3 GPU_B=4 GPU_C=7 \
  RAY_PORT_A=35621 RAY_PORT_B=35622 RAY_PORT_C=35623 \
  RAY_WORKER_MIN_A=57000 RAY_WORKER_MAX_A=57079 \
  RAY_WORKER_MIN_B=57100 RAY_WORKER_MAX_B=57179 \
  RAY_WORKER_MIN_C=57200 RAY_WORKER_MAX_C=57279 \
  CASE_FILTER=B01,B08,C03 \
  LOG_DIR=/workspace/vllm/tream/f_test/Ego4d/ab48/02_spec_only/retry \
  ./inference_logs/launch_ab48_trigpu_shift_taskpool.sh

  echo "[$(date '+%F %T')] NOSPEC retry cases=A01,A08,B03,B08,B05,C01,C02,C03,C04"
  RUN_MODE=static RUN_SPEC=0 DATASET_SUBDIR=Ego4d \
  GPU_A=3 GPU_B=4 GPU_C=7 \
  RAY_PORT_A=35721 RAY_PORT_B=35722 RAY_PORT_C=35723 \
  RAY_WORKER_MIN_A=57300 RAY_WORKER_MAX_A=57379 \
  RAY_WORKER_MIN_B=57400 RAY_WORKER_MAX_B=57479 \
  RAY_WORKER_MIN_C=57500 RAY_WORKER_MAX_C=57579 \
  CASE_FILTER=A01,A08,B03,B08,B05,C01,C02,C03,C04 \
  LOG_DIR=/workspace/vllm/tream/f_test/Ego4d/ab48/01_no_feature/retry \
  ./inference_logs/launch_ab48_trigpu_shift_taskpool.sh

  echo "[$(date '+%F %T')] FINISH fail-retry batch"
} > "$LOG" 2>&1
