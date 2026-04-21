#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_MODE="dynamic"
export GPU_A="${GPU_A:-4}"
export GPU_B="${GPU_B:-5}"
export GPU_C="${GPU_C:-6}"

# Explicitly mark whitelist-break dynamic config in run names
export SHIFT_COLD_WHITELIST_PROBE="${SHIFT_COLD_WHITELIST_PROBE:-1}"
export SHIFT_COLD_WHITELIST_BUDGET="${SHIFT_COLD_WHITELIST_BUDGET:-12}"
export SHIFT_COLD_WHITELIST_MAX_SWITCH="${SHIFT_COLD_WHITELIST_MAX_SWITCH:-2}"
export SHIFT_COLD_WHITELIST_LAT_SLACK="${SHIFT_COLD_WHITELIST_LAT_SLACK:-0.15}"
export SHIFT_COLD_WHITELIST_Q_SLACK="${SHIFT_COLD_WHITELIST_Q_SLACK:-0.60}"
export SHIFT_SHOCK_USE_JSD_ONLY="${SHIFT_SHOCK_USE_JSD_ONLY:-1}"

exec "${SCRIPT_DIR}/launch_ab48_trigpu_shift_taskpool.sh"
