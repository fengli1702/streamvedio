#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_MODE="static"
export GPU_A="${GPU_A:-1}"
export GPU_B="${GPU_B:-2}"
export GPU_C="${GPU_C:-3}"

exec "${SCRIPT_DIR}/launch_ab48_trigpu_shift_taskpool.sh"
