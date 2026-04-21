#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vllm/tream
RETRY_DIR=/workspace/vllm/tream/f_test/Ego4d/ab48/01_no_feature/retry
mkdir -p "$RETRY_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$RETRY_DIR/ego4d_ab48_nospec_full_remaining_2gpu_${TS}.master.log"

echo "[$(date "+%F %T")] START wait-current-then-fill-48" | tee -a "$MASTER_LOG"

while ps -eo cmd | grep -F "doh_shift_ab48x3_static_nodyn_nospec_ib32_tb16_C03_20260421_015836" | grep -v grep >/dev/null 2>&1; do
  echo "[$(date "+%F %T")] WAIT current C03 still running..." | tee -a "$MASTER_LOG"
  sleep 30
done

echo "[$(date "+%F %T")] current C03 finished, start filling missing cases" | tee -a "$MASTER_LOG"

get_missing() {
python - <<'PY2'
import pathlib, json, re
base=pathlib.Path('/workspace/vllm/tream/f_test/Ego4d/ab48/01_no_feature')
cases=[f'{g}{i:02d}' for g in 'ABCDEF' for i in range(1,9)]
pat=re.compile(r'_(A\d\d|B\d\d|C\d\d|D\d\d|E\d\d|F\d\d)_')
best={c:-1 for c in cases}
for p in base.rglob('*.jsonl'):
    m=pat.search(p.name)
    if not m: continue
    c=m.group(1)
    if c not in best: continue
    last=-1
    try:
        with p.open() as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: d=json.loads(line)
                except: continue
                fi=d.get('frame_index')
                if isinstance(fi,int): last=fi
    except: continue
    if last>best[c]: best[c]=last
for c in cases:
    if best[c] < 3999:
        print(c)
PY2
}

mapfile -t MISSING < <(get_missing)
echo "[$(date "+%F %T")] missing_count=${#MISSING[@]}" | tee -a "$MASTER_LOG"

if (( ${#MISSING[@]} == 0 )); then
  echo "[$(date "+%F %T")] ALL_DONE already 48/48" | tee -a "$MASTER_LOG"
  exit 0
fi

idx=0
batch=0
while (( idx < ${#MISSING[@]} )); do
  c1="${MISSING[$idx]}"
  c2=""
  if (( idx + 1 < ${#MISSING[@]} )); then
    c2="${MISSING[$((idx+1))]}"
  fi
  if [[ -n "$c2" ]]; then
    CASES="$c1,$c2"
  else
    CASES="$c1"
  fi
  batch=$((batch+1))
  echo "[$(date "+%F %T")] BATCH_${batch}_START cases=${CASES}" | tee -a "$MASTER_LOG"
  RUN_MODE=static \
  RUN_SPEC=0 \
  DATASET_SUBDIR=Ego4d \
  LOG_DIR="$RETRY_DIR" \
  GPU_A=0 GPU_B=1 GPU_C=1 \
  CASE_FILTER="$CASES" \
  bash /workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh >> "$MASTER_LOG" 2>&1 || true
  echo "[$(date "+%F %T")] BATCH_${batch}_DONE cases=${CASES}" | tee -a "$MASTER_LOG"
  idx=$((idx+2))
done

mapfile -t LEFT < <(get_missing)
echo "[$(date "+%F %T")] FINAL missing_count=${#LEFT[@]}" | tee -a "$MASTER_LOG"
if (( ${#LEFT[@]} > 0 )); then
  echo "[$(date "+%F %T")] FINAL missing_cases=$(IFS=,; echo "${LEFT[*]}")" | tee -a "$MASTER_LOG"
else
  echo "[$(date "+%F %T")] FINAL ALL_DONE 48/48" | tee -a "$MASTER_LOG"
fi
