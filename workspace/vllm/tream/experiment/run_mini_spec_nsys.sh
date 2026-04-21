#!/usr/bin/env bash
set -euo pipefail

# Mini Spec Benchmark runner with Nsight Systems profiling.
# Runs four cases on the streaming path:
#  1) LoRA OFF,  Spec OFF   (baseline)
#  2) LoRA ON,   Spec OFF   (baseline)
#  3) LoRA ON,   Spec ON    (EAGLE-3)
#  4) LoRA OFF,  Spec ON    (EAGLE-3)

# Defaults (override via flags or env)
GPU_ID=${GPU_ID:-4}
DATASET_ROOT=${DATASET_ROOT:-/app/data/streaming-lvm-dataset/DOH}
TEACHER_MODEL=${TEACHER_MODEL:-/app/saved_models/lvm-llama2-7b}
LORA_DIR=${LORA_DIR:-/app/tmp/bench_lora_adapter}
DRAFT_MODEL=${DRAFT_MODEL:-/app/SpecForge/checkpoints/lvm_eagle3_v1_8292/epoch_0}
VOCAB_MAP=${VOCAB_MAP:-/app/SpecForge/cache/lvm_eagle3_v1_8292/vocab_mapping/027d603b856cb0a6c76f074c0414b23b.pt}

INFER_BATCH=${INFER_BATCH:-32}
CTX_LEN=${CTX_LEN:-4}
INF_LEN=${INF_LEN:-4}
NUM_PROMPTS=${NUM_PROMPTS:-256}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
OUT_DIR=${OUT_DIR:-./inference_logs/mini}

SCRIPT=./experiments/mini_spec_benchmark.py

usage() {
  cat <<EOF
Usage: $0 [options]
  -g GPU_ID              CUDA device id (default: ${GPU_ID})
  -d DATASET_ROOT        Dataset root with frames (default: ${DATASET_ROOT})
  -m TEACHER_MODEL       HF model path (default: ${TEACHER_MODEL})
  -a LORA_DIR            LoRA adapter dir (default: ${LORA_DIR})
  -r DRAFT_MODEL         EAGLE-3 draft model path (default: ${DRAFT_MODEL})
  -v VOCAB_MAP           Vocab mapping .pt path (default: ${VOCAB_MAP})
  -b INFER_BATCH         Inference batch size (default: ${INFER_BATCH})
  -c CTX_LEN             Context length (default: ${CTX_LEN})
  -l INF_LEN             Inference length per step (default: ${INF_LEN})
  -n NUM_PROMPTS         Total prompts (default: ${NUM_PROMPTS})
  -o OUT_DIR             Output dir for logs (default: ${OUT_DIR})
EOF
}

while getopts ":g:d:m:a:r:v:b:c:l:n:o:h" opt; do
  case $opt in
    g) GPU_ID=$OPTARG;;
    d) DATASET_ROOT=$OPTARG;;
    m) TEACHER_MODEL=$OPTARG;;
    a) LORA_DIR=$OPTARG;;
    r) DRAFT_MODEL=$OPTARG;;
    v) VOCAB_MAP=$OPTARG;;
    b) INFER_BATCH=$OPTARG;;
    c) CTX_LEN=$OPTARG;;
    l) INF_LEN=$OPTARG;;
    n) NUM_PROMPTS=$OPTARG;;
    o) OUT_DIR=$OPTARG;;
    h) usage; exit 0;;
    *) usage; exit 1;;
  esac
done

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERR] Cannot find $SCRIPT. Run from repo root or set correct path." >&2
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERR] DATASET_ROOT not found: $DATASET_ROOT" >&2
  exit 1
fi

if [[ ! -d "$TEACHER_MODEL" ]]; then
  echo "[ERR] TEACHER_MODEL not found: $TEACHER_MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[INFO] Running 1/4: LoRA ON, Spec ON (EAGLE-3)"
if [[ ! -d "$DRAFT_MODEL" ]]; then
  echo "[ERR] DRAFT_MODEL not found: $DRAFT_MODEL" >&2; exit 1
fi
if [[ ! -f "$VOCAB_MAP" ]]; then
  echo "[ERR] VOCAB_MAP not found: $VOCAB_MAP" >&2; exit 1
fi
CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_USE_V1=1 VLLM_MAX_SPEC_TOKENS=3 \
nsys profile -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --wait=all \
  --trace-fork-before-exec=true \
  --sample=none \
  --force-overwrite=true \
  -o nsys_mini_stream_lora_on_spec \
  python3 "$SCRIPT" \
    --dataset-root "$DATASET_ROOT" \
    --teacher-model "$TEACHER_MODEL" \
    --streaming \
    --inference-batch-size "$INFER_BATCH" \
    --context-length "$CTX_LEN" \
    --inference-length "$INF_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --output-dir "$OUT_DIR" \
    --max-loras 1 --max-lora-rank 16 \
    --lora-adapter-dir "$LORA_DIR" \
    --spec-method eagle3 \
    --draft-model "$DRAFT_MODEL" \
    --vocab-mapping-path "$VOCAB_MAP" \
    --num-spec-tokens 3 \
    --spec-disable-mqa-scorer \
    --run-mode spec

echo "[INFO] Running 2/4: LoRA OFF, Spec ON (EAGLE-3)"
CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_USE_V1=1 VLLM_MAX_SPEC_TOKENS=3 \
nsys profile -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --wait=all \
  --trace-fork-before-exec=true \
  --sample=none \
  --force-overwrite=true \
  -o nsys_mini_stream_lora_off_spec \
  python3 "$SCRIPT" \
    --dataset-root "$DATASET_ROOT" \
    --teacher-model "$TEACHER_MODEL" \
    --streaming \
    --inference-batch-size "$INFER_BATCH" \
    --context-length "$CTX_LEN" \
    --inference-length "$INF_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --output-dir "$OUT_DIR" \
    --disable-lora \
    --spec-method eagle3 \
    --draft-model "$DRAFT_MODEL" \
    --vocab-mapping-path "$VOCAB_MAP" \
    --num-spec-tokens 3 \
    --spec-disable-mqa-scorer \
    --run-mode spec

echo "[INFO] Running 3/4: LoRA OFF, Spec OFF (baseline)"
CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_USE_V1=1 \
nsys profile -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --wait=all \
  --trace-fork-before-exec=true \
  --sample=none \
  --force-overwrite=true \
  -o nsys_mini_stream_lora_off_nospec \
  python3 "$SCRIPT" \
    --dataset-root "$DATASET_ROOT" \
    --teacher-model "$TEACHER_MODEL" \
    --streaming \
    --inference-batch-size "$INFER_BATCH" \
    --context-length "$CTX_LEN" \
    --inference-length "$INF_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --output-dir "$OUT_DIR" \
    --disable-lora \
    --run-mode baseline

echo "[INFO] Running 4/4: LoRA ON, Spec OFF (baseline)"
CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_USE_V1=1 \
nsys profile -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --wait=all \
  --trace-fork-before-exec=true \
  --sample=none \
  --force-overwrite=true \
  -o nsys_mini_stream_lora_on_nospec \
  python3 "$SCRIPT" \
    --dataset-root "$DATASET_ROOT" \
    --teacher-model "$TEACHER_MODEL" \
    --streaming \
    --inference-batch-size "$INFER_BATCH" \
    --context-length "$CTX_LEN" \
    --inference-length "$INF_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --output-dir "$OUT_DIR" \
    --max-loras 1 --max-lora-rank 16 \
    --lora-adapter-dir "$LORA_DIR" \
    --run-mode baseline

echo "[DONE] Nsight runs completed. Reports: *.nsys-rep in current directory."
