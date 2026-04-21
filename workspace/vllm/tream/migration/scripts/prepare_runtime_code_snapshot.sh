#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/m-coriander/coriander/daifeng/testvllm/vllm}"
SNAPSHOT_DIR="${2:-/tmp/tream_code_snapshot}"

mkdir -p "${SNAPSHOT_DIR}"
rm -rf "${SNAPSHOT_DIR:?}"/*

# Keep runnable code/scripts, drop large artifacts/logs/checkpoints.
rsync -a \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '**/__pycache__/' \
  --exclude '**/.pytest_cache/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude 'tream/data/' \
  --exclude 'tream/wandb/' \
  --exclude 'tream/tmp/' \
  --exclude 'tream/lvm-llama2-7b/' \
  --exclude 'SpecForge/output/' \
  --exclude 'tream/saved_models/' \
  --exclude 'tream/training_logs/' \
  --exclude 'tream/experiment/logs/' \
  --exclude 'tream/experiment/wandb/' \
  --exclude 'tream/experiment/tmp/' \
  --exclude 'tream/experiment/plots/' \
  --exclude 'tream/ablation/log/' \
  --exclude 'tream/migration/bundle/' \
  --exclude 'tream/inference_logs/backups/' \
  --exclude 'tream/inference_logs/rerun_backup_*/' \
  --exclude 'tream/inference_logs/*.jsonl' \
  --exclude 'tream/inference_logs/*.driver.log' \
  --exclude 'tream/inference_logs/*.status.log' \
  --exclude 'tream/inference_logs/*.tsv' \
  --exclude 'tream/inference_logs/*.txt' \
  --exclude 'tream/inference_logs/*.png' \
  --exclude 'tream/inference_logs/*.tar.gz' \
  --exclude 'tream/inference_logs/*.zip' \
  "${REPO_ROOT}/" "${SNAPSHOT_DIR}/vllm/"

echo "Code snapshot prepared: ${SNAPSHOT_DIR}/vllm"
