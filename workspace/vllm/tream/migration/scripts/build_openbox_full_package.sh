#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/m-coriander/coriander/daifeng/testvllm/vllm}"
MIGRATION_ROOT="${MIGRATION_ROOT:-${SRC_ROOT}/tream/migration}"
OUT_ROOT="${OUT_ROOT:-/m-coriander/coriander/daifeng/release_packages}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PKG_NAME="openbox_full_${STAMP}"
OUT_DIR="${OUT_ROOT}/${PKG_NAME}"
TAR_PATH="${OUT_ROOT}/${PKG_NAME}.tar.gz"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "SRC_ROOT not found: ${SRC_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "[build] source: ${SRC_ROOT}"
echo "[build] out:    ${OUT_DIR}"
echo "[build] tar:    ${TAR_PATH}"

rm -rf "${OUT_DIR}" "${TAR_PATH}"
mkdir -p "${OUT_DIR}/workspace" "${OUT_DIR}/tools" "${OUT_DIR}/docs"

echo "[1/6] rsync repo snapshot (code + required assets, excluding logs/tmp)..."
rsync -a \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '**/__pycache__/' \
  --exclude '**/.pytest_cache/' \
  --exclude 'tream/tmp/' \
  --exclude 'tream/wandb/' \
  --exclude 'tream/inference_logs/' \
  --exclude 'tream/inference_logs_static_baseline/' \
  --exclude 'tream/inference_logs_stone/' \
  --exclude 'tream/inference_logs——121diaodu/' \
  --exclude 'tream/actor_log/' \
  --exclude 'tream/ablation/log/' \
  --exclude 'tream/experiment/logs/' \
  --exclude 'tream/experiment/wandb/' \
  --exclude 'tream/experiment/plots/' \
  --exclude 'tream/experiment/tmp/' \
  --exclude 'tream/experiment/inference_logs/' \
  --exclude 'tream/experiment/training_logs/' \
  --exclude 'tream/training_logs/' \
  --exclude 'tream/saved_models/' \
  --exclude 'tream/migration/bundle/' \
  --exclude 'tream/migration/openbox_full_*/' \
  --exclude 'SpecForge/output/' \
  "${SRC_ROOT}/" "${OUT_DIR}/workspace/vllm/"

echo "[2/6] copy required draft checkpoint only..."
mkdir -p "${OUT_DIR}/workspace/vllm/SpecForge/output/lvm_eagle3_lora_v1"
rsync -a \
  "${SRC_ROOT}/SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475" \
  "${OUT_DIR}/workspace/vllm/SpecForge/output/lvm_eagle3_lora_v1/"

echo "[3/6] copy migration helpers..."
cp -a "${MIGRATION_ROOT}/no_docker" "${OUT_DIR}/tools/"
cp -a "${MIGRATION_ROOT}/env_snapshots" "${OUT_DIR}/docs/"
cp "${MIGRATION_ROOT}/scripts/launch_ab48_8gpu.sh" "${OUT_DIR}/tools/launch_ab48_8gpu.sh"
chmod +x \
  "${OUT_DIR}/tools/no_docker/setup.sh" \
  "${OUT_DIR}/tools/launch_ab48_8gpu.sh"

cat > "${OUT_DIR}/tools/run_ab48_spec_static_8gpu.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${ROOT}/workspace/vllm"
TREAM_ROOT="${REPO}/tream"

ENV_METHOD="${ENV_METHOD:-conda}"     # conda | venv
ENV_NAME="${ENV_NAME:-tream}"
DATASET_SUBDIR="${DATASET_SUBDIR:-Ego4d}"

bash "${ROOT}/tools/no_docker/setup.sh" \
  --method "${ENV_METHOD}" \
  --env-name "${ENV_NAME}" \
  --repo-root "${REPO}"

if [[ "${ENV_METHOD}" == "conda" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
else
  source "${HOME}/.venvs/${ENV_NAME}/bin/activate"
fi

TREAM_ROOT="${TREAM_ROOT}" \
RUN_MODE=static \
RUN_SPEC=1 \
DATASET_SUBDIR="${DATASET_SUBDIR}" \
GPU_LIST=0,1,2,3,4,5,6,7 \
CONCURRENCY=8 \
bash "${ROOT}/tools/launch_ab48_8gpu.sh"
SH
chmod +x "${OUT_DIR}/tools/run_ab48_spec_static_8gpu.sh"

cat > "${OUT_DIR}/tools/run_ab48_ablation_4modes_8gpu.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${ROOT}/workspace/vllm"
TREAM_ROOT="${REPO}/tream"
ENV_METHOD="${ENV_METHOD:-conda}"     # conda | venv
ENV_NAME="${ENV_NAME:-tream}"
DATASET_SUBDIR="${DATASET_SUBDIR:-Ego4d}"

bash "${ROOT}/tools/no_docker/setup.sh" \
  --method "${ENV_METHOD}" \
  --env-name "${ENV_NAME}" \
  --repo-root "${REPO}"

if [[ "${ENV_METHOD}" == "conda" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
else
  source "${HOME}/.venvs/${ENV_NAME}/bin/activate"
fi

run_mode() {
  local mode="$1"
  echo "[ablation] START ${mode}"
  case "${mode}" in
    no_feature)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=static RUN_SPEC=0 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    spec_only)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=static RUN_SPEC=1 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    dyn_nojsd)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=dynamic RUN_SPEC=1 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      SHIFT_SHOCK_DISABLE_DRIFT=1 \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    dyn_jsd)
      TREAM_ROOT="${TREAM_ROOT}" RUN_MODE=dynamic RUN_SPEC=1 DATASET_SUBDIR="${DATASET_SUBDIR}" \
      SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0 \
      GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
      bash "${ROOT}/tools/launch_ab48_8gpu.sh"
      ;;
    *)
      echo "unknown mode ${mode}" >&2; exit 2 ;;
  esac
  echo "[ablation] DONE ${mode}"
}

run_mode no_feature
run_mode spec_only
run_mode dyn_nojsd
run_mode dyn_jsd
SH
chmod +x "${OUT_DIR}/tools/run_ab48_ablation_4modes_8gpu.sh"

cat > "${OUT_DIR}/README.md" <<'MD'
# Open-box Full Package (No Docker)

This package is intended to be copied to a remote machine and run directly.

## Included
- `workspace/vllm/`: runnable code snapshot (hacked vLLM + tream + SpecForge code)
- `workspace/vllm/tream/lvm-llama2-7b/`: base model
- `workspace/vllm/tream/data/streaming-lvm-dataset/`: datasets
- `workspace/vllm/SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475/`: draft checkpoint
- `tools/no_docker/`: environment setup files
- `tools/launch_ab48_8gpu.sh`: 8-GPU launcher
- `tools/run_ab48_spec_static_8gpu.sh`: one-click single experiment
- `tools/run_ab48_ablation_4modes_8gpu.sh`: one-click 4-mode ablation

## Quick start

```bash
tar -xzf openbox_full_*.tar.gz
cd openbox_full_*

# one-click: AB48 static spec, 8-GPU
bash tools/run_ab48_spec_static_8gpu.sh
```

## One-click full ablation (4 modes)

```bash
bash tools/run_ab48_ablation_4modes_8gpu.sh
```

Defaults:
- environment: `conda`, env name `tream`
- dataset: `Ego4d`
- GPUs: `0,1,2,3,4,5,6,7`
- concurrency: `8`
MD

echo "[4/6] add package manifest..."
{
  echo "name=${PKG_NAME}"
  echo "build_time=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "source=${SRC_ROOT}"
  echo "git_head=$(cd "${SRC_ROOT}" && git rev-parse HEAD 2>/dev/null || echo n/a)"
} > "${OUT_DIR}/docs/package_manifest.txt"

echo "[5/6] size summary..."
du -sh "${OUT_DIR}"/* | sort -h | tee "${OUT_DIR}/docs/size_summary.txt"

echo "[6/6] create tar.gz (this may take long)..."
tar -C "${OUT_ROOT}" -czf "${TAR_PATH}" "${PKG_NAME}"

echo "[done]"
echo "dir: ${OUT_DIR}"
echo "tar: ${TAR_PATH}"
ls -lh "${TAR_PATH}"
