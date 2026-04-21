#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-tream}"
REPO_ROOT="${REPO_ROOT:-/m-coriander/coriander/daifeng/testvllm/vllm}"
MIGRATION_ROOT="${MIGRATION_ROOT:-${REPO_ROOT}/tream/migration}"
SAVE_IMAGE="${SAVE_IMAGE:-1}"   # 1=save docker image tar, 0=skip
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  exit 1
fi

IMAGE_NAME="$(docker inspect "${CONTAINER_NAME}" --format '{{.Config.Image}}')"
IMAGE_SAFE="$(echo "${IMAGE_NAME}" | tr '/:' '__')"

BUNDLE_DIR="${MIGRATION_ROOT}/bundle/runtime_bundle_${STAMP}"
SNAPSHOT_DIR="${BUNDLE_DIR}/env_snapshots"
CODE_DIR="${BUNDLE_DIR}/code_snapshot"
IMAGE_DIR="${BUNDLE_DIR}/docker_image"

mkdir -p "${SNAPSHOT_DIR}" "${CODE_DIR}" "${IMAGE_DIR}"

echo "[1/5] export container snapshots..."
docker inspect "${CONTAINER_NAME}" > "${SNAPSHOT_DIR}/docker_inspect_${CONTAINER_NAME}.json"
docker exec "${CONTAINER_NAME}" bash -lc 'python -m pip freeze' > "${SNAPSHOT_DIR}/pip_freeze.txt"
docker exec "${CONTAINER_NAME}" bash -lc 'python -m pip list --format=freeze' > "${SNAPSHOT_DIR}/pip_list.txt"
docker exec "${CONTAINER_NAME}" bash -lc 'printenv | sort' > "${SNAPSHOT_DIR}/container_env.txt"
docker exec "${CONTAINER_NAME}" bash -lc 'cat /etc/os-release; echo; uname -a' > "${SNAPSHOT_DIR}/os_uname.txt"
docker exec "${CONTAINER_NAME}" bash -lc 'nvidia-smi -L; echo; nvidia-smi' > "${SNAPSHOT_DIR}/nvidia_smi.txt"
docker exec "${CONTAINER_NAME}" bash -lc 'dpkg-query -W 2>/dev/null || true' > "${SNAPSHOT_DIR}/dpkg_query.txt"
docker exec -i "${CONTAINER_NAME}" python - <<'PY' > "${SNAPSHOT_DIR}/python_module_versions.txt"
import importlib
import platform
import sys

mods = ["torch", "ray", "transformers", "vllm", "numpy"]
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m}={getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        print(f"{m}=ERROR:{type(e).__name__}:{e}")
PY
{
  echo "container=${CONTAINER_NAME}"
  echo "image=${IMAGE_NAME}"
  echo "export_time=${STAMP}"
  echo "repo_root=${REPO_ROOT}"
  echo "git_head=$(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null || echo n/a)"
} > "${SNAPSHOT_DIR}/bundle_meta.txt"
cd "${REPO_ROOT}" && git status --short > "${SNAPSHOT_DIR}/repo_status_short.txt" || true

echo "[2/5] prepare code snapshot..."
"${MIGRATION_ROOT}/scripts/prepare_runtime_code_snapshot.sh" "${REPO_ROOT}" "${CODE_DIR}"

echo "[3/5] copy migration templates..."
cp "${MIGRATION_ROOT}/Dockerfile" "${BUNDLE_DIR}/Dockerfile"
cp "${MIGRATION_ROOT}/README.md" "${BUNDLE_DIR}/README.md"
mkdir -p "${BUNDLE_DIR}/scripts"
cp "${MIGRATION_ROOT}/scripts/"*.sh "${BUNDLE_DIR}/scripts/"

if [[ "${SAVE_IMAGE}" == "1" ]]; then
  echo "[4/5] save docker image (${IMAGE_NAME})..."
  docker save "${IMAGE_NAME}" -o "${IMAGE_DIR}/${IMAGE_SAFE}.tar"
else
  echo "[4/5] skip docker save (SAVE_IMAGE=${SAVE_IMAGE})"
fi

echo "[5/5] pack bundle..."
tar -C "$(dirname "${BUNDLE_DIR}")" -czf "${BUNDLE_DIR}.tar.gz" "$(basename "${BUNDLE_DIR}")"

echo
echo "Bundle ready:"
echo "  dir: ${BUNDLE_DIR}"
echo "  tar: ${BUNDLE_DIR}.tar.gz"
