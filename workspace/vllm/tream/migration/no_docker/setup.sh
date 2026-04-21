#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REQ_LOCK="${SCRIPT_DIR}/requirements.lock"

ENV_METHOD="conda"          # conda | venv
ENV_NAME="tream"
PYTHON_VERSION="3.11"
REPO_ROOT="${REPO_ROOT_DEFAULT}"
SKIP_APT="0"
SKIP_VLLM_EDITABLE="0"
ALLOW_UPSTREAM_VLLM_FALLBACK="0"

usage() {
  cat <<EOF
Usage:
  bash setup.sh [options]

Options:
  --method <conda|venv>         Env method (default: conda)
  --env-name <name>             Env name (default: tream)
  --python <version>            Python version (default: 3.11)
  --repo-root <path>            vllm repo root (default: ${REPO_ROOT_DEFAULT})
  --skip-apt                    Skip apt package install
  --skip-vllm-editable          Skip 'pip install -e <repo-root>'
  --allow-upstream-vllm-fallback
                                If local hacked vLLM editable install fails,
                                fallback to upstream wheel (NOT recommended)
  -h, --help                    Show this help

Examples:
  bash setup.sh --method conda --env-name tream
  bash setup.sh --method venv --env-name tream-venv --skip-apt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) ENV_METHOD="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --skip-apt) SKIP_APT="1"; shift 1 ;;
    --skip-vllm-editable) SKIP_VLLM_EDITABLE="1"; shift 1 ;;
    --allow-upstream-vllm-fallback) ALLOW_UPSTREAM_VLLM_FALLBACK="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "${REQ_LOCK}" ]]; then
  echo "requirements.lock not found: ${REQ_LOCK}" >&2
  exit 1
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "repo root not found: ${REPO_ROOT}" >&2
  exit 1
fi

echo "[setup] repo_root=${REPO_ROOT}"
echo "[setup] env_method=${ENV_METHOD} env_name=${ENV_NAME} python=${PYTHON_VERSION}"

if [[ "${SKIP_APT}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  PKGS=(build-essential git curl wget pkg-config cmake ninja-build libgl1 libglib2.0-0 ffmpeg)
  echo "[setup] installing apt packages: ${PKGS[*]}"
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${PKGS[@]}"
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${PKGS[@]}"
  else
    echo "[setup] WARN: no sudo/root, skip apt install."
  fi
fi

activate_env() {
  if [[ "${ENV_METHOD}" == "conda" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "conda not found. Use --method venv or install conda/mamba first." >&2
      exit 1
    fi
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
      conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    fi
    conda activate "${ENV_NAME}"
  elif [[ "${ENV_METHOD}" == "venv" ]]; then
    local venv_dir="${HOME}/.venvs/${ENV_NAME}"
    if [[ ! -d "${venv_dir}" ]]; then
      python3 -m venv "${venv_dir}"
    fi
    # shellcheck source=/dev/null
    source "${venv_dir}/bin/activate"
  else
    echo "Unsupported --method ${ENV_METHOD}" >&2
    exit 1
  fi
}

activate_env

python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing torch cu128"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

TMP_REQ="$(mktemp)"
trap 'rm -f "${TMP_REQ}"' EXIT
grep -vE '^torch==|^torchvision==|^torchaudio==|^vllm==' "${REQ_LOCK}" > "${TMP_REQ}"

echo "[setup] installing python deps from requirements.lock (without torch trio)"
python -m pip install -r "${TMP_REQ}"

if [[ "${SKIP_VLLM_EDITABLE}" != "1" ]]; then
  echo "[setup] installing HACKED local vLLM from repo (editable)"
  if ! (cd "${REPO_ROOT}" && VLLM_USE_PRECOMPILED=1 python -m pip install -e . --no-build-isolation); then
    if [[ "${ALLOW_UPSTREAM_VLLM_FALLBACK}" == "1" ]]; then
      echo "[setup] WARN: local editable install failed, fallback to upstream vLLM wheel."
      python -m pip install vllm==0.11.2 || true
    else
      echo "[setup] ERROR: local hacked vLLM install failed."
      echo "        Do NOT continue with upstream wheel for this project."
      echo "        Re-run with --allow-upstream-vllm-fallback only for debug."
      exit 2
    fi
  fi
fi

python - <<PY
import importlib
import inspect
mods=["torch","ray","transformers","numpy"]
for m in mods:
    mod=importlib.import_module(m)
    print(f"{m}={getattr(mod,'__version__','n/a')}")
try:
    import vllm
    print(f"vllm={getattr(vllm,'__version__','n/a')}")
    print("vllm_path", inspect.getfile(vllm))
except Exception as e:
    print("vllm=ERROR", type(e).__name__, e)
print("setup_ok")
PY

cat <<EOF

[setup] completed.

Activate env:
  conda activate ${ENV_NAME}    # if using conda
  source ~/.venvs/${ENV_NAME}/bin/activate   # if using venv

8-GPU AB48 launch (example):
  TREAM_ROOT=${REPO_ROOT}/tream \\
  RUN_MODE=static RUN_SPEC=1 DATASET_SUBDIR=Ego4d \\
  GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \\
  bash ${REPO_ROOT}/tream/migration/scripts/launch_ab48_8gpu.sh
EOF
