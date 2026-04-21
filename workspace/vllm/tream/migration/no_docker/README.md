# No-Docker Deployment (8-GPU, exclusive node)

适用于云端没有 Docker 的场景：直接在主机上创建 Python 环境并运行。

## 1. 机器前提

- Linux x86_64
- NVIDIA Driver 建议 `>= 570`（当前实验环境是 `570.211.01`）
- 8 张 GPU（独占）
- Python 3.11（脚本会自动创建 conda/venv）

## 2. 目录说明

- `requirements.lock`：从当前可运行容器导出的 Python 依赖锁定（剔除了容器专有包）
- `environment.yml`：conda 环境模板
- `setup.sh`：开箱安装脚本（推荐）

## 2.1 重要：必须使用我们 hack 的 vLLM

本项目依赖本仓库里的 vLLM 修改版，不是官方原版 wheel。

- 正确方式：在仓库根目录执行 `pip install -e .`（editable）
- 错误方式：只装 `pip install vllm` 官方包

`setup.sh` 默认会强制安装本地 editable vLLM；如果失败会直接报错退出（避免误跑）。

## 3. 快速安装

在仓库根目录（`testvllm/vllm`）执行：

```bash
cd /path/to/testvllm/vllm/tream/migration/no_docker

# conda方式（推荐）
bash setup.sh --method conda --env-name tream

# 或 venv 方式
# bash setup.sh --method venv --env-name tream-venv
```

如果你不希望脚本碰 apt：

```bash
bash setup.sh --method conda --env-name tream --skip-apt
```

如果你只想调试、临时允许 fallback 到官方 vLLM（不推荐）：

```bash
bash setup.sh --method conda --env-name tream --allow-upstream-vllm-fallback
```

## 4. 激活环境

```bash
# conda
conda activate tream

# venv
# source ~/.venvs/tream-venv/bin/activate
```

## 5. 8卡并发运行 AB48

```bash
TREAM_ROOT=/path/to/testvllm/vllm/tream \
RUN_MODE=static \
RUN_SPEC=1 \
DATASET_SUBDIR=Ego4d \
GPU_LIST=0,1,2,3,4,5,6,7 \
CONCURRENCY=8 \
bash /path/to/testvllm/vllm/tream/migration/scripts/launch_ab48_8gpu.sh
```

说明：

- 这是“全任务一次发射 + 队列补位”，并发上限由 `CONCURRENCY=8` 决定。
- 首批会启动 8 条，后续有空闲卡自动补下一条。

## 6. 常用模式

- `RUN_MODE=static RUN_SPEC=0`：NoFeature
- `RUN_MODE=static RUN_SPEC=1`：SpecOnly
- `RUN_MODE=dynamic RUN_SPEC=1`：Dyn (可叠加 JSD 相关环境变量)

## 7. 自检

```bash
python -V
python - <<'PY'
import torch, ray, transformers, inspect, vllm
print("torch", torch.__version__)
print("ray", ray.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("vllm_path", inspect.getfile(vllm))
print("cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY
```

`vllm_path` 应该指向你的仓库路径（例如 `/path/to/testvllm/vllm/...`），而不是 site-packages 里的官方 wheel 路径。
