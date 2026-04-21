# TREAM Runtime Migration (Out-of-the-box)

这个目录用于把当前 `vllm-tream` 运行环境迁移到另一台机器，目标是尽量保持和当前实验机一致。

如果目标机**不能用 Docker**，请直接走：

- `no_docker/README.md`
- `no_docker/setup.sh`

## 1. 你会得到什么

运行导出脚本后会生成：

- `runtime_bundle_*.tar.gz`
- 包含：
  - `Dockerfile`（基于当前容器镜像）
  - `code_snapshot/vllm/`（可运行代码与脚本快照，已排除大体积日志/数据）
  - `env_snapshots/`（pip、系统、GPU、docker inspect、git 状态）
  - `scripts/`（启动容器/再导出脚本）
  - 可选 `docker_image/*.tar`（容器镜像导出）

## 2. 在源机器导出

在当前机器执行：

```bash
cd /m-coriander/coriander/daifeng/testvllm/vllm/tream/migration

# 默认会导出镜像 + 代码快照 + 环境快照
bash scripts/export_runtime_bundle.sh

# 如果只要代码和环境，不导出镜像：
# SAVE_IMAGE=0 bash scripts/export_runtime_bundle.sh
```

输出目录默认在：

- `tream/migration/bundle/runtime_bundle_<timestamp>/`
- 以及对应压缩包 `runtime_bundle_<timestamp>.tar.gz`

## 3. 在目标机器导入

### 3.1 解压

```bash
tar -xzf runtime_bundle_<timestamp>.tar.gz
cd runtime_bundle_<timestamp>
```

### 3.2 导入基础镜像（如果 bundle 里包含 docker_image）

```bash
docker load -i docker_image/*.tar
```

### 3.3 构建迁移镜像

```bash
# 若镜像名与导出时一致，通常可直接 build
docker build -t vllm-tream-portable:latest .

# 如果你想显式指定 base image：
# docker build --build-arg BASE_IMAGE=vllm-tream-recover:20260209 -t vllm-tream-portable:latest .
```

### 3.4 启动容器

```bash
# 最简启动
bash scripts/run_container.sh

# 推荐：挂载数据集、模型、HF缓存
# IMAGE=vllm-tream-portable:latest \
# CONTAINER=vllm-tream-portable \
# HOST_DATASET_DIR=/data/streaming-lvm-dataset \
# HOST_MODEL_DIR=/data/lvm-llama2-7b \
# HOST_HF_CACHE_DIR=/data/hf_cache \
# bash scripts/run_container.sh
```

## 4. 开箱即用实验命令（容器内）

进入容器：

```bash
docker exec -it vllm-tream-portable bash
cd /workspace/vllm/tream
```

### 4.0 8 卡并发（独占机器推荐）

远端 8 卡独占时，推荐用这个脚本直接 8 并发发射 AB48：

```bash
cd /workspace/vllm/tream

# 8 卡并发、静态+Spec
RUN_MODE=static RUN_SPEC=1 DATASET_SUBDIR=Ego4d \
GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
bash /workspace/vllm/tream/migration/scripts/launch_ab48_8gpu.sh
```

脚本输出：
- `inference_logs/ab48x8_*.status.log`
- `inference_logs/ab48x8_*.tsv`
- 每个 case 对应 `*.driver.log`

### 4.1 AB48 静态 NoSpec（原三卡 launcher）

```bash
RUN_MODE=static \
RUN_SPEC=0 \
DATASET_SUBDIR=Ego4d \
bash /workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 4.2 AB48 静态 SpecOnly

```bash
RUN_MODE=static \
RUN_SPEC=1 \
DATASET_SUBDIR=Ego4d \
bash /workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 4.3 AB48 Dynamic + Spec（NoJSD / JSD）

```bash
# NoJSD（关闭 token-shift 漂移触发）
RUN_MODE=dynamic \
RUN_SPEC=1 \
DATASET_SUBDIR=Ego4d \
SHIFT_SHOCK_DISABLE_DRIFT=1 \
bash /workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh

# JSD-only（只用 JSD 触发）
RUN_MODE=dynamic \
RUN_SPEC=1 \
DATASET_SUBDIR=Ego4d \
SHIFT_SHOCK_USE_JSD_ONLY=1 \
SHIFT_SHOCK_DISABLE_DRIFT=0 \
bash /workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

## 5. 重要说明

- `code_snapshot` 默认不包含超大资产（数据集、模型权重、wandb、历史日志）。
- 这些大文件建议在目标机器单独挂载到以下路径：
  - 数据：`/workspace/vllm/tream/data/streaming-lvm-dataset`
  - 主模型：`/workspace/vllm/tream/lvm-llama2-7b`
  - HF cache：`/workspace/vllm/hf_cache`
- 当前容器 `shm-size` 常用值是 `64g`；8 卡满并发建议 `64g~128g`。

## 6. 快速自检

```bash
python -V
python -m pip list | head
nvidia-smi
python /workspace/vllm/tream/tream.py --help
```
