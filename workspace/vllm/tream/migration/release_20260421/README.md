# TREAM Migration Release (2026-04-21)

这个目录是可直接迁移到远程机器的整理版。

## 目录

- `no_docker/`：无 Docker 开箱运行（推荐）
- `launch/`：8卡并发 AB48 发射脚本
- `env_snapshot/`：当前可运行环境快照（版本/依赖/GPU/系统）
- `optional_docker/`：可选 Docker 方案（如果将来机器支持）

## 推荐路径（无 Docker）

1. 进入 `no_docker/` 按 `README.md` 执行 `setup.sh`。
2. 环境激活后，使用 `launch/launch_ab48_8gpu.sh` 发射。

示例：

```bash
TREAM_ROOT=/path/to/testvllm/vllm/tream \
RUN_MODE=static RUN_SPEC=1 DATASET_SUBDIR=Ego4d \
GPU_LIST=0,1,2,3,4,5,6,7 CONCURRENCY=8 \
bash launch/launch_ab48_8gpu.sh
```

## 注意

- 本项目依赖 **hack 版本地 vLLM**，不要直接只装 upstream wheel。
- 关键检查：`python -c "import inspect,vllm; print(inspect.getfile(vllm))"` 应指向本地仓库路径。
