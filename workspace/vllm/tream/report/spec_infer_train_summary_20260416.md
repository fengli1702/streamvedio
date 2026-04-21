# Spec 训练与 `spec*infer*train` 启动总结（2026-04-16）

## 1. 结论先说
当前仓库里用于实验的 spec 草稿模型是：
- `../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475`

当前主实验启动（含 `spec + infer + train`）主脚本是：
- `testvllm/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh`

单条动态实验（含 `spec + infer + train`）脚本是：
- `testvllm/vllm/tream/experiment/exp_doh_spec_dynamic.sh`

### 1.1 当时用的 Docker
当前这套实验环境对应容器/镜像是：
- container: `vllm-tream`
- image: `vllm-tream-recover:20260209`

快速确认命令：
```bash
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}' | rg 'vllm-tream|vllm-tream-recover'
```

进入容器：
```bash
docker exec -it vllm-tream bash
```

若需要重新拉起同镜像（示例）：
```bash
docker run -d --name vllm-tream --gpus all \
  -v /m-coriander/coriander/daifeng/testvllm/vllm:/workspace/vllm \
  -v /m-coriander/coriander/daifeng:/m-coriander/coriander/daifeng \
  --shm-size=64g \
  vllm-tream-recover:20260209 \
  bash -lc "sleep infinity"
```

---

## 2. draft 训练前的“数据 + tokenizer 转换”流程

你说的这一步是有的，核心是先把视频帧转换成 SpecForge 可训练的 `jsonl`，并确保 tokenizer 能对 VQ token 做可逆映射。

### 2.0 数据集具体来自哪里（原始来源与训练来源）
原始流式帧数据来源：
- `/workspace/vllm/tream/data/streaming-lvm-dataset/DOH`

转换脚本会把原始帧数据转换为 SpecForge 训练格式：
- 输出目录：`/workspace/SpecForge/datasets/lvm_stream_v1`
- 训练集：`/workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl`
- 验证集池：`/workspace/SpecForge/datasets/lvm_stream_v1/val.jsonl`
- 实际训练使用 eval：`/workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl`

当前实际条数（已落盘）：
- `train.jsonl`: 13,900 条
- `val.jsonl`: 5,925 条
- `val_sample_100.jsonl`: 100 条

即：draft 训练本体使用 13,900 条训练样本 + 100 条 eval 样本。

### 2.1 tokenizer 准备（identity tokenizer，VQ code 可逆）
脚本：
- `testvllm/vllm/SpecForge/scripts/build_identity_tokenizer.py`

示例命令（容器内）：
```bash
cd /workspace/vllm/SpecForge
python scripts/build_identity_tokenizer.py \
  --out-dir /workspace/vllm/tream/saved_models/lvm-llama2-7b
```

### 2.2 原始帧数据转换为 SpecForge 训练数据
脚本：
- `testvllm/vllm/tream/experiment/build_specforge_dataset.py`

示例命令（DOH 流数据 -> `train.jsonl` / `val.jsonl`）：
```bash
cd /workspace/vllm/tream
python experiment/build_specforge_dataset.py \
  --input-root /workspace/vllm/tream/data/streaming-lvm-dataset/DOH \
  --output-dir /workspace/SpecForge/datasets/lvm_stream_v1 \
  --tokenizer-path /workspace/vllm/tream/saved_models/lvm-llama2-7b \
  --context-length 4 \
  --inference-length 4 \
  --train-ratio 0.7 \
  --stride 1 \
  --batch-size 32 \
  --recursive \
  --verify-roundtrip 10 \
  --overwrite
```

训练时用的是：
- `/workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl`
- `/workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl`

如果只有 `val.jsonl`，可先切 100 条用于 eval：
```bash
head -n 100 /workspace/SpecForge/datasets/lvm_stream_v1/val.jsonl > /workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl
```

### 2.3 训练前可选：先构建 dataset cache + vocab mapping
脚本：
- `testvllm/vllm/SpecForge/scripts/build_eagle3_dataset_cache.py`

```bash
cd /workspace/vllm/SpecForge
python scripts/build_eagle3_dataset_cache.py \
  --target-model-path /workspace/tream/lvm-llama2-7b \
  --draft-model-config /workspace/SpecForge/configs/llama2-7B-eagle3.json \
  --train-data-path /workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl \
  --eval-data-path /workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl \
  --max-length 2048 \
  --chat-template llama2-lite \
  --cache-dir /workspace/SpecForge/cache/lvm_eagle3_v1_8292 \
  --build-dataset-num-proc 16
```

注：
- `train_eagle3.py` 训练时也会触发 dataset build / vocab mapping 逻辑，但离线先构建一遍更稳。

---

## 3. Spec 是怎么训练出来的（按历史 checkpoint 反解）

来源：
- `testvllm/vllm/SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475/training_state.pt`
- 通过解析 `training_state.pt` 内部 `training_state/data.pkl` 的 `argparse Namespace` 得到历史参数。

关键参数（真实历史值）：
- `target_model_path=/workspace/tream/lvm-llama2-7b`
- `draft_model_config=/workspace/SpecForge/configs/llama2-7B-eagle3.json`
- `train_data_path=/workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl`
- `eval_data_path=/workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl`
- `num_epochs=1`
- `total_steps=3475`
- `batch_size=4`
- `learning_rate=1e-4`
- `max_length=2048`
- `warmup_ratio=0.015`
- `max_grad_norm=0.5`
- `ttt_length=7`
- `chat_template=llama2-lite`
- `tp_size=1`, `dp_size=1`, `draft_accumulation_steps=1`
- `target_model_backend=sglang`
- `sglang_attention_backend=flashinfer`
- `cache_dir=/workspace/SpecForge/cache/lvm_eagle3_v1_8292`
- `output_dir=/workspace/SpecForge/output/lvm_eagle3_lora_v1`
- 最终 step：`epoch_0_step_3475`

### 3.1 复现训练命令（等价写法）
在容器内（示例工作目录 `/workspace/vllm/SpecForge`）：

```bash
cd /workspace/vllm/SpecForge
python scripts/train_eagle3.py \
  --target-model-path /workspace/tream/lvm-llama2-7b \
  --draft-model-config /workspace/SpecForge/configs/llama2-7B-eagle3.json \
  --train-data-path /workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl \
  --eval-data-path /workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl \
  --num-epochs 1 \
  --total-steps 3475 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --warmup-ratio 0.015 \
  --max-grad-norm 0.5 \
  --log-interval 50 \
  --ttt-length 7 \
  --chat-template llama2-lite \
  --tp-size 1 \
  --dp-size 1 \
  --draft-accumulation-steps 1 \
  --target-model-backend sglang \
  --sglang-attention-backend flashinfer \
  --cache-dir /workspace/SpecForge/cache/lvm_eagle3_v1_8292 \
  --output-dir /workspace/SpecForge/output/lvm_eagle3_lora_v1
```

---

## 4. 如何启动 `spec*infer*train`（单条）

主入口：`testvllm/vllm/tream/tream.py`

关键开关：
- `-train`：开启 TrainingActor
- `--use_speculative_decoding`：开启 spec infer
- `--spec_draft_model`：必填（开启 spec 时）
- `--spec_vocab_mapping_path`：建议必填
- `-gc`：梯度检查点

最小单条命令（动态调度开启，单 worker）：

```bash
python /workspace/vllm/tream/tream.py \
  --input_frames_path /workspace/vllm/tream/data/streaming-lvm-dataset/DOH \
  --model_name /workspace/vllm/tream/lvm-llama2-7b \
  --context_length 4 \
  --inference_length 4 \
  --inference_batch_size 32 \
  --training_batch_size 16 \
  --num_workers 1 \
  --max_frames 4000 \
  --gpu_memory_utilization 0.9 \
  --max_loras 1 \
  --wandb_run_name doh_spec_dyn_smoke \
  --inference_logs_dir /workspace/vllm/tream/inference_logs \
  --use_speculative_decoding \
  --spec_draft_model ../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475 \
  --spec_vocab_mapping_path vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt \
  --num_spec_tokens 3 \
  --spec_disable_mqa_scorer \
  -gc \
  -train
```

说明：
- `num_workers=1` 时可启用动态调度。
- 如果 `num_workers>=2`，`tream.py` 内会强制关闭动态调度（但 `spec + infer + train` 仍可运行）。

---

## 5. AB48 三卡批量启动（当前主范式）

脚本：`testvllm/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh`

### 5.1 开 spec + 开 dyn + 开 JSD
```bash
cd /workspace/vllm/tream/inference_logs
GPU_A=4 GPU_B=5 GPU_C=6 \
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh
```

### 5.2 开 spec + 开 dyn + 关 JSD（no token-shift）
```bash
cd /workspace/vllm/tream/inference_logs
GPU_A=4 GPU_B=5 GPU_C=6 \
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=1 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh
```

### 5.3 开 spec + 关 dyn（静态）
```bash
cd /workspace/vllm/tream/inference_logs
GPU_A=4 GPU_B=5 GPU_C=6 \
RUN_MODE=static RUN_SPEC=1 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh
```

---

## 6. 关键路径与参数文件

- 主驱动：`testvllm/vllm/tream/tream.py`
- 动态调度：`testvllm/vllm/tream/scheduler_Shift/core.py`
- 调度 actor：`testvllm/vllm/tream/actors/scheduler_actor.py`
- AB48 启动脚本：`testvllm/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh`
- 单条动态脚本：`testvllm/vllm/tream/experiment/exp_doh_spec_dynamic.sh`
- spec 训练脚本：`testvllm/vllm/SpecForge/scripts/train_eagle3.py`
- 当前 draft checkpoint：`testvllm/vllm/SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475`
- 当前 vocab mapping：`testvllm/vllm/tream/vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt`

---

## 7. 启动后的检查项（建议固定执行）

1. 检查 run 状态日志：
```bash
tail -n 50 /workspace/vllm/tream/inference_logs/<RUN_ID>.status.log
```

2. 检查单条 driver 日志是否包含 spec 参数：
```bash
rg -n "use_speculative_decoding|spec_draft_model|spec_vocab_mapping_path|num_spec_tokens|disable_dynamic_scheduling" \
  /workspace/vllm/tream/inference_logs/<run_name>.driver.log -S
```

3. 检查调度日志是否写出：
```bash
ls /workspace/vllm/tream/inference_logs/scheduler_<run_name>.jsonl
```

4. 检查 spec 指标日志是否写出：
```bash
ls /workspace/vllm/tream/inference_logs/spec_metrics_<run_name>.jsonl
```

---

## 8. 常见误区

- 误区 1：只开 `-train` 就算 `spec*infer*train`。  
  实际还必须加 `--use_speculative_decoding` 和 `--spec_draft_model`。

- 误区 2：`num_workers>=2` 还能开动态调度。  
  当前代码会在多 worker 模式下强制 `disable_dynamic_scheduling=True`。

- 误区 3：`spec_vocab_mapping_path` 可随便写。  
  必须对应当前 draft vocab mapping，建议使用 `bbea9992d144f6f64fd3cf54ec80f899.pt`。

---

## 9. 启动命令（已整理，可直接执行）

### 9.1 Host 侧一键进入容器
```bash
docker exec -it vllm-tream bash
```

### 9.2 容器内：完整准备与训练 draft（tokenizer + 数据转换 + cache + 训练）
```bash
# 1) tokenizer
cd /workspace/vllm/SpecForge
python scripts/build_identity_tokenizer.py \
  --out-dir /workspace/vllm/tream/saved_models/lvm-llama2-7b

# 2) 原始 DOH 帧 -> SpecForge train/val
cd /workspace/vllm/tream
python experiment/build_specforge_dataset.py \
  --input-root /workspace/vllm/tream/data/streaming-lvm-dataset/DOH \
  --output-dir /workspace/SpecForge/datasets/lvm_stream_v1 \
  --tokenizer-path /workspace/vllm/tream/saved_models/lvm-llama2-7b \
  --context-length 4 \
  --inference-length 4 \
  --train-ratio 0.7 \
  --stride 1 \
  --batch-size 32 \
  --recursive \
  --verify-roundtrip 10 \
  --overwrite

# 3) 取 100 条 eval
head -n 100 /workspace/SpecForge/datasets/lvm_stream_v1/val.jsonl > \
  /workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl

# 4) 可选：预构建 cache + vocab mapping
cd /workspace/vllm/SpecForge
python scripts/build_eagle3_dataset_cache.py \
  --target-model-path /workspace/tream/lvm-llama2-7b \
  --draft-model-config /workspace/SpecForge/configs/llama2-7B-eagle3.json \
  --train-data-path /workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl \
  --eval-data-path /workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl \
  --max-length 2048 \
  --chat-template llama2-lite \
  --cache-dir /workspace/SpecForge/cache/lvm_eagle3_v1_8292 \
  --build-dataset-num-proc 16

# 5) 训练 draft（与历史 checkpoint 参数一致）
python scripts/train_eagle3.py \
  --target-model-path /workspace/tream/lvm-llama2-7b \
  --draft-model-config /workspace/SpecForge/configs/llama2-7B-eagle3.json \
  --train-data-path /workspace/SpecForge/datasets/lvm_stream_v1/train.jsonl \
  --eval-data-path /workspace/SpecForge/datasets/lvm_stream_v1/val_sample_100.jsonl \
  --num-epochs 1 \
  --total-steps 3475 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --warmup-ratio 0.015 \
  --max-grad-norm 0.5 \
  --log-interval 50 \
  --ttt-length 7 \
  --chat-template llama2-lite \
  --tp-size 1 \
  --dp-size 1 \
  --draft-accumulation-steps 1 \
  --target-model-backend sglang \
  --sglang-attention-backend flashinfer \
  --cache-dir /workspace/SpecForge/cache/lvm_eagle3_v1_8292 \
  --output-dir /workspace/SpecForge/output/lvm_eagle3_lora_v1
```

### 9.3 容器内：单条 `spec*infer*train` 启动
```bash
cd /workspace/vllm/tream
python tream.py \
  --input_frames_path /workspace/vllm/tream/data/streaming-lvm-dataset/DOH \
  --model_name /workspace/vllm/tream/lvm-llama2-7b \
  --context_length 4 \
  --inference_length 4 \
  --inference_batch_size 32 \
  --training_batch_size 16 \
  --num_workers 1 \
  --max_frames 4000 \
  --gpu_memory_utilization 0.9 \
  --max_loras 1 \
  --wandb_run_name doh_spec_dyn_smoke \
  --inference_logs_dir /workspace/vllm/tream/inference_logs \
  --use_speculative_decoding \
  --spec_draft_model ../SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475 \
  --spec_vocab_mapping_path vocab_mapping/bbea9992d144f6f64fd3cf54ec80f899.pt \
  --num_spec_tokens 3 \
  --spec_disable_mqa_scorer \
  -gc \
  -train
```
