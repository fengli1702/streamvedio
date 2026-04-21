# AB48 Ablation Skill (Spec / Dyn / JSD)

## 1) 目标与范式
本目录用于 AB48（`A01..F08` 共 48 点）消融实验与可视化。  
统一任务是同一组 48 参数点在 4 种模式下对比：

1. `NoFeature`：`no spec + no dyn`（静态，无 speculative decoding）
2. `SpecOnly`：`spec + no dyn`（静态，有 speculative decoding）
3. `Spec+Dyn(NoJSD)`：`spec + dyn`，但关闭 token-shift 漂移驱动（`td1`）
4. `Spec+Dyn+JSD`：`spec + dyn + JSD`（`td0`）

核心对比维度：`latency`（越低越好）和 `accuracy`（越高越好）。

---

## 2) 启动脚本与参数入口

### 2.1 主启动脚本
`/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh`

脚本关键参数（环境变量）：
- `RUN_MODE=static|dynamic`
- `RUN_SPEC=0|1`
- `GPU_A/GPU_B/GPU_C`
- `MAX_FRAMES`（建议固定 `4000`）
- `IB/TB`（建议固定 `32/16`）
- `CASE_FILTER`（重跑子集）
- `SOURCE_RUN_ID`（按旧 run 自动补跑缺失）

动态相关关键参数：
- `SHIFT_COLD_WHITELIST_PROBE`（是否开 whitelist break）
- `SHIFT_COLD_WHITELIST_BUDGET`
- `SHIFT_COLD_WHITELIST_MAX_SWITCH`
- `SHIFT_SHOCK_USE_JSD_ONLY`
- `SHIFT_SHOCK_DISABLE_DRIFT`（`td1` 表示禁 token-shift 漂移驱动）
- `SHIFT_COLD_START_WINDOWS` / `SHIFT_COLD_START_MAX_WINDOWS`

---

## 3) 四组实验启动命令（标准模板）

以下命令在容器 `vllm-tream` 内执行，统一三卡 `1/2/3`、`4000` 帧、`IB=32/TB=16`。

### 3.1 NoFeature（no spec, no dyn）
```bash
docker exec -d vllm-tream bash -lc '
cd /workspace/vllm/tream/inference_logs &&
GPU_A=1 GPU_B=2 GPU_C=3 \
RUN_MODE=static RUN_SPEC=0 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh > /workspace/vllm/tream/inference_logs/ab48_nofeature.launcher.log 2>&1'
```

### 3.2 SpecOnly（spec, no dyn）
```bash
docker exec -d vllm-tream bash -lc '
cd /workspace/vllm/tream/inference_logs &&
GPU_A=1 GPU_B=2 GPU_C=3 \
RUN_MODE=static RUN_SPEC=1 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh > /workspace/vllm/tream/inference_logs/ab48_spec_only.launcher.log 2>&1'
```

### 3.3 Spec+Dyn(NoJSD)（spec + dyn, td1）
```bash
docker exec -d vllm-tream bash -lc '
cd /workspace/vllm/tream/inference_logs &&
GPU_A=1 GPU_B=2 GPU_C=3 \
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_COLD_WHITELIST_PROBE=1 SHIFT_COLD_WHITELIST_BUDGET=12 SHIFT_COLD_WHITELIST_MAX_SWITCH=2 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=1 \
SHIFT_COLD_START_WINDOWS=15 SHIFT_COLD_START_MAX_WINDOWS=15 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh > /workspace/vllm/tream/inference_logs/ab48_dyn_nojsd.launcher.log 2>&1'
```

### 3.4 Spec+Dyn+JSD（spec + dyn + jsd, td0）
```bash
docker exec -d vllm-tream bash -lc '
cd /workspace/vllm/tream/inference_logs &&
GPU_A=1 GPU_B=2 GPU_C=3 \
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_COLD_WHITELIST_PROBE=1 SHIFT_COLD_WHITELIST_BUDGET=12 SHIFT_COLD_WHITELIST_MAX_SWITCH=2 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0 \
SHIFT_COLD_START_WINDOWS=15 SHIFT_COLD_START_MAX_WINDOWS=15 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh > /workspace/vllm/tream/inference_logs/ab48_dyn_jsd.launcher.log 2>&1'
```

### 3.5 失败补跑（示例）
```bash
docker exec -d vllm-tream bash -lc '
cd /workspace/vllm/tream/inference_logs &&
GPU_A=1 GPU_B=2 GPU_C=3 \
CASE_FILTER=B08,F07 \
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_COLD_WHITELIST_PROBE=1 SHIFT_COLD_WHITELIST_BUDGET=12 SHIFT_COLD_WHITELIST_MAX_SWITCH=2 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0 \
SHIFT_COLD_START_WINDOWS=15 SHIFT_COLD_START_MAX_WINDOWS=15 \
MAX_FRAMES=4000 IB=32 TB=16 \
./launch_ab48_trigpu_shift_taskpool.sh > /workspace/vllm/tream/inference_logs/ab48_retry.launcher.log 2>&1'
```

---

## 4) 画图脚本与启动方式

### 4.1 四模式总对比图脚本
`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/plot_ab48_four_modes_refresh.py`

运行：
```bash
cd /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation
python plot_ab48_four_modes_refresh.py
```

输出目录：
`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404`

关键输出图：
- `ab48_four_modes_mean_bar_20260404.png`
- `ab48_four_modes_common_cases_bar_20260404.png`
- `ab48_scatter_jsd_vs_nojsd_20260404.png`
- `ab48_scatter_static_vs_jsd_20260404.png`
- `ab48_scatter_static_vs_nojsd_20260404.png`
- `ab48_four_modes_latency_accuracy_20260404.csv`

### 4.2 调度轨迹图（ctx/inf/jsd）
脚本：`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/plot_ab48_done_scheduler_trajectories.py`

运行（示例：DONE=48 合并日志）：
```bash
python /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/plot_ab48_done_scheduler_trajectories.py \
  --status-log /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404/ab48_dyn_jsd_done48_merged_20260409.status.log \
  --inference-logs-dir /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs \
  --out-dir /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404/ab48x3_done48_ctx_inf_jsd_annotated_20260409
```

### 4.3 调度轨迹 vs Static Oracle 注释图
脚本：`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/plot_ab48_done_vs_static_oracle_annotated.py`

运行：
```bash
python /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/plot_ab48_done_vs_static_oracle_annotated.py \
  --status-log /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404/ab48_dyn_jsd_done48_merged_20260409.status.log \
  --inference-logs-dir /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs \
  --tau 0.5 --steps 41 \
  --out-dir /m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404/ab48x3_done48_annotated_vs_oracle_20260409
```

---

## 5) 要画哪些图 / 哪些对比

### 5.1 必画图
1. 四模式均值柱图（lat/acc）
2. 四模式共同样本柱图（逐 case）
3. 三张散点图：
   - `JSD vs NoJSD`
   - `Static vs JSD`
   - `Static vs NoJSD`
4. 每个 DONE case 的 `ctx/inf/jsd` 轨迹图
5. 每个 DONE case 的 `dynamic vs static-oracle` 注释图

### 5.2 必做对比
1. `SpecOnly` 相对 `NoFeature`：spec 本身收益
2. `Spec+Dyn(NoJSD)` 相对 `SpecOnly`：只靠冷启动/动态不含 token-shift 的收益
3. `Spec+Dyn+JSD` 相对 `Spec+Dyn(NoJSD)`：JSD 驱动局部调整的增量收益
4. `Spec+Dyn+JSD` 相对 `SpecOnly`：完整动态策略是否整体更优

---

## 6) 结果验收要求（含波动范围）

### 6.1 全量覆盖要求
1. 每组应覆盖 `48` 个 case（`A01..F08`）。
2. 对比图至少使用共同交集样本，优先保证 `48` 全覆盖。

### 6.2 调度行为要求
1. 冷启动阶段应把明显偏离点（如 `(8,1)/(1,8)`）向目标区间快速收敛（目标中心 `(2,2)`）。
2. 进入稳定后，JSD 触发应以局部探索为主，避免长期大幅反弹。

### 6.3 波动范围（建议作为硬检查）
1. `shock` 参考阈值：`t_low=0.2`、`t_high=0.35`（图上画线）。
2. `JSD` 局部放宽阈值参考：`~0.3`（来自 `local_relax_drift_threshold`）。
3. 非冷启动阶段，`ctx/inf` 应优先局部波动（通常 ±1 邻域）；若连续多窗跳回高负载区（如 `max(ctx,inf)>=6`）需标记为异常。
4. 冷启动后若进入目标区（如 `max(ctx,inf)<=3`）应有可观察的保持段，再进入 JSD 微调。

---

## 7) 推荐结果目录（当前）

1. 四模式对比图目录：  
`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404`

2. DONE=48 轨迹图目录：  
`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404/ab48x3_done48_ctx_inf_jsd_annotated_20260409`

3. DONE=48 对照 oracle 目录：  
`/m-coriander/coriander/daifeng/testvllm/vllm/tream/ablation/four_mode_compare_20260404/ab48x3_done48_annotated_vs_oracle_20260409`

