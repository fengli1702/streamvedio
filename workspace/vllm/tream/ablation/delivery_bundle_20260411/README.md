# AB48 打包交付（2026-04-11）

## 1. 包含内容
- 四模式对比图（四柱图 + 三散点图）
- 典型调度曲线（低/中/高 workload，各含 `JSD` 与 `NoJSD` 两组）
- 调度器本体代码：`code/scheduler_Shift/`
- 重要 Actor 代码：
  - `code/actors/scheduler_actor.py`
  - `code/actors/inference_actor.py`
  - `code/actors/training_actor.py`
- 启动脚本：`run_configs/launch_ab48_trigpu_shift_taskpool.sh`
- 状态日志与说明：`logs/`

## 2. 本次运行参数（关键项）
### 2.1 共同参数（JSD / NoJSD）
- `RUN_MODE=dynamic`
- `RUN_SPEC=1`
- `MAX_FRAMES=4000`
- `IB=32, TB=16`
- `SHIFT_COLD_START_WINDOWS=15`
- `SHIFT_COLD_START_MAX_WINDOWS=15`
- `SHIFT_COLD_WHITELIST_PROBE=1`
- `SHIFT_COLD_WHITELIST_BUDGET=12`
- `SHIFT_COLD_WHITELIST_MAX_SWITCH=2`
- `SHIFT_SHOCK_USE_JSD_ONLY=1`
- `whitelist_slack_lat=0.15, whitelist_slack_q=0.60`（来自 status log）

### 2.2 JSD / NoJSD 唯一区别
- `JSD`：`SHIFT_SHOCK_DISABLE_DRIFT=0`
- `NoJSD`：`SHIFT_SHOCK_DISABLE_DRIFT=1`

### 2.3 Warmup / 冷启动 / 下界（代码默认）
- `warmup_hold_windows=2`（`scheduler_actor.py` 默认）
- `warmup_explore_enable=0`（默认）
- `cold_target=(2,2)`（默认）
- `context_bounds=(1,8), inference_bounds=(1,8)`（全局下界 1）

## 3. 典型运行轨迹（低/中/高 workload）
说明：下表字段为 `start -> end`、首次进入目标区步（`max(ctx,inf)<=3`）、冷启动结束步、首次 JSD 探测步。

| Case | 类型 | JSD | NoJSD |
|---|---|---|---|
| A01 | 低 workload | (1,1) -> (1,1), target_step=-, cold_end=15, first_jsd_probe=16 | (1,1) -> (2,1), target_step=-, cold_end=15, first_jsd_probe=36 |
| B03 | 中 workload | (4,4) -> (2,1), target_step=5, cold_end=15, first_jsd_probe=17 | (4,4) -> (2,1), target_step=5, cold_end=15, first_jsd_probe=36 |
| C04 | 高 workload | (8,2) -> (3,2), target_step=7, cold_end=15, first_jsd_probe=16 | (8,2) -> (2,1), target_step=7, cold_end=15, first_jsd_probe=36 |


## 4. 图文件位置
### 4.1 四模式对比图
- `plots/compare/ab48_four_modes_common_cases_bar_20260404.png`
- `plots/compare/ab48_four_modes_mean_bar_20260404.png`
- `plots/compare/ab48_scatter_jsd_vs_nojsd_20260404.png`
- `plots/compare/ab48_scatter_static_vs_jsd_20260404.png`
- `plots/compare/ab48_scatter_static_vs_nojsd_20260404.png`

### 4.2 典型曲线图
- 低 workload：`plots/trajectories/low_A01/`
- 中 workload：`plots/trajectories/mid_B03/`
- 高 workload：`plots/trajectories/high_C04/`

每组包含 4 张：
- `jsd_ctx_inf.png`
- `jsd_vs_oracle.png`
- `nojsd_ctx_inf.png`
- `nojsd_vs_oracle.png`

## 5. 运行轨迹概述（四阶段）
1. Warmup：前 2 窗稳态保持（默认不探索）。
2. Cold start：15 窗快速收敛到目标区附近（目标中心 `(2,2)`，下界允许到 1）。
3. 稳定段：进入目标区后短时保持，避免过度频繁切换。
4. 局部微调：
   - JSD 模式：由 drift/JSD 触发局部重路由（受 safety 与候选池约束）。
   - NoJSD 模式：仅依赖非 drift 信号，不启用 token-shift 漂移驱动。

## 6. 备注
- 当前 NoJSD 全量序列图为 47 条（缺 `C02` 的 `scheduler_*.jsonl`），已在 `logs/scheduler_sequence_plot_note_20260411.md` 记录。
- 四模式对比统计 CSV：`plots/compare/ab48_four_modes_latency_accuracy_20260404.csv`。
