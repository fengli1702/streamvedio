# Scheduler 参数与运行说明（ABC24 当前实验）

## 1. 当前实验概况
- Run ID: `abc24x3_trigpu_shift_taskpool_20260310_184603`
- 启动脚本: `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/launch_abc24_trigpu_shift_taskpool.sh`
- 调度核心: `scheduler_Shift`
- 主要入口:
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/tream.py`
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/actors/scheduler_actor.py`
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/scheduler_Shift/core.py`

## 2. 现在在跑的“算法”到底有哪些
当前不是多套互斥算法，而是一套 `scheduler_Shift`，按优先级触发不同决策分支：

1. Shock + mode 切换（`STABLE/ADAPT`）
- 文件: `scheduler_Shift/core.py`
- 根据 drift/accept/verify_ratio 等指标更新模式。

2. 候选集生成（CandidateGenerator）
- 文件: `scheduler_Shift/candidates.py`
- 候选来源: `x_prev` + 邻域 + anchors + probes。

3. 安全过滤（SafetyFilter）
- 文件: `scheduler_Shift/safety.py`
- 用质量/时延/显存/训练吞吐的保守置信边界（LCB/UCB）筛选安全集合。

4. 冷启动探测（cold_start_probe）
- 文件: `scheduler_Shift/core.py`
- 支持局部步长（fast 轴 `eps`）和 axis rotation。

5. 冷启动白名单探测（cold_start_whitelist_probe）
- 文件: `scheduler_Shift/core.py`
- 当 safe_set 退化为单点且允许时，用白名单预算做“死锁解锁”探测。

6. 冲击探测（accept_shock_probe / adapt_probe_window）
- 文件: `scheduler_Shift/core.py`
- 当 accept shock 达阈值时，触发局部探测。

7. 主策略回退（anchor_only_score）
- 文件: `scheduler_Shift/core.py`, `scheduler_Shift/policies.py`
- 若前面探测分支不触发，走 anchor-only 打分。

8. 守卫器（switch guards）
- 文件: `scheduler_Shift/core.py`
- 包括非冷启动单步上限（non-cold step cap）、dwell hold、min-gain hold。

## 3. 我在启动脚本里设置的参数（ABC24）
下面是 `launch_abc24_trigpu_shift_taskpool.sh` 的当前默认值（可被同名环境变量覆盖）：

### 3.1 任务编排与资源参数
| 参数 | 当前值 | 说明 |
|---|---:|---|
| `MAX_FRAMES` | `4000` | 每个 case 最大处理帧数 |
| `IB` | `32` | inference batch size |
| `TB` | `16` | training batch size |
| `GPU_MEM_UTIL` | `0.9` | vLLM 可用显存比例 |
| `RAY_NUM_CPUS` | `16` | 每个进程 Ray CPU 配额 |
| `GPU_A/GPU_B/GPU_C` | `1/2/3` | 三卡并行 |
| `TOTAL_CASES` | `24` | A/B/C 各 8 组，任务池分发 |

### 3.2 Scheduler 关键参数（脚本设定）
| 环境变量 | 当前值 | 作用 |
|---|---:|---|
| `TREAM_SHIFT_COLD_START_WINDOWS` | `12` | 冷启动最少窗口数 |
| `TREAM_SHIFT_COLD_START_MAX_WINDOWS` | `24` | 冷启动最大窗口数 |
| `TREAM_SHIFT_COLD_PROBE_EVERY` | `1` | 每几窗口做一次冷启动探测 |
| `TREAM_SHIFT_COLD_PROBE_EPS_FAST` | `2` | 冷启动 fast 轴探测步长 |
| `TREAM_SHIFT_COLD_PROBE_EPS_SLOW` | `0` | 冷启动 slow 轴步长 |
| `TREAM_SHIFT_COLD_AVOID_TWO_CYCLE` | `1` | 避免 A->B->A 快速回摆 |
| `TREAM_SHIFT_COLD_PATIENCE_DIRECTIONS` | `8` | 冷启动“无改进”耐心阈值 |
| `TREAM_SHIFT_COLD_AXIS_ROTATION` | `1` | 冷启动轴轮换（i-major） |
| `TREAM_SHIFT_COLD_RELAX_SAFETY` | `1` | 冷启动是否放松安全筛选 |
| `TREAM_SHIFT_COLD_WHITELIST_BUDGET` | `12` | 白名单探测预算 |
| `TREAM_SHIFT_ACCEPT_SHOCK_DELTA` | `0.12` | accept shock 探测阈值 |
| `TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN` | `1` | accept shock 探测冷却 |
| `TREAM_SHIFT_MIN_COUNT_FOR_TRUST` | `1` | trust 最小样本数 |
| `TREAM_SHIFT_ADAPT_PROBE_WINDOWS` | `8` | ADAPT 探测预算窗口 |
| `TREAM_SHIFT_DEFAULT_QUALITY_MIN` | `0.2` | 默认质量阈值 |

### 3.3 运行中进程环境抽样（用于确认是否真正生效）
对当前正在运行的 `tream.py` 进程做环境变量抽样，看到如下值：

| 变量 | 抽样值 |
|---|---:|
| `TREAM_SHIFT_COLD_START_WINDOWS` | `12` |
| `TREAM_SHIFT_COLD_START_MAX_WINDOWS` | `24` |
| `TREAM_SHIFT_COLD_PROBE_EVERY` | `1` |
| `TREAM_SHIFT_COLD_PROBE_EPS_FAST` | `2` |
| `TREAM_SHIFT_COLD_PROBE_EPS_SLOW` | `0` |
| `TREAM_SHIFT_COLD_AVOID_TWO_CYCLE` | `1` |
| `TREAM_SHIFT_COLD_PATIENCE_DIRECTIONS` | `8` |
| `TREAM_SHIFT_COLD_RELAX_SAFETY` | `1` |
| `TREAM_SHIFT_COLD_WHITELIST_BUDGET` | `12` |
| `TREAM_SHIFT_MIN_COUNT_FOR_TRUST` | `1` |
| `TREAM_SHIFT_ADAPT_PROBE_WINDOWS` | `8` |
| `TREAM_SHIFT_ACCEPT_SHOCK_DELTA` | `0.12` |
| `TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN` | `1` |

注：如果你在 `scheduler_*.jsonl` 中观察到和上表不一致的值，优先按 `scheduler_*.jsonl` 判定“调度器真实行为”；再排查 Ray Actor 环境继承与启动时序问题。

## 4. 代码里可配置但当前脚本未显式设置的参数
这些参数由 `scheduler_actor.py` 读取环境变量；未设置时会走默认值：

| 参数 | 默认值（代码） | 说明 |
|---|---:|---|
| `TREAM_SHIFT_STABLE_EPS_FAST` | 空（回退 epsilon） | STABLE fast 邻域 |
| `TREAM_SHIFT_ADAPT_EPS_FAST` | 空（回退 epsilon_adapt） | ADAPT fast 邻域 |
| `TREAM_SHIFT_WARMUP_HOLD_WINDOWS` | `4` | warmup 长度 |
| `TREAM_SHIFT_COLD_IMPROVE_LAT_EPS` | `0.0` | 认定延迟改进的 epsilon |
| `TREAM_SHIFT_COLD_IMPROVE_Q_EPS` | `0.0` | 认定质量改进的 epsilon |
| `TREAM_SHIFT_ACCEPT_SHOCK_PROBE` | `1` | 开关：accept shock probe |
| `TREAM_SHIFT_COLD_I_MAJOR_SPAN` | `3` | i-i-i-c 轮换跨度 |
| `TREAM_SHIFT_COLD_WHITELIST_PROBE` | `1` | 开关：whitelist probe |
| `TREAM_SHIFT_COLD_WHITELIST_LAT_SLACK` | `0.15` | 白名单延迟松弛 |
| `TREAM_SHIFT_COLD_WHITELIST_Q_SLACK` | `0.6` | 白名单质量松弛 |
| `TREAM_SHIFT_MAX_STALENESS_TICKS` | `64` | trust 时效窗口 |
| `TREAM_SHIFT_FORCE_ADAPT_MIN_COUNT` | `max(min_count, warmup+2)` | force adapt 的最小样本 |
| `TREAM_SHIFT_SIGMA_FLOOR` | `0.01` | 安全过滤 sigma 下限 |
| `TREAM_SHIFT_SAFETY_LATENCY_METRIC` | `lat_mean` | `lat_mean` / `lat_p95` |
| `TREAM_SHIFT_CANDIDATE_MIN_PROBE_KEEP` | `2` | ADAPT 时保底 probe 数 |
| `TREAM_SHIFT_ADAPT_EXIT_Q_MARGIN` | `0.03` | ADAPT 退出质量边际 |
| `TREAM_SHIFT_ADAPT_EXIT_L_MARGIN` | `0.08` | ADAPT 退出时延边际 |
| `TREAM_SHIFT_SALVAGE_INFEASIBLE_STREAK` | `2` | salvage 触发阈值 |
| `TREAM_SHIFT_SALVAGE_DWELL_STEPS` | `max(2, t_dwell)` | salvage 驻留步数 |
| `TREAM_SHIFT_MIN_SWITCH_GAIN_EPS` | `0.01` | 最小切换收益 |
| `TREAM_SHIFT_FORCE_ADAPT_Q_MARGIN` | `0.02` | near-violation 质量阈值 |
| `TREAM_SHIFT_FORCE_ADAPT_L_MARGIN` | `0.05` | near-violation 延迟阈值 |
| `TREAM_SHIFT_POLICY` | `anchor_only` | `anchor_only/direction/hier` |

## 5. 哪些参数“必须设置”
为了避免实验不可复现，建议把下面参数视为必填（即使有默认值，也应显式传）：

### 5.1 必须显式设置（强烈建议）
1. `--scheduler_quality_min`
- 不设时会退化到代码分支默认行为，质量约束可能不一致。

2. `TREAM_SHIFT_COLD_RELAX_SAFETY`
- 直接决定冷启动是否走严格安全筛选。

3. `TREAM_SHIFT_MIN_COUNT_FOR_TRUST`
- 直接决定 safe_set/trusted_safe 是否退化为单点。

4. `TREAM_SHIFT_COLD_PROBE_EVERY`
- 决定探测频率，是“有效调度”与“慢速抖动”的关键开关。

5. `TREAM_SHIFT_COLD_PROBE_EPS_FAST`
- 决定冷启动是否允许更大步探索（例如 `±2`）。

6. `TREAM_SHIFT_COLD_WHITELIST_BUDGET`
- 影响 deadlock break 的可探测次数。

7. `--context_length` / `--inference_length` / `--inference_batch_size` / `--training_batch_size`
- 这四个直接定义实验工作负载，不可省略。

8. `--gpu_memory_utilization`
- 与 OOM/吞吐稳定性强相关。

### 5.2 建议显式设置
1. `TREAM_SHIFT_POLICY`（建议固定为 `anchor_only` 或你指定策略）
2. `TREAM_SHIFT_SAFETY_LATENCY_METRIC`（`lat_mean` 或 `lat_p95`）
3. `TREAM_SHIFT_FORCE_ADAPT_*`（明确 near-violation 触发逻辑）
4. `TREAM_SHIFT_MIN_SWITCH_GAIN_EPS`（避免无效回摆）

## 6. 实践建议（针对你当前的“看起来没调度”问题）
1. 先固定一版“最小可控参数集”再跑
- `COLD_RELAX_SAFETY`, `MIN_COUNT_FOR_TRUST`, `COLD_PROBE_EVERY`, `COLD_PROBE_EPS_FAST`, `COLD_WHITELIST_BUDGET`, `POLICY`。

2. 每轮跑完先看三件事
- unique config 数量（不是 switch 次数）。
- `safe_set_size` 与 `trusted_safe_size` 是否长期接近 1。
- `force_adapt_reasons` 是否长期为空。

3. 建议验收阈值
- 冷启动阶段至少覆盖 3 个以上 `inf` 值；
- 非冷启动不允许持续 `A<->B` 回摆超过 3 次。

## 7. 一键核对命令
```bash
# 当前 run id
cat /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/latest_abc24x3_trigpu_shift_taskpool_run.txt

# 进度
RUN_ID=$(cat /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/latest_abc24x3_trigpu_shift_taskpool_run.txt)
grep -c " DONE " /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/${RUN_ID}.status.log
grep -c " FAIL " /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/${RUN_ID}.status.log

# 查看某个 case 的调度原始日志
ls /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/scheduler_doh_shift_abc24x3_*${RUN_ID#abc24x3_trigpu_shift_taskpool_}*.jsonl
```

## 8. 备注
- 如果你希望，我可以下一步再出一份“参数对照表（脚本目标值 vs 调度日志实际值）”自动生成版，逐 case 给出差异和风险等级。

## 9. 函数级行为说明（按文件）

### 9.1 `scheduler_Shift/core.py`

#### A) 主流程函数
| 函数 | 作用 | 输入 | 输出 | 关键副作用 |
|---|---|---|---|---|
| `SchedulerCore.__init__` | 初始化调度状态、统计器、策略对象 | `config_space`, `initial_config`, `scheduler_config` | 无 | 初始化 mode/regime/history/probe budget 等状态 |
| `_build_policy` | 构建策略对象（anchor_only/direction/hier） | 无 | policy 实例 | 决定后续 `policy.select_next` 行为 |
| `step` | 每窗口完整调度循环（核心入口） | `WindowMetrics` | `Decision` | 更新 mode/regime/stats/anchors/history，更新 `x_prev` |

#### B) 探测与分支选择
| 函数 | 作用 | 触发条件 | 典型决策 reason |
|---|---|---|---|
| `_select_cold_start_probe` | 冷启动局部探测（支持 `eps_fast/slow`） | `cold_start_active` 且 probe 窗口 | `cold_start_probe` |
| `_select_cold_start_whitelist_probe` | safe_set 单点死锁时白名单探测 | whitelist 开启 + budget>0 + deadlock | `cold_start_whitelist_probe` |
| `_select_accept_shock_probe` | accept shock 触发的局部探测 | `accept_delta >= threshold` 且 cooldown 满足 | `accept_shock_probe` |
| `_select_quality_salvage_choice` | 质量/可行性恶化时回退选择 | infeasible streak 达阈值 | `quality_salvage_fallback` |

#### C) 约束与守卫
| 函数 | 作用 |
|---|---|
| `_apply_non_cold_step_cap` | 非冷启动阶段限制 fast 轴每步最多 `±1`，防大跳 |
| `_apply_switch_guards` | 统一执行 step cap、dwell hold、min-gain hold |
| `_probe_abort_due_to_violation` | 发现硬约束 violation 时中止探测 |
| `_is_non_cold_single_step` | 判断是否单步变化 |

#### D) 冷启动生命周期管理
| 函数 | 作用 |
|---|---|
| `_compute_cold_start_active` | 根据 min/max windows + patience 判定冷启动是否继续 |
| `_update_cold_start_progress` | 用上一窗口观测更新 `best_latency/quality` 与 no-improve streak |
| `_cold_start_preferred_axis` | 冷启动轴轮换（默认 i-major） |

#### E) 风险/可信度与统计辅助
| 函数 | 作用 |
|---|---|
| `_trusted_subset` | 对 safe_set 做 trust gate（count + staleness）过滤 |
| `_force_adapt_reasons` | near-violation 时生成 force-adapt 原因 |
| `_adapt_exit_ready` | 判定是否满足退出 ADAPT 条件 |
| `_should_force_quality_salvage` | 判定是否触发质量救援 |
| `_derive_metrics` | 从窗口指标导出 verify_ratio/waste_rate/margins |
| `_stats_payload` | 组装写入 `StatsStore` 的观测 payload |

#### F) 排序/目标与工具函数
| 函数 | 作用 |
|---|---|
| `_predicted_switch_gain` | 预测切换收益，用于 min-gain hold |
| `_mu_pair` | 取某配置在当前窗口下的 `(lat, acc)` 估计 |
| `_preference_score` | 根据 preference 计算标量目标 |
| `_mu_vectors` | 生成 Pareto/anchor 用目标向量 |
| `_pareto_configs` | 计算 Pareto 安全集 |
| `_alarm_severity` | 汇总告警强度 |
| `_update_state_after_decision` | 写回 dwell/adapt/probe 状态并更新 `x_prev_prev/x_prev` |
| `_append_history` | 维护 history 缓冲 |

### 9.2 `scheduler_Shift/safety.py`
| 函数 | 作用 | 关键点 |
|---|---|---|
| `SafetyFilter.__init__` | 初始化安全过滤参数 | 包含 `tau/sla/mem_limit/train_min/beta/sigma_floor/trust gate` |
| `filter_candidates` | 对候选做安全筛选 | safe_set 为空时进入 salvage |
| `_evaluate_one` | 单个候选的 LCB/UCB 安全判断 | active 约束下，未 trusted 视为 unsafe |
| `_effective_sigma` | sigma 下限处理 | 防止零方差过于乐观 |
| `_margin_ge/_margin_le` | 统一 margin 方向 | 质量用 `>=`，延迟/显存用 `<=` |
| `_salvage` | safe_set 为空时保底选择 | safe_default -> conservative -> least_violation |
| `_violation_penalty` | 负 margin 惩罚聚合 | 用于 least-violation 排序 |
| `_switch_distance` | 切换距离 | `ctx/inf` 差 + `ib/tb` 惩罚 |
| `_select_least_violation` | 最小违约配置 | 同时考虑接近 `x_prev` |
| `_select_conservative` | 保守模板安全候选 | 优先更保守方向 |

### 9.3 `scheduler_Shift/candidates.py`
| 函数 | 作用 | 关键点 |
|---|---|---|
| `CandidateGenerator.__init__` | 初始化候选生成器 | 控制 `cap_size` 和默认邻域 eps |
| `generate` | 生成本窗口候选集 | 组合 `x_prev`、anchors、neighbors、safe_default、probes |
| `generate._push_many` | 去重+截断写入工具 | 保证 cap 和顺序优先级 |

### 9.4 `actors/scheduler_actor.py`

#### A) 顶层工具函数
| 函数 | 作用 |
|---|---|
| `_safe_mean/_safe_p95/_safe_std` | 对可空序列做稳健统计 |
| `_finite_array` | 过滤非有限值 |
| `_normal_cdf` | 标准正态 CDF |
| `dominates/pareto_filter/update_pareto` | Pareto 支配维护 |
| `hypervolume_2d/hypervolume_3d` | 超体积计算 |

#### B) `SimpleGP`（轻量 GP 代理）
| 函数 | 作用 |
|---|---|
| `__init__` | 初始化 GP 超参与缓存 |
| `_kernel` | RBF 核 |
| `_estimate_lengthscale` | 长度尺度估计 |
| `fit` | 拟合 GP |
| `predict` | 输出均值方差 |

#### C) `SchedulerActor` 公共接口
| 函数 | 作用 |
|---|---|
| `__init__` | 构建 actor，读取 `TREAM_SHIFT_*` 并初始化 `SchedulerCore` |
| `register_actor_handles` | 注册 inference/training actor 句柄 |
| `get_current_config_tuple` | 返回当前配置 |
| `report_inference_metrics` | 接收推理指标 |
| `report_training_metrics` | 接收训练指标 |
| `report_inference_time` | 接收推理时延 |
| `report_training_time` | 接收训练时延 |

#### D) 控制与决策主逻辑
| 函数 | 作用 |
|---|---|
| `_aggregate_metrics` | 聚合窗口指标 |
| `_to_window_metrics` | 转成 `WindowMetrics` |
| `_step_control_shift` | 调用 `SchedulerCore.step` 做主决策 |
| `_step_control` | 控制步入口（含锁和节流） |
| `_log` | 写 scheduler jsonl 日志 |
| `_apply_config` | 应用新配置并更新版本 |

#### E) 探索/利用与约束模型（兼容旧 BO 路径）
| 函数 | 作用 |
|---|---|
| `_should_explore/_effective_epsilon/_reachable_candidates` | 旧探索逻辑基础组件 |
| `_select_exploit/_select_explore` | 旧 exploit/explore 分支 |
| `_fit_models/_predict_objectives/_predict_constraints` | GP 模型与预测 |
| `_expected_hv_improvement/_batch_acquisition/_optimize_acquisition` | qNEHVI 近似采集 |
| `_propose_next` | 旧路径下提议 next config |
| `_emergency_degrade` | 紧急降级策略 |

### 9.5 `tream.py`
| 函数/模块 | 作用 |
|---|---|
| `tail_and_log_spec_metrics` | 后台 tail speculative metrics 文件并转发到 wandb |
| `setup_spec_metrics_logging` | 设置 speculative metrics 日志线程与退出清理 |
| `if __name__ == "__main__"` 主流程 | 参数解析 -> `ray.init` -> 创建 `SchedulerActor/TrainingActor/InferenceActor` -> 流式处理帧并记录日志 |

## 10. 文件级 summary（快速定位版）
| 文件 | 一句话 summary | 常见改动点 |
|---|---|---|
| `scheduler_Shift/core.py` | 调度状态机与分支决策中枢 | probe 规则、guard、state 更新 |
| `scheduler_Shift/safety.py` | 安全约束与保底选择 | trust gate、margin、salvage |
| `scheduler_Shift/candidates.py` | 候选生成与去重裁剪 | eps/cap/probe keep |
| `actors/scheduler_actor.py` | 运行时接收指标并驱动 core 决策 | 环境变量映射、聚合窗口、日志字段 |
| `tream.py` | 实验入口与 actor 编排 | CLI 参数、Ray 初始化、日志输出 |

