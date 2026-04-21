# TREAM 调度器阶段报告（2026-03-08）

## 1) 这套东西“整体怎么跑”

### 1.1 运行主链路
- 入口：`tream.py`
- 条件：当 `--disable_dynamic_scheduling` **未开启** 且 `--num_workers 1` 时，启用调度器（`SchedulerActor`）。
- 数据流：
  - `InferenceActor` / `TrainingActor` 上报窗口指标；
  - `SchedulerActor._aggregate_metrics()` 聚合为窗口统计；
  - `SchedulerActor._to_window_metrics()` 转成 `WindowMetrics`；
  - `scheduler_Shift/core.py:SchedulerCore.step()` 输出 `Decision`；
  - `SchedulerActor._apply_config()` 下发新 `(ctx, inf)` 到推理/训练 actor。

### 1.2 两类实验脚本
- 静态网格（不启用动态调度）：
  - `experiment/exp_doh_spec_static_grid_a.sh`
  - `experiment/exp_doh_spec_static_grid_b.sh`
  - 双卡并行入口：`experiment/run_doh_spec_static_dual.sh`
- 动态调度 AB 套件（启用 scheduler_Shift）：
  - AB 各 4 条：`inference_logs/launch_ab4_dualgpu_shift.sh`
  - AB 各 8 条：`inference_logs/launch_ab16_dualgpu_shift_v2.sh`

### 1.2.1 全量实验到底用哪个脚本
- 要跑**静态全量网格**（A+B 全覆盖，198 个组合）：
  - 用 `experiment/run_doh_spec_static_dual.sh`
  - 其中 A 覆盖 `ctx+inf<=8`，B 覆盖 `8<ctx+inf<=12`，每个 `(ctx,inf)` 组合再乘 `ib∈{8,16,32}`、`tb∈{8,16,32}`。
  - 组合数：A=117，B=81，总计=198。
- 要跑**动态全量集合**（当前定义的 16 组代表点）：
  - 用 `inference_logs/launch_ab16_dualgpu_shift_v2.sh`
  - A 队 8 组 + B 队 8 组，共 16 条，固定 `ib=32,tb=16`，每条默认 `max_frames=4000`。
- 要跑**动态快速回归**（你常用的 AB 各四条）：
  - 用 `inference_logs/launch_ab4_dualgpu_shift.sh`

### 1.3 标准运行命令（容器内）
```bash
cd /workspace/tream

# 动态 AB4（两卡）
GPU_A=5 GPU_B=7 GPU_MEM_UTIL=0.62 bash inference_logs/launch_ab4_dualgpu_shift.sh

# 动态 AB16（两卡）
GPU_A=5 GPU_B=7 GPU_MEM_UTIL=0.62 bash inference_logs/launch_ab16_dualgpu_shift_v2.sh

# 静态 baseline（两卡）
GPU_A=5 GPU_B=7 MAX_FRAMES=4000 bash experiment/run_doh_spec_static_dual.sh
```

### 1.3.1 建议的“全量执行顺序”
1. 静态全量：`experiment/run_doh_spec_static_dual.sh`（拿 baseline 面）
2. 动态 AB16：`inference_logs/launch_ab16_dualgpu_shift_v2.sh`（看调度行为与稳定性）
3. 动态 AB4：`inference_logs/launch_ab4_dualgpu_shift.sh`（做参数回归/快速迭代）

---

### 1.4 参数字典（你后续改脚本时最常用）

### 1.4.1 通用运行参数（`tream.py`）
- `--context_length`：上下文长度（调度快旋钮之一，`ctx`）。
- `--inference_length`：推理长度（调度快旋钮之一，`inf`）。
- `--inference_batch_size`：推理 batch（`ib`）。
- `--training_batch_size`：训练 batch（`tb`）。
- `--max_frames`：每条实验的总帧数（验收通常看 4000）。
- `--gpu_memory_utilization`：vLLM 可占用显存比例上限。
- `--disable_dynamic_scheduling`：开启则关闭动态调度（静态脚本会带这个参数）。
- `--scheduler_quality_min`：质量下限（当前动态脚本默认 0.5）。
- `--use_speculative_decoding` / `--spec_*`：spec decode 相关开关与模型路径。
- `--ray_address` / `--ray_namespace`：连接到对应 Ray head，避免多实验互相干扰。

### 1.4.2 静态脚本参数（`run_doh_spec_static_dual.sh` 及 A/B）
- `GPU_A`,`GPU_B`：A/B 队列使用的 GPU。
- `MAX_FRAMES`：覆盖每条静态组合的帧数。
- `RUNNER`：Python 启动器（默认 `python`）。
- `RAY_PORT_A`,`RAY_PORT_B`：A/B 两个独立 Ray 端口。
- `RAY_TEMP_DIR_A`,`RAY_TEMP_DIR_B`：Ray 临时目录。
- `RUN_TAG`,`RUN_PREFIX`：run name 前缀与标签。
- `OVERWRITE`：是否覆盖已存在日志（`1` 覆盖，默认跳过已完成组合）。
- `DRY_RUN`：只打印命令不执行。

### 1.4.3 动态脚本参数（AB4/AB16 启动器）
- `GPU_A`,`GPU_B`：两张卡分别跑 A/B 队列。
- `GPU_MEM_UTIL`：每条动态 run 的显存利用率上限。
- `MAX_FRAMES`：每条动态 run 帧数（脚本默认 4000）。
- 固定实验维度：
  - AB4：8 条（A4+B4），`ib=32,tb=16`。
  - AB16：16 条（A8+B8），`ib=32,tb=16`。

### 1.4.4 动态调度关键 env（由 `actors/scheduler_actor.py` 读取）
- 冷启动相关：
  - `TREAM_SHIFT_WARMUP_HOLD_WINDOWS`：前 N 窗口固定保守配置（只积累统计）。
  - `TREAM_SHIFT_COLD_START_WINDOWS`：冷启动窗口总长度。
  - `TREAM_SHIFT_COLD_PROBE_EVERY`：冷启动 probe 频率（每几窗一次）。
  - `TREAM_SHIFT_COLD_AXIS_ROTATION`：是否做 ctx/inf 维度轮转探测。
  - `TREAM_SHIFT_COLD_I_MAJOR_SPAN`：i 维优先轮转跨度（控制 i:i:i:c 比例）。
  - `TREAM_SHIFT_COLD_RELAX_SAFETY`：冷启动是否放宽安全过滤（建议 0）。
  - `TREAM_SHIFT_COLD_WHITELIST_PROBE`：严格安全下是否允许白名单探测打破锁死。
  - `TREAM_SHIFT_COLD_WHITELIST_BUDGET`：白名单探测预算（最多几次）。
  - `TREAM_SHIFT_COLD_WHITELIST_LAT_SLACK`：白名单探测可接受的延迟 margin 放宽。
  - `TREAM_SHIFT_COLD_WHITELIST_Q_SLACK`：白名单探测可接受的质量 margin 放宽。
- ADAPT 相关：
  - `TREAM_SHIFT_ACCEPT_SHOCK_PROBE`：是否启用 acceptance shock 触发的 probe。
  - `TREAM_SHIFT_ACCEPT_SHOCK_DELTA`：accept 波动触发阈值（常用 0.12/0.15）。
  - `TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN`：shock probe 冷却窗口。
  - `TREAM_SHIFT_ADAPT_PROBE_WINDOWS`：进入 ADAPT 后可用 probe 窗口预算。
  - `TREAM_SHIFT_ADAPT_EXIT_Q_MARGIN` / `TREAM_SHIFT_ADAPT_EXIT_L_MARGIN`：退出 ADAPT 的质量/延迟 margin 门槛。
  - `TREAM_SHIFT_FORCE_ADAPT_MIN_COUNT`：强制 ADAPT 前的最小可信样本量。
  - `TREAM_SHIFT_FORCE_ADAPT_Q_MARGIN` / `TREAM_SHIFT_FORCE_ADAPT_L_MARGIN`：近违规触发 ADAPT 门槛。
- 安全/稳定性：
  - `TREAM_SHIFT_MIN_COUNT_FOR_TRUST`：进入可信估计的最小样本。
  - `TREAM_SHIFT_MAX_STALENESS_TICKS`：统计过期阈值。
  - `TREAM_SHIFT_SIGMA_FLOOR`：置信下界 sigma 下限，防止过度自信。
  - `TREAM_SHIFT_SAFETY_LATENCY_METRIC`：`lat_mean` 或 `lat_p95`（当前默认 `lat_mean`）。
  - `TREAM_SHIFT_SALVAGE_INFEASIBLE_STREAK`：连续不可行多少窗后启用 salvage。
  - `TREAM_SHIFT_SALVAGE_DWELL_STEPS`：salvage 后强制停留窗口数。
  - `TREAM_SHIFT_MIN_SWITCH_GAIN_EPS`：切换收益阈值（太小则不切）。
  - `TREAM_SHIFT_DEFAULT_QUALITY_MIN`：默认质量下限（脚本里常设 0.5）。

### 1.5 结果与日志
- 动态运行主状态：
  - `inference_logs/ab4_dualgpu_shift_<ts>.status.log`
  - `inference_logs/ab16_dualgpu_shift_<ts>.status.log`
- 调度器原始 JSONL：
  - `inference_logs/scheduler_doh_shift_*.jsonl`
- 可读序列汇总（已用过）：
  - `inference_logs/scheduler_seq_reason_20260221_062703.txt`
  - `inference_logs/scheduler_seq_reason_20260222_040150.txt`
  - `inference_logs/scheduler_seq_reason_20260222_061820.txt`
- AB16 的序列导出脚本（当前是按该批次命名）：
  - `inference_logs/gen_scheduler_seq_reason_ab16_20260222_061820.sh`

---

## 2) 代码进度总结（你我这段周期）

最近调度器相关提交（节选）：
- `b493e3728`：增加受限冷启动白名单探测（用于 strict-safety 下打破“单点锁死”）。
- `66fc9ae76`：冷启动与 salvage 安全链路加固。
- `a719d88e4`：ADAPT 探测预算 + 退出门控。
- `b4dde8299`：冷启动维度轮转、质量/置信相关参数。
- `b1276c2b0`：probe tie-break 改为偏局部渐进移动。
- `5b282617c`：前沿/安全与 replay 工具打通（`tools/replay.py`）。
- `eeeed1720`：接入 spec 指标用于 shock/probe。
- 更早一组提交：完成 `scheduler_Shift` 模块化骨架、策略、集成替换。

当前 `scheduler_Shift` 已具备：
- ShockDetector（滞回 + hold）；
- RegimeQuantizer；
- StatsStore（EMA + 置信边界）；
- SafetyFilter（UCB/LCB 约束与 salvage）；
- Candidate/anchors/Pareto；
- 策略插件（anchor/direction/hier）；
- 冷启动、accept-shock probe、ADAPT probe budget、dwell/hysteresis、quality salvage、force-adapt near-violation。

---

## 3) 已有实验结果（关键批次）

### 3.1 批次 A：`scheduler_seq_reason_20260222_040150.txt`（AB4）
- 8 个子实验
- 总切换 `131` 次（平均 `16.375` 次/实验）
- 现象：明显“过度切换”，早期存在大量 `quality_salvage_fallback`，有些实验出现扫参式行为和抖动。

### 3.2 批次 B：`scheduler_seq_reason_20260222_061820.txt`（AB16）
- 16 个子实验
- 总切换 `0` 次（全部保持初始点）
- 现象：从“过度抖动”转成“完全不动”，说明门控/安全侧过强，探索触发不足或被抑制。

### 3.3 更早批次：`scheduler_seq_reason_20260221_062703.txt`
- 8 个子实验
- 总切换 `235` 次（平均 `29.375` 次/实验）
- 现象：典型 early-stage 高频切换，后续优化是为抑制这一问题。

---

## 4) 当前判断（结论）

- 功能面：调度器框架与关键机制已实现到“可实验迭代”阶段。
- 行为面：目前在“过动”和“不动”之间摆动，尚未稳定达到“有意义、可控地动态切换”。
- 工具面：日志与回放评估链路已经有基础（含 `tools/replay.py`），可继续做量化闭环。

### 4.1 我们当前对调度器的“期望行为”
- 冷启动阶段（前几窗）：
  - 允许有限探测，但应“局部渐进”，不应出现远距离大跳扫参。
  - 若安全集单点锁死，允许白名单探测打破死锁，但预算有限。
- ADAPT 阶段：
  - 探测预算有限（N 窗后强制 exploit），不能长期 `*_probe` 抖动。
  - Shock 降低且 margin 恢复后应回 STABLE（滞回 + hold 生效）。
- 约束：
  - 延迟不应长期越过 SLA；
  - 质量低于 `tau` 时要触发恢复策略，但恢复路径也必须受安全约束。
- 结果形态：
  - 不应是“几乎每窗都切”，也不应是“4000 帧全程不切”；
  - 理想是“变化阶段有切换、稳态阶段少切换”。

### 4.2 建议验收 KPI（可落地）
- `switch_rate`：相对早期高抖动明显下降，但非零（避免完全锁死）。
- `violation_rate`：低于当前基线，尤其 SLA 违规不能集中出现在 salvage 阶段。
- `lat_regret`：相对静态点有收益（或至少不恶化）。
- `adapt_delay`：在 oracle 变化后有可接受延迟内响应。

---

## 5) 立即可执行的下一步（建议）

1. 用最新 commit（含 `cold_whitelist_probe`）重跑一批 AB4，优先验证“从锁死中小步解锁”是否出现。  
2. 固化一个统一的“序列导出脚本”命名（不要带日期后缀），避免每批次手改。  
3. 对同一批次固定输出四项指标：`switch_rate / violation_rate / lat_regret / adapt_delay`（可直接用 `tools/replay.py`）。  
4. 如果重跑仍 0 切换：优先下调保守门控（force_adapt gate 与 whitelist slack/预算）再试。  

---

## 6) 当前环境说明（简）

- 代码仓：`main` 分支本地领先远端，且存在大量非本任务改动文件（仓库本身很脏，已避免触碰无关内容）。
- 近期动态实验（AB4/AB16）历史日志均显示 `FINISH ... all_success`，说明当时运行链路是通的。
- 你刚提到的容器启动问题单独存在，不影响本报告对“如何跑 + 进度”的梳理。

---

## 7) AGENTS/技能目录说明（按你要求补充）

- 当前会话的 AGENTS 指令作用域是：
  - `/m-coriander/coriander/daifeng`
- 可用 skill 文件目录（会话给出的路径）：
  - `/home/daifeng/.codex/skills/.system/skill-creator/SKILL.md`
  - `/home/daifeng/.codex/skills/.system/skill-installer/SKILL.md`
- 说明：
  - 目前仓库里未检索到实体 `AGENTS.md` 文件（`find`/`rg` 结果为空），当前 AGENTS 规则是会话注入生效。
