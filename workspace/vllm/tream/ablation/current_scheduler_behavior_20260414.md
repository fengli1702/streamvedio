# 当前真实调度器行为总结（As-Is, 2026-04-14）

## 1) 证据范围
- 代码（当前生效实现）：
  - `testvllm/vllm/tream/scheduler_Shift/core.py`
  - `testvllm/vllm/tream/scheduler_Shift/phases.py`
  - `testvllm/vllm/tream/scheduler_Shift/seek_policy.py`
  - `testvllm/vllm/tream/scheduler_Shift/regime.py`
  - `testvllm/vllm/tream/actors/scheduler_actor.py`
- 启动参数（当前 AB48 三卡脚本）：
  - `testvllm/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh`
- 实际运行日志样本（C02）：
  - `testvllm/vllm/tream/inference_logs/scheduler_doh_shift_ab48x3_dyn_wb1b12j2_jsd1td0_cw15_C02_20260413_205249_wB_c8_i1_ib32_tb16.jsonl`
  - `testvllm/vllm/tream/inference_logs/doh_shift_ab48x3_dyn_wb1b12j2_jsd1td0_cw15_C02_20260413_205249_wB_c8_i1_ib32_tb16.driver.log`
  - `testvllm/vllm/tream/inference_logs/ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260413_205249.status.log`

---

## 2) 单个 scheduler step 的真实执行顺序
按 `core.py:294` 起始的 `step()`，当前顺序是：

1. 派生指标 `_derive_metrics`（`core.py:2243`）  
   - `shock_drift = jsd_mean`（若为空回退 `token_drift_mean`）。
2. 更新 warmup / cold-start 状态（`core.py:298-310`）  
   - 包含 cold-start 结束后 pause 计数装载。
3. ShockDetector 更新 mode（`core.py:314` + `regime.py:120`）。  
4. PhaseMachine 更新 phase（`core.py:363` + `phases.py:75`）。  
5. 候选生成（`core.py:375`），注意依赖 `mode`（ADAPT 可能放大候选池）。  
6. Safety 过滤（`core.py:397`），cold-start+relax 时可直接 `safe_set=candidates`。  
7. Pareto/anchors 更新（`core.py:417`）。  
8. 按 phase 选点（`core.py:507/524/555/587`）。  
9. 全局 switch guards（`core.py:2046`），包括 step-cap/dwell 等。  
10. anchor/trial 状态更新 + history 落账（`core.py:810`、`core.py:2332`）。

关键语义：`mode` 在 phase 选点前已经更新，所以“mode=ADAPT 但 phase=SEEK_ANCHOR”是可能且常见的（尤其 cold-start 期）。

---

## 3) 四阶段（含过渡）当前语义

### A. Warmup（`phases.py:78`，`core.py:448`）
- 条件：`total_steps < warmup_hold_windows`（`core.py:298`）。
- 行为：`warmup_hold_safe_default`，不做探索。
- 同时强制 mode 稳定：`core.py:316-319`。

### B. Cold-start / SEEK_ANCHOR（`phases.py:82-83`，`core.py:524`）
- 进入：warmup 结束且 `cold_start_active=true`。
- 主要动作：`seek_policy.select_next`。
- `seek` 真实策略（`seek_policy.py:98-110`）：
  - 优先下降子集（descending）；
  - 避免二周期；
  - 延迟压力下禁止上行；
  - 极端卡死时可走 `seek_anchor_whitelist_break`（`seek_policy.py:53-68`）。
- cold-start 活跃判定（`core.py:1032`）：
  - 步数未到 `cold_start_windows` 必活跃；
  - 达到“观测足够 + 连续无改进”可提前结束；
  - 到 `cold_start_max_windows` 强制结束。

### C. Post-cold pause（冷启动后停顿）
- 当 cold-start 从 true->false，装载 pause 计数（`core.py:305-309`）。
- pause 期直接 `post_cold_start_pause_hold`（`core.py:471-481`）。
- 每步递减（`core.py:839-844`）。

### D. Shock reroute + JSD local probe（`phases.py:86-88`，`core.py:555`）
- 只有 phase 到 `SHOCK_LOCAL_REROUTE` 且 `local_relax_active=true` 才会尝试 JSD probe（`core.py:1433`,`1442`）。
- `local_relax_active` 条件（`core.py:1049`）：
  - 非 warmup、非 cold-start、mode=ADAPT；
  - 且 `shock_drift >= local_relax_drift_threshold` 或剩余窗口未耗尽。
- JSD probe 状态机（`core.py:1420`）：
  - `probe_up -> probe_down -> commit`（可 `continue_probe/continue_commit`）；
  - axis 顺序按 `jsd_probe_axis_order`（默认 `ctx,inf`）。
- 邻域失效刷新（`core.py:1225`）：
  - 在 probe 启动时对邻域做 stats invalidate。
- probe 候选池来源（`core.py:1248`）：
  - 限定本地邻域；
  - 必须过 mem/train margin 门槛；
  - 来自 safe_set 或被 invalidate 的邻居。

---

## 4) 关键防护逻辑（当前生效）

### 4.1 非冷启动 step-cap（`core.py:1825`，调用 `core.py:2061`）
- 对 fast knobs（ctx/inf）做 phase-aware 步长上限；
- 若找不到可用一步候选，直接 `non_cold_step_cap_hold`（`core.py:1878-1888`）。

### 4.2 Dwell 防抖（`core.py:2070-2088` + `core.py:2332`）
- 切换后持有若干窗，减少来回抖动。

### 4.3 Reroute guard（`core.py:1750`）
- reroute 失败率过高会进入 cooldown，暂时不 reroute。

### 4.4 Salvage（`core.py:507`）
- 质量/可行性连坏时触发 `salvage_local_recovery`，优先保住约束。

---

## 5) 当前 AB48 动态脚本默认参数（核心项）
来自 `launch_ab48_trigpu_shift_taskpool.sh`：

- warmup: `2`（脚本导出到 `TREAM_SHIFT_WARMUP_HOLD_WINDOWS`）
- cold-start: `windows=15`, `max_windows=15`, `probe_every=1`
- cold probe 步长：`eps_fast=2`, `eps_slow=0`
- directional 开启：`prefer_large_descend=1`, `prefer_small_ascend=1`
- 目标区参数：`target_ctx=2`, `target_inf=2`, band=1（已导出）
- whitelist break：`enable=1`, `budget=12`, `max_switch=2`
- shock: `JSD-only=1`, `disable_drift=0`
- spec 动态实验默认开启（`RUN_SPEC=1`, `RUN_MODE=dynamic`）

说明：这些值通过 `scheduler_actor.py` 映射进 `SchedulerConfig`（见 `scheduler_actor.py:683` 起）。

---

## 6) C02（8,1）样本的真实行为（2026-04-13）

从 `scheduler_doh_shift_...C02...jsonl` 读取到 8 个调度窗：

- s0: warmup_hold_safe_default, `(8,1)`
- s1: warmup_hold_safe_default, `(8,1)`
- s2: seek_anchor, `(7,1)`
- s3: seek_anchor, `(6,1)`
- s4: seek_anchor, `(5,1)`
- s5: seek_anchor, `(4,1)`
- s6: seek_anchor, `(3,1)`
- s7: seek_anchor, `(2,1)`

即：该样本中 warmup 后进入 cold-start seek，并持续下压 ctx（8→2）。

该任务随后 `rc=134` 结束，driver log 显示在处理到约 frame 830 / step 51 时收到 `SIGTERM`；status log 记录为 `FAIL ... rc=134`。  
此失败属于进程级终止，不是调度器单步决策异常（日志可见持续正常调度到 s7）。

---

## 7) 当前“真实行为”结论（不含改动建议）
- 当前实现确实是：`warmup hold -> cold-start seek -> (可选)post-pause -> shock reroute/jsd -> hold/salvage`。
- cold-start 主体仍由 seek policy 驱动，具备下降偏好与白名单 break 兜底。
- JSD 不会在 warmup/cold-start 直接驱动 probe；它在 `SHOCK_LOCAL_REROUTE + local_relax_active` 才生效。
- 现网行为里，`mode=ADAPT` 与 `phase=SEEK_ANCHOR` 可同时出现（先 mode 后 phase+policy 的顺序决定）。
- 全局 guards（step-cap/dwell/reroute_guard）会限制大跨步与频繁回摆，部分窗会出现“该动但被防护拦下”的现象，这是当前设计内行为。

---

## 8) 运行时部件（包含 spec decoding / dyn）

### 8.1 Driver（`tream.py`）
- 负责启动 Ray、组装配置、创建 actor、主循环拉帧与日志汇总。
- 在 `--use_wandb` 下创建主 run，并把 scheduler JSONL 增量汇总到同一 run（`drain_scheduler_log_to_wandb`）。

### 8.2 InferenceActor
- 负责推理执行。
- 当 `--use_speculative_decoding` 开启时，启用 spec decoding（EAGLE3）：
  - `--spec_draft_model`
  - `--spec_vocab_mapping_path`
  - `--num_spec_tokens`
  - `--spec_disable_mqa_scorer`
- 同时 driver 会 tail `VLLM_SPEC_METRICS_PATH` 把 spec 指标写到 W&B。

### 8.3 TrainingActor
- 脚本默认带 `-train`，训练 actor 默认开启。
- 脚本默认也带 `-gc`（梯度检查点）。

### 8.4 SchedulerActor（动态调度入口）
- 条件：`(not --disable_dynamic_scheduling) and (num_workers == 1)`。
- 当前 AB48 启动脚本固定 `--num_workers 1`，所以在 `RUN_MODE=dynamic` 下会启用 SchedulerActor。
- 若 `num_workers>=2`，`tream.py` 会强制关动态调度（保护逻辑）。

### 8.5 三卡 task-pool launcher
- `launch_ab48_trigpu_shift_taskpool.sh` 会在 A/B/C 三张卡各起一个 Ray head，并行分发 48 case。
- 每个 case 都是独立 run_name / driver log / scheduler log。

---

## 9) 参数开关矩阵（怎么跑 spec / dyn / jsd）

以 `launch_ab48_trigpu_shift_taskpool.sh` 为准：

### 9.1 一级开关
- `RUN_SPEC`：
  - `1`：开启 speculative decoding（脚本会加 `--use_speculative_decoding` 及 draft/vocab 参数）
  - `0`：不加 spec 参数（等价关闭 spec）
- `RUN_MODE`：
  - `dynamic`：动态调度开启（不传 `--disable_dynamic_scheduling`，并注入整套 `TREAM_SHIFT_*`）
  - `static`：关闭动态调度（脚本加 `--disable_dynamic_scheduling`）

### 9.2 JSD/Token-shift 相关（二级开关）
- `SHIFT_SHOCK_USE_JSD_ONLY=1`：
  - 只让 drift/JSD 分量驱动 shock（accept/reverify/verify_ratio/waste 权重置 0，仍可记录日志）。
- `SHIFT_SHOCK_DISABLE_DRIFT=1`：
  - 直接把 drift 权重置 0，等价“禁用 token-shift/JSD 触发”。
- 常见组合：
  - `jsd 开`：`SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0`
  - `nojsd`：`SHIFT_SHOCK_DISABLE_DRIFT=1`（通常也保留 `SHIFT_SHOCK_USE_JSD_ONLY=1`）

### 9.3 当前脚本动态默认核心参数
- `SHIFT_COLD_START_WINDOWS=15`
- `SHIFT_COLD_START_MAX_WINDOWS=15`
- `SHIFT_COLD_PROBE_EVERY=1`
- `SHIFT_COLD_PROBE_EPS_FAST=2`
- `SHIFT_COLD_TARGET_CTX=2`
- `SHIFT_COLD_TARGET_INF=2`
- `SHIFT_COLD_WHITELIST_PROBE=1`
- `SHIFT_COLD_WHITELIST_BUDGET=12`
- `SHIFT_COLD_WHITELIST_MAX_SWITCH=2`
- `SHIFT_SHOCK_USE_JSD_ONLY=1`
- `SHIFT_SHOCK_DISABLE_DRIFT=0`

---

## 10) 可直接执行的命令模板

以下命令默认在目录 `testvllm/vllm/tream` 执行。

### 10.1 spec + dyn + jsd（当前主动态范式）
```bash
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=0 \
bash inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 10.2 spec + dyn + nojsd（动态但禁 token-shift 触发）
```bash
RUN_MODE=dynamic RUN_SPEC=1 \
SHIFT_SHOCK_USE_JSD_ONLY=1 SHIFT_SHOCK_DISABLE_DRIFT=1 \
bash inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 10.3 spec only（静态，开 spec，关 dyn）
```bash
RUN_MODE=static RUN_SPEC=1 \
bash inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 10.4 no feature（静态，关 spec，关 dyn）
```bash
RUN_MODE=static RUN_SPEC=0 \
bash inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 10.5 只跑指定 case（例如 C02）
```bash
RUN_MODE=dynamic RUN_SPEC=1 CASE_FILTER=C02 \
bash inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

### 10.6 从历史 run 自动补跑失败项
```bash
RUN_MODE=dynamic RUN_SPEC=1 SOURCE_RUN_ID=<旧run_id> \
bash inference_logs/launch_ab48_trigpu_shift_taskpool.sh
```

---

## 11) 一次运行会产出哪些日志文件

以 `RUN_ID=ab48x3_trigpu_shift_taskpool_<mode_tag>_<ts>` 为例：

- `<RUN_ID>.tsv`：任务清单（每条 case 的 run_name / gpu / driver_log）
- `<RUN_ID>.status.log`：发射过程状态、DONE/FAIL、汇总
- `doh_shift_...<case>....driver.log`：单条实验 stdout/stderr
- `scheduler_doh_shift_...jsonl`：调度器逐窗日志（phase/reason/shock/ctx/inf）
- `spec_metrics_...jsonl`：spec decoding 指标（仅 RUN_SPEC=1）

`mode_tag` 由脚本拼接：
- dynamic：`dyn_wb..._jsd..._cw...`
- static + no spec：`static_nodyn_nospec_...`
- static + spec：`static_nodyn_...`
