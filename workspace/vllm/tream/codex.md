**use git to control version**
**use docker exec to execute program test  [daifeng@coriander daifeng]$ docker start -ai ef01962574e4
root@ef01962574e4:/workspace/tream#**


## 2026-02-22 Cold-start hardening update

### Behavior changes (scheduler_Shift)

- Added warm-up hold: first `TREAM_SHIFT_WARMUP_HOLD_WINDOWS` windows hold `x_safe_default` and do not enter ADAPT.
- Added force-adapt sample gate: quality-triggered force ADAPT only after enough trusted samples (`TREAM_SHIFT_FORCE_ADAPT_MIN_COUNT` and `TREAM_SHIFT_MIN_COUNT_FOR_TRUST`).
- Changed default cold-start safety behavior to strict mode (`TREAM_SHIFT_COLD_RELAX_SAFETY=0` by default in `SchedulerActor`).
- Tightened quality salvage:
  - choose only from `safe_set` (no unsafe candidate sweep),
  - prefer local reachable points,
  - penalize large `inf` jumps and prefer `ctx` moves under quality pressure.
- Improved recovery fallback:
  - `SafetyFilter` now prefers `x_safe_default` when safe, otherwise prefers a safe conservative template.
- Stabilized safe default:
  - `x_safe_default` is updated only when current window is feasible, instead of every switch.

### New/updated env knobs

- `TREAM_SHIFT_WARMUP_HOLD_WINDOWS` (default `4`)
- `TREAM_SHIFT_FORCE_ADAPT_MIN_COUNT` (default `max(TREAM_SHIFT_MIN_COUNT_FOR_TRUST, TREAM_SHIFT_WARMUP_HOLD_WINDOWS + 2)`)
- `TREAM_SHIFT_COLD_RELAX_SAFETY` default changed from `1` to `0`

# Repo scan + overall plan

* **Task:** Scan the current repository and identify existing actors/interfaces related to scheduling (InferenceActor, TrainingActor, SchedulerActor or equivalents).
* **Goal:** Produce an integration plan for a new modular scheduler package implementing “DSPFT framework + policies A/B/C” while minimizing coupling.
* **Output:**

  1. a short design doc section (in `codex.md`) describing where new code lives (e.g., `scheduler_Shift`), and what existing entrypoints will call it;
  2. a list of required runtime metrics fields and where to fetch them from;
  3. identify gaps (e.g., spec metrics / drift not currently surfaced) and propose concrete changes to emit them.
* **Constraints:**

  * Do not implement code yet.
  * Be explicit about filenames/classes/functions you will touch.

## Step 0 deliverable (repo scan results)

### 0.1 Integration boundary (minimal coupling)

- New package location: `scheduler_Shift/` (pure-python modular scheduler).
- Existing runtime entrypoints that will call into it:
  - `actors/scheduler_actor.py:SchedulerActor._step_control`
  - `actors/scheduler_actor.py:SchedulerActor._aggregate_metrics`
  - `actors/scheduler_actor.py:SchedulerActor._apply_config`
  - `actors/scheduler_actor.py:SchedulerActor.report_inference_metrics`
  - `actors/scheduler_actor.py:SchedulerActor.report_training_metrics`
- Existing actor update path to keep unchanged:
  - `actors/inference_actor.py:InferenceActor.update_config`
  - `actors/training_actor.py:TrainingActor.update_config`
- Driver entrypoint remains:
  - `tream.py` scheduler construction (`SchedulerActor.remote(...)`) and handle registration.

Proposed integration pattern:
- Keep `SchedulerActor` as adapter/orchestrator owner in Ray.
- Move decision logic into `scheduler_Shift/core.py:SchedulerCore`.
- `SchedulerActor` only does:
  - ingest raw reports,
  - build `WindowMetrics`,
  - call `SchedulerCore.step(window_metrics)`,
  - apply returned `Decision` through existing actor update APIs.

### 0.2 Required runtime metrics and current sources

- `lat_mean`, `lat_p95`:
  - source: `report_inference_metrics` payload (`latency`) aggregated in `_aggregate_metrics`.
- `acc_mean`:
  - source: `report_inference_metrics` payload (`accuracy`) aggregated in `_aggregate_metrics`.
- `train_tps`:
  - source exists in `TrainingActor` report payload (`tokens_per_second`), but currently not aggregated by scheduler.
- `mem_peak`:
  - source: `report_inference_metrics` payload (`mem_peak_gb`) aggregated as `mem_peak_p95`.
- `token_drift_mean` or `jsd_mean`:
  - scheduler can consume (`jsd`/`drift_jsd`/`jsd_mean`) but producer side currently does not emit it.
- spec internals (`spec_accept_mean`, `spec_reverify_per_step`, `spec_draft_ms_per_step`, `spec_verify_ms_per_step`, `rejected_tokens_per_step`, `accepted_tokens_per_step`):
  - currently written by vLLM to spec JSONL and tailed to W&B in `tream.py`,
  - not forwarded into scheduler report stream today.

### 0.3 Gaps and concrete changes to emit missing fields

- Gap A: no drift in inference report.
  - change: add `jsd_mean` (or `token_drift_mean`) into `InferenceActor` scheduler payload in `actors/inference_actor.py` near `report_inference_metrics.remote(...)`.
  - optional source strategy: rolling JSD from prediction-vs-truth cache deltas already tracked in actor state, fallback `None`.
- Gap B: no spec-internal metrics in scheduler input.
  - change: extend spec metrics tail path in `tream.py` to expose rolling aggregates (thread-safe) to `InferenceActor`, then inject into scheduler report payload.
  - alternate (cleaner): add a small `SpecMetricsBridge` helper under `scheduler_Shift/` and pass summary callback into actors.
- Gap C: training throughput not surfaced as scheduler objective.
  - change: in `actors/scheduler_actor.py:_aggregate_metrics`, aggregate `tokens_per_second` from `training_metrics` into `train_tps_mean`.
- Gap D: current scheduler config dimensions only `(ctx, inf)`.
  - change: keep current runtime control path for fast knobs `(ctx, inf)` and prepare optional slow template knobs `(ib, tb)` in data contract; wire slow knobs when actors expose template-switch API.

### 0.4 Files/classes/functions to touch in implementation phase

- New:
  - `scheduler_Shift/types.py`
  - `scheduler_Shift/config_space.py`
  - `scheduler_Shift/regime.py`
  - `scheduler_Shift/stats.py`
  - `scheduler_Shift/pareto.py`
  - `scheduler_Shift/candidates.py`
  - `scheduler_Shift/safety.py`
  - `scheduler_Shift/policies/base.py`
  - `scheduler_Shift/policies/anchor_only.py`
  - `scheduler_Shift/policies/direction.py`
  - `scheduler_Shift/policies/hier.py`
  - `scheduler_Shift/core.py`
- Existing (integration only, later):
  - `actors/scheduler_actor.py` (`_aggregate_metrics`, `_step_control`, `_apply_config`)
  - `actors/inference_actor.py` (scheduler report payload fields)
  - `actors/training_actor.py` (if extra train metrics needed)
  - `tream.py` (spec-metrics bridge wiring)
- Tests:
  - `tests/test_types.py`, `tests/test_config_space.py`, `tests/test_regime.py`, `tests/test_stats.py`, `tests/test_pareto.py`, `tests/test_candidates.py`, `tests/test_safety.py`, `tests/test_policies_anchor_only.py`, `tests/test_policies_direction.py`, `tests/test_policies_hier.py`, `tests/test_core_integration.py`

# step 1 — Data contract + config definitions

* **Task:** Create `scheduler_Shift` defining **dataclasses** and **type aliases** for:

  * `ConfigX`: (ctx:int, inf:int, ib:int|None, tb:int|None, …) including `to_dict/from_dict/hash_key`
  * `WindowMetrics`: all per-window fields (see below)
  * `DerivedMetrics`: verify_ratio, waste_rate, margins, shock components
  * `Decision`: next config + reason + predicted gains + safety margins
* **Required WindowMetrics fields:**

  * objectives/constraints: `lat_mean`, `lat_p95` (optional), `acc_mean`, `train_tps` (optional), `mem_peak` (optional)
  * drift: `token_drift_mean` (or `jsd_mean`)
  * spec internal: `spec_accept_mean`, `spec_reverify_per_step`, `spec_draft_ms_per_step`, `spec_verify_ms_per_step`, `rejected_tokens_per_step`, `accepted_tokens_per_step` (if available)
* **Task:** Add `scheduler_Shift/config_space.py`:

  * `ConfigSpace` with methods:

    * `all_configs()` (if small)
    * `neighbors(x, eps_fast, eps_slow)` (ctx/inf small step; ib/tb template step)
    * `orthogonal_probes(x, deltas)` (x±Δctx, x±Δinf, optional cross)
* **Output:** Implement code + unit tests in `tests/test_types.py` and `tests/test_config_space.py`.
* **Constraints:** Keep it pure Python, no heavy deps.

---

#  step 2 — Shock detector + mode state machine + regime quantization

* **Task:** Implement `scheduler_Shift/regime.py` containing:

  * `ShockDetector`:

    * computes `shock_score` = w1*|Δdrift| + w2*|Δaccept| + w3*|Δreverify| + w4*|Δverify_ratio| + w5*|Δwaste_rate|
    * hysteresis thresholds T_high/T_low
    * `mode ∈ {STABLE, ADAPT}` with `adapt_hold` counter
  * `RegimeQuantizer` (optional but recommended):

    * quantize (drift, accept, verify_ratio) into discrete bins → `regime_id`
* **Output:** code + tests verifying:

  * mode flips only when thresholds crossed
  * hysteresis prevents oscillation
  * regime_id is stable under small noise

---

# step 3 — Recency stats store

* **Task:** Implement `scheduler_Shift/stats.py`:

  * `EMAStats` tracking mean/variance for multiple metrics with configurable alpha
  * `StatsStore` keyed by `(regime_id, config_key)`:

    * `update(regime_id, x, metrics, alpha)`
    * `get_mu_sigma(regime_id, x, metric)`
    * tracks last_seen/count
  * Provide `confidence_bounds(mu, sigma, beta, direction)` helper:

    * LCB for “bigger is better” metrics (quality, train_tps)
    * UCB for “smaller is better” metrics (latency, mem)
* **Output:** code + tests: EMA correctness; bounds monotonic with beta.

---

# step 4 — Pareto + anchors

* **Task:** Implement `scheduler_Shift/pareto.py`:

  * `pareto_filter(points, minimize_mask)` returning non-dominated set
  * `AnchorManager`:

    * `update_anchors(candidates, mu_vectors, K)`
    * keep endpoints (min latency, max quality, max train_tps)
    * pick knee points (use crowding distance / simple curvature proxy)
    * fill remaining anchors for coverage (bin latency axis into buckets)
* **Output:** code + tests:

  * pareto correctness on synthetic points
  * anchors contain endpoints and remain size K
* **Note:** anchors are per regime: `anchors[regime_id]`.

---

# step 5 — Candidate generation

* **Task:** Implement `scheduler_Shift/candidates.py`:

  * `CandidateGenerator` producing `C_t`:

    * `C_base = anchors ∪ neighbors(x_prev)`
    * if mode==ADAPT: add `orthogonal_probes(x_prev)`
    * optional: add safe default config
    * cap size (e.g., 64/128) with dedup
* **Output:** code + tests:

  * ADAPT adds probes
  * STABLE does not
  * size cap respected

---

# step 6 — Safety filter

* **Task:** Implement `scheduler_Shift/safety.py`:

  * `SafetyFilter`:

    * For each candidate x, compute conservative bounds using StatsStore:

      * LCB_quality >= tau
      * UCB_latency <= SLA (optional)
      * UCB_mem <= mem_limit (optional)
      * LCB_train_tps >= train_min (optional)
    * Return `SafeSet` + per-candidate margin report
  * Implement salvage policy when SafeSet empty:

    * return `x_safe_default` or safest template
* **Output:** code + tests: filtering logic; empty-safe triggers salvage.

---

# step 7 — Decision policies（A/B/C 三个可插拔策略）

## step 7A — Policy interface + AnchorOnlyPolicy

* **Task:** Create `scheduler_Shift/policies/base.py` defining:

  * `DecisionPolicy` interface: `select_next(ctx: SchedulerContext) -> Decision`
  * `SchedulerContext` includes: mode, regime_id, x_prev, safe_set, pareto_set, anchors, stats_store, config_space, preference, switch_cost_lambda, dwell state
* **Task:** Implement `AnchorOnlyPolicy` in `scheduler_Shift/policies/anchor_only.py`:

  * choose x minimizing `preference_score(mu(x)) + λ*switch_cost(x,x_prev)`
  * STABLE: restrict to reachable small-step neighbors; enforce dwell/hysteresis
  * ADAPT: allow larger reachable set; allow 1-step “uncertainty exploration” if configured
* **Output:** tests verifying:

  * preference works
  * switch_cost influences choice
  * dwell prevents thrash

## step 7B — DirectionPolicy

* **Task:** Implement `DirectionPolicy` in `scheduler_Shift/policies/direction.py`:

  * ADAPT only:

    1. Build a small fit dataset from last W windows: features = x + z (drift/accept/reverify/verify_ratio/waste), labels = objectives
    2. Fit ridge regression per objective (no heavy libs; implement closed-form ridge with numpy)
    3. Derive J_x (sensitivity of objectives w.r.t params)
    4. Compute target direction in objective space:

       * if quality_margin < eta_q: prioritize increasing quality
       * if latency_margin < eta_l: prioritize decreasing latency
       * else follow preference (latency-first / knee)
    5. Choose Δx in discrete neighbor steps maximizing projected improvement and satisfying SafetyFilter bounds
  * STABLE: fallback to AnchorOnlyPolicy behavior
* **Output:** code + tests on synthetic data:

  * regression recovers known coefficients
  * direction chooses expected neighbor

## step 7C — HierPolicy

* **Task:** Implement `HierPolicy` in `scheduler_Shift/policies/hier.py`:

  * Outer loop (slow): switch `u=(ib,tb)` only when:

    * persistent infeasibility OR persistent degradation OR explicit alarm severity high
  * Inner loop (fast): within fixed u, delegate to AnchorOnlyPolicy or DirectionPolicy
  * Must account for higher switch cost for u changes
* **Output:** tests verifying outer loop rarely switches; inner loop still moves fast knobs.

---

# step 8 — Orchestrator

* **Task:** Implement `scheduler_Shift/core.py` with `SchedulerCore`:

  * `step(window_metrics: WindowMetrics) -> Decision`
  * Pipeline:

    1. derive metrics (verify_ratio/waste/margins)
    2. shock detect → mode
    3. regime quantize
    4. stats update for x_prev
    5. candidates generate
    6. safety filter → safe_set
    7. pareto + anchors update
    8. policy.select_next → decision
    9. update internal dwell state; set x_prev = decision.x_next
  * Provide `SchedulerConfig` for all thresholds/weights.
* **Output:** integration test `tests/test_core_integration.py`:

  * feed synthetic stream with a regime shift; verify mode switches and config changes.

---

# 2026-02-21 动态调度补齐（MVP）

## 目标判断（该做 / 可后移）

- 该做（已实现到代码）
  - `SchedulerCore.step()` 明确产出并记录：`P_hat_safe`（可信安全前沿）和 `anchors`（前沿摘要）。
  - `CandidateGenerator` 在 `ADAPT` 模式下保证 probe 不被 cap 截断（可配置最少保留数量）。
  - 前沿计算只用“近期可置信”点（`count` + `last_seen` 门控），避免陈旧点拖回历史前沿。
  - `SafetyFilter` 使用高置信约束：`UCB_latency <= SLA` / `LCB_quality >= tau`，并新增 `sigma_floor` + `min_count_for_trust`。
  - `safe_set` 为空时走 salvage 回退；若默认安全点也不安全，改走更保守模板并记录 `violation_recovery`。
  - 增加“结果兜底触发”：当 `quality_margin` 或 `latency_margin` 接近阈值时，即使 shock 低也强制 `ADAPT`。
  - 增加 `Probe -> Commit` 轻量状态：`ADAPT` 进入后前 N 个窗口做 probe，违规时中止 probe。
  - Inference -> Scheduler 补齐 drift/spec 字段透传，Training -> Scheduler 已聚合 `train_tps_mean`。
- 可后移（v2）
  - `safety` 从 `lat_mean` 升级为 `lat_p95`（MVP 默认 `lat_mean`，支持环境变量切换为 `lat_p95`）。
  - 更完整局部建模（DirectionPolicy 在线拟合强化）和慢变量层级切换（ib/tb fully online）。

## 前沿定义（MVP）

- `P_hat_safe`：当前 regime 下，`safe_set` 中满足可信门控（`count >= min_count_for_trust` 且未过期）的估计 Pareto 集。
- `anchors`：`P_hat_safe` 的压缩摘要（端点 + 覆盖点，`K=8` 默认，可调到 `8~16`）。
- 说明：`P_hat_safe` 为空时触发 trust fallback（至少保留当前点），并在日志中标记。

## 默认参数（本轮更新）

- `TREAM_SHIFT_COLD_START_WINDOWS=8`
- `TREAM_SHIFT_COLD_PROBE_EVERY=2`
- `TREAM_SHIFT_ACCEPT_SHOCK_DELTA=0.12`
- `TREAM_SHIFT_ACCEPT_SHOCK_COOLDOWN=1`
- `TREAM_SHIFT_COLD_AXIS_ROTATION=1`
- `TREAM_SHIFT_COLD_I_MAJOR_SPAN=3`（i 主导，周期性插入 c 探测）
- `TREAM_SHIFT_DEFAULT_QUALITY_MIN=0.5`（可选；未显式传 `--scheduler_quality_min` 时生效）
- `TREAM_SHIFT_COLD_WHITELIST_PROBE=1`（严格安全下的“死锁打破器”，仅在 `safe_set` 退化为单点时生效）
- `TREAM_SHIFT_COLD_WHITELIST_BUDGET=4`（冷启动白名单 probe 预算，耗尽后停止）
- `TREAM_SHIFT_COLD_WHITELIST_LAT_SLACK=0.15`（允许的延迟 margin 放宽幅度，防止激进探测）
- `TREAM_SHIFT_COLD_WHITELIST_Q_SLACK=0.6`（允许的质量 margin 放宽幅度，避免质量约束把候选全清空）
- 新增建议：
  - `TREAM_SHIFT_MIN_COUNT_FOR_TRUST=2`
  - `TREAM_SHIFT_MAX_STALENESS_TICKS=64`
  - `TREAM_SHIFT_SIGMA_FLOOR=0.01`
  - `TREAM_SHIFT_CANDIDATE_MIN_PROBE_KEEP=2`
  - `TREAM_SHIFT_ADAPT_PROBE_WINDOWS=2`
  - `TREAM_SHIFT_ADAPT_EXIT_Q_MARGIN=0.03`
  - `TREAM_SHIFT_ADAPT_EXIT_L_MARGIN=0.08`
  - `TREAM_SHIFT_SALVAGE_INFEASIBLE_STREAK=2`
  - `TREAM_SHIFT_SALVAGE_DWELL_STEPS=4`
  - `TREAM_SHIFT_MIN_SWITCH_GAIN_EPS=0.01`
  - `TREAM_SHIFT_FORCE_ADAPT_Q_MARGIN=0.02`
  - `TREAM_SHIFT_FORCE_ADAPT_L_MARGIN=0.05`
  - `TREAM_SHIFT_SAFETY_LATENCY_METRIC=lat_mean`（v2 可改 `lat_p95`）

## 2026-02-22 调度行为收紧（新增）

- ADAPT 现在是“有限 probe 预算”：进入 ADAPT 后仅前 `TREAM_SHIFT_ADAPT_PROBE_WINDOWS` 个窗口允许 `*_probe`，预算耗尽后强制回到 exploit（`anchor_only_score`）。
- ADAPT 退出条件升级为双门控：必须“连续低 shock（由 `adapt_hold` 计数）+ 质量/延迟 margin 健康（`TREAM_SHIFT_ADAPT_EXIT_Q_MARGIN/L_MARGIN`）”。
- 连续不可行时触发质量保底：`infeasible_streak >= TREAM_SHIFT_SALVAGE_INFEASIBLE_STREAK` 且质量仍不达标（或 safe_set 走了 salvage）时，切到 quality fallback 并施加 `TREAM_SHIFT_SALVAGE_DWELL_STEPS`。
- 增加全局切换抑制：不只 STABLE，ADAPT 也受 dwell 控制；且当预测收益 `< TREAM_SHIFT_MIN_SWITCH_GAIN_EPS` 时保持当前配置，避免 `(1,1)↔(2,1)` 抖动。

## 离线评估工具

- 新增 `tools/replay.py`：输入 scheduler JSONL + oracle CSV/JSONL，输出
  - `lat_regret`
  - `adapt_delay`
  - `violation_rate`
  - `switch_rate`
