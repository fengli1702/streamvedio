你真正要保住的东西

你最在意的是这条：

先把 (2,8) 这类坏点压到 (2,2) 一带；然后不要再大范围乱跳，而是在 (2,2) 附近搜。

这个目标我建议拆成两个阶段来实现。

阶段 1：Compression seek

职责只有一个：

把明显偏离的 workload 压回合理带。

这一段现在其实已经做得不错，靠的就是：

warmup hold safe default

seek 的 descending-only

latency pressure 下禁止上行

后续 hold/dwell 防回弹
这些都不该动。

阶段 2：Neighborhood search

一旦已经进入合理带，比如到了 (2,2) 附近，目标就不再是继续“压”，而是：

围绕当前点/anchor 做小半径搜索。

这一步才应该让 JSD 发挥作用。

修改方案

我给你一版最小风险修改方案，只动入口和执行顺序，不改 seek/hold/reroute 策略函数主体。

方案总原则
不动的

seek_policy.py

hold_policy.py

reroute_policy.py

现有 cold start 的大逻辑

要动的

types.py

regime.py

core.py

少量 anchor.py / pareto.py / safety.py 的提交与放宽逻辑

第一部分：先把 drift 收成唯一 JSD
改动目标

在线调度只认一个漂移定义：

drift := jsd_mean

文件
types.py

WindowMetrics.drift_value() 直接返回 jsd_mean

token_drift_mean 视为废弃字段，不再参与在线调度

上游 JSD 上报链路

继续用你现在的 JSD 实现

但只保留 jsd_mean

低 token 数窗口输出 invalid/None，不要写 0

目的

先把“shift 信号”变干净，不再有双口径回退。当前 core.step() 的 regime quantize 和 shock 都是从 window_metrics.drift_value() 开始的，所以这里只要收口，后面整条链都会干净。

第二部分：把 ShockDetector 改成 JSD-only
改动目标

只保留一套 shift 系统：

shift 只由 JSD 决定

文件
regime.py

把现在的五分量 shock：

drift

accept

reverify

verify_ratio

waste_rate

改成只看 drift。

最小改法不是重写类，而是：

保留 ShockDetector

但只保留 drift 分量

其余分量权重设为 0 或直接移除

目的

让 shift 的语义重新回到“环境/内容分布变了”，不再混入 runtime health。

第三部分：在 core.step() 加 pre-anchor clamp

这是最关键的一步。

改动目标

在 warmup 和 pre-anchor 阶段，JSD 只观测，不得改变 mode/candidate/frontier。

文件
core.py

在 shock = self.shock_detector.update(...) 之后，加一层硬保护：

逻辑上等价于：

pre_anchor_active = warmup_active or (cold_start_active and not seek_exit_ready)

if pre_anchor_active:
    self.mode = "STABLE"
    self.shock_detector.force_stable()

但还不够，还要继续保护后面的几步：

要同时限制的 3 件事
1. 禁止 force_adapt_reasons 把 pre-anchor 推进 ADAPT

现在 force_adapt_reasons 一旦触发，会直接把 self.mode = "ADAPT"。这在 pre-anchor 阶段也要拦住。

2. pre-anchor 禁止 adapt_use_all_configs_if_small

也就是即使 shock/JSD 高，也不能在这阶段 all_configs() 扩池。当前 core.step() 里这一步就在 candidate generation 之后直接发生。

3. pre-anchor 期间不提交新的 anchors_by_regime

因为这时候系统还在压缩，不是在识别新 frontier。当前代码每一步都会更新 anchors，这里要先冻结。

目的

保住你现在最珍贵的行为：

(2,8) -> ... -> (2,2) 这种压缩链，不能被新接入的 JSD 打断。

第四部分：把 SEEK 拆成两个语义阶段，但不改 SeekAnchorPolicy

你同事说得对，问题不在 warmup 本身，而在 warmup 后的 SEEK 持续太久。

改动目标

保留当前 SeekAnchorPolicy，但让 SEEK_ANCHOR 更早从“压缩 seek”切到“邻域搜”。

文件
core.py

新增一个轻量判定，比如：

compression_done

或 seek_local_band_ready

判定条件用很保守的版本就行，例如满足任意两条：

当前点已经没有下降邻居

当前 quality/latency margin 都过线

最近 2~3 步 switch distance 很小

已经处在 anchor 邻域内

一旦成立：

不再继续按 cold-start probe 节奏全速 seek

降低 cold_start_probe_every 的有效频率

或直接允许更早 seek_exit_ready

目的

不是缩短 warmup_hold_windows，而是让：

压缩完成后，尽快从“持续 seek”切成“近邻稳态搜索”。

你同事说 “WARMUP 只有 4 窗，不慢；真正长的是 SEEK” 这点是对的。当前 SchedulerConfig 的默认值也确实是 warmup_hold_windows=4，而 cold_start_windows/cold_start_max_windows 更长。

第五部分：anchor-ready 后，JSD 才能触发 local rediscovery
改动目标

JSD 不接管 warmup/seek；JSD 只接管 post-anchor 的局部前沿刷新。

文件
core.py

新增一个门：

jsd_trigger_allowed = anchor_ready and not pre_anchor_active

只有满足这个门，JSD 高才允许：

进入现有 SHOCK_LOCAL_REROUTE

打开局部重识别

行为

触发后：

旧 anchor 保留为 baseline

只在 anchor_cfg / x_prev 邻域里试 2~4 个点

不开全局大池

不重跑 cold start

不立刻全局重建 Pareto

目的

把 JSD 的职责限定为：

已经回到合理带后，用 token shift 解释为什么附近最优点变了。

这才是安全接入方式。

第六部分：shift 后只做“局部 frontier refresh”，不做“立刻全局 Pareto 重建”
改动目标

你前面说的很对：全局前沿只提供参考，真正执行上应该是局部探索。

文件
core.py

在 SHOCK_LOCAL_REROUTE 期间：

暂缓写 anchors_by_regime[self.regime_id]

改成只维护 temporary trial anchors

anchor.py

保留现有 trial-promote 机制

只有连续赢够窗口，才 promotion

pareto.py

加一个 delayed commit 入口

允许 “refresh frontier candidate” 和 “commit frontier” 分开

目的

避免刚发生 shift 时样本太少，就把新 front 误写死。

第七部分：safe 只做局部临时放松
改动目标

你说“临时开放临近最优点”这个方向是对的，但一定要局部。

文件
safety.py 或 core.py

加一层 rediscovery-local relaxation：

只对 anchor_cfg / x_prev 邻域生效

只放软约束

只放 1~2 个窗口

结束后自动恢复

不该做的

不能全局放开 safety

不能把所有 untrusted 点都放进池子

不能把 relaxed 结果直接写成长久 anchor/frontier

最终实施顺序

我建议就按这个顺序改，风险最低。

第 1 步

types.py + 上游链路：唯一 drift = JSD

第 2 步

regime.py：ShockDetector 改 JSD-only

第 3 步

core.py：pre-anchor clamp

warmup / pre-anchor 禁 ADAPT

禁 all_configs 扩池

禁 anchor commit

第 4 步

core.py：compression_done / early seek-exit

保住 (2,8)->(2,2)

到合理带后尽快转近邻搜索

第 5 步

core.py + anchor.py + pareto.py：anchor-ready 后的 local rediscovery + delayed commit

第 6 步

safe 层：局部临时放宽