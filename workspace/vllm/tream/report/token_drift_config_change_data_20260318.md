# Token Drift & Config Change Data Map (2026-03-18)

## 1) 你要的“最优点曲线”对应 CSV 在哪里

目标图：
- `testvllm/vllm/tream/analysis_out_doh/continuity_analysis_token_abc24_static_20260316_235206_w50/best_latency_curve_fix_ib32_tb16_tau0p6_window50_static_4000.png`

对应的核心 CSV（同目录）：
- `best_fix_ib32_tb16_tau0p6.csv`
  - 每个窗口选中的最优 `(ctx,inf)` 及其 `lat_mean/acc_mean/token_drift_mean`
- `chain_fix_ib32_tb16_tau0p6.csv`
  - 专门用于分析 config 切换链路（`selected_key`, `prev_selected_key`, `switch_flag`）
- `drift_fix_ib32_tb16.csv`
  - 按 frame 聚合后的 token drift 时间序列
- `acceptance_fix_ib32_tb16.csv`
  - 按 frame 聚合后的 draft acceptance 时间序列

同一批（非 w50）版本在：
- `testvllm/vllm/tream/analysis_out_doh/continuity_analysis_token_abc24_static_20260316_235206/`

这里还有你之前提到的“最优曲线（ctx,inf 折线）”CSV：
- `best_ctx_inf_curve_fix_ib32_tb16_tau0p6_static_4000.csv`

## 2) ABC 实验结果 CSV（延迟/准确率）在哪里

- 动态 24 条：
  - `testvllm/vllm/tream/experiment/plots/abc24_dynamic_latency_accuracy_20260316_052059.csv`
- 动静态 24 vs 24 对比：
  - `testvllm/vllm/tream/experiment/plots/abc24_latency_accuracy_compare_20260317_062025_vs_20260316_235206.csv`
  - `testvllm/vllm/tream/experiment/plots/abc24_latency_accuracy_compare_20260316_052059_vs_20260316_235206.csv`

## 3) 为什么“现在和原来不一样”最常见

从现有落盘数据看，主要不是一条链路坏了，而是统计口径不同：

1. 窗口宽度不同（`window_frames=100` vs `window_frames=50`）
   - 非 w50（100）：`rows=40`, `switches=9`, `switch_rate=0.225`
   - w50（50）：`rows=80`, `switches=25`, `switch_rate=0.3125`
   - 结论：窗口变窄后，切换检测更敏感，曲线会更“抖”。

2. 输出 schema 演进
   - `continuity_analysis_token_spec4000` 里 `best_fix_*.csv` 没有 `switch_flag`（旧口径）
   - `*_dense2x` 与 `abc24_static_*` 有 `chain_fix_*.csv` 与 `switch_flag`

3. 数据集子集不同
   - `spec4000` 系列与 `abc24_static_20260316_235206` 系列不是完全同一 run 集合，最优点轨迹会变。

## 4) 你要算的指标，直接用这几个字段

从 `chain_fix_ib32_tb16_tau0p6.csv`：
- `token_drift_mean`
- `selected_key`
- `prev_selected_key`
- `switch_flag`
- `lat_mean`
- `acc_mean`
- `spec_reject_ratio`

推荐定义：
- config 变化量（0/1）：`switch_flag`
- 同步分析：`corr(token_drift_mean, switch_flag)`
- 条件概率：`P(switch=1 | token_drift_mean > threshold)`

## 5) 一条命令快速复算（w50 版本）

```bash
python - <<'PY'
import pandas as pd
p='testvllm/vllm/tream/analysis_out_doh/continuity_analysis_token_abc24_static_20260316_235206_w50/chain_fix_ib32_tb16_tau0p6.csv'
df=pd.read_csv(p)
print('rows=',len(df),'switches=',int(df.switch_flag.sum()),'switch_rate=',df.switch_flag.mean())
print('corr(drift,switch)=',df['token_drift_mean'].corr(df['switch_flag']))
for th in [0.10,0.15,0.20,0.25,0.30]:
    m=df['token_drift_mean']>th
    if m.sum()==0:
        continue
    print(f'th>{th:.2f}: P(switch|drift>th)=',df.loc[m,'switch_flag'].mean(),'n=',int(m.sum()))
PY
```

## 6) 如果你要对齐“原来口径”

建议固定以下参数重导出：
- `window_frames=100`
- `tau=0.6`
- 同一批 logs（同一 run list）
- 同一目录版本（避免 mixed old/new schema）

脚本：
- `testvllm/vllm/tream/tools/plot_token_drift_spec_static.py`

