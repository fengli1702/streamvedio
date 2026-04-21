# Scheduler Effect: Mathematical Validation (ABC24)

Generated at: `2026-03-18 (UTC)`

## 1) Data Used

- Dynamic manifest (24 runs):  
  `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/abc24x3_trigpu_shift_taskpool_20260316_052059_a06rerun_20260317_062025.tsv`
- Static manifest (24 runs):  
  `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/abc24x3_trigpu_shift_taskpool_static_balanced_20260316_235206.tsv`
- Drift/switch chain (oracle under `acc>=tau`):  
  `/m-coriander/coriander/daifeng/testvllm/vllm/tream/analysis_out_doh/continuity_analysis_token_abc24_static_20260316_235206_w50/chain_fix_ib32_tb16_tau0p6.csv`

Paired runs used in test: `24` (dynamic vs static by same case id).

## 2) Mathematical Criteria

### Criterion A: Dynamic should improve latency without hurting accuracy

For each paired case `i`:

- `Δlat_i = lat_dyn_i - lat_static_i` (smaller is better)
- `Δacc_i = acc_dyn_i - acc_static_i` (larger is better)

We report:
- mean difference + bootstrap 95% CI
- paired sign test on latency (`H0: P(Δlat<0)=0.5`)

### Criterion B: High-workload points should be compressed

For runs with initial workload `W0 = ctx0 * inf0 >= 16`, check:
- `W1 - W0` at the end of scheduling
- count of reduced/equal/increased

### Criterion C: Switching should be statistically associated with token drift

Using chain windows:
- point-biserial/pearson correlation `corr(drift, switch_flag)`
- `P(switch | drift >= q75)` vs `P(switch | drift < q75)`
- `P(switch | drift >= q90)` vs `P(switch | drift < q90)`
- risk ratio `RR = P_high / P_low`

## 3) Results

### A. Dynamic vs Static (paired)

- `mean(Δlat) = -0.1849 s`
- `95% CI(Δlat) = [-0.2478, -0.1223]`
- Sign test on latency: `p = 0.00661`
- Improved latency pairs: `19 / 24`

- `mean(Δacc) = +0.0248`
- `95% CI(Δacc) = [+0.0130, +0.0364]`

Interpretation: dynamic is faster on average, and accuracy is also higher on average.

### B. High-workload compression (`W0 >= 16`)

- Number of high-workload runs: `10`
- `mean(W1 - W0) = -18.3`
- Reduced: `10`, Equal: `0`, Increased: `0`

Representative cases:
- `B03: (4,4)->(2,3), 16->6`
- `B08: (5,5)->(2,2), 25->4`
- `C03: (2,8)->(2,2), 16->4`
- `C04: (8,2)->(2,2), 16->4`

`C03` early trajectory snippet confirms expected compression path:
- step 0..11: `(2,8),(2,8),(2,8),(2,8),(2,7),(2,6),(2,5),(2,4),(2,3),(2,2),(2,4),(2,3)`

### C. Drift-switch association (oracle chain)

- `corr(drift, switch_flag) = 0.3303`
- Global switch rate: `0.3165`

At `q75` threshold:
- `q75 = 0.2415`
- `P(switch | drift>=q75) = 0.5500`
- `P(switch | drift<q75) = 0.2373`
- `RR = 2.3179` (high-drift windows are ~2.32x more likely to switch)

At `q90` threshold:
- `q90 = 0.3695`
- `P(switch | drift>=q90) = 0.7500`
- `P(switch | drift<q90) = 0.2676`
- `RR = 2.8026` (high-drift windows are ~2.80x more likely to switch)

Threshold used in figure (`P(switch|drift>=th) >= 0.60`, support>=5):
- `th = 0.2688`
- `P = 0.6111`
- `support = 18 windows`

## 4) Conclusion

Under this ABC24 dataset, the scheduler shows mathematically consistent behavior with the design goal:

1. **Efficiency gain**: significant latency reduction vs static (`p=0.00661`, CI fully negative).
2. **No quality sacrifice**: accuracy difference is positive with CI fully above zero.
3. **Workload compression**: all high-start workloads were compressed (`10/10` reduced).
4. **Drift-linked switching**: switch probability increases strongly in high-drift regimes (`RR 2.32~2.80`).

So the observed effect is not only visual; it is statistically supported by paired and conditional analyses.

## 5) This-Round Change Log (2026-03-18)

### 5.1 Scheduler core: minimal-risk alignment changes

Code:
- `/m-coriander/coriander/daifeng/testvllm/vllm/tream/scheduler_Shift/core.py`

Applied changes:
- Added `shock_use_jsd_only` switch in `SchedulerConfig`.
- Added local neighborhood safety-relax knobs:
  - `local_relax_enable`
  - `local_relax_windows`
  - `local_relax_radius`
  - `local_relax_drift_threshold`
- Added `pre_anchor_active` clamp in `step()`:
  - During `warmup` and `cold_start && !seek_exit_ready`, force `mode=STABLE`.
  - Call `shock_detector.force_stable()` to prevent early ADAPT contamination.
  - Clear force-adapt reasons in pre-anchor stage.
- Drift signal unification path:
  - Added `_drift_signal(window)`.
  - If `shock_use_jsd_only=true`, use `window.jsd_mean`.
  - Else keep old `window.drift_value()` path.
  - Routed regime quantization and shock drift input through `_drift_signal`.
- Added temporary local safe-set extension under post-anchor ADAPT drift:
  - Triggered only when `local_relax` condition is met.
  - Extends safety set only in small `(ctx,inf)` neighborhood around anchor/current point.
  - Keeps `ib/tb` unchanged.
- Added decision meta observability:
  - `pre_anchor_active`
  - `local_relax_active`
  - `local_relax_remaining`
  - `local_relax_radius`
  - `local_relax_drift_threshold`

Design intent:
- Keep existing good compression behavior (`(2,8)->...->(2,2)` class behavior) untouched.
- Let JSD drive shift detection only after pre-anchor phase.
- Allow controlled local rediscovery instead of global unsafe expansion.

### 5.2 Actor config wiring: env -> SchedulerConfig

Code:
- `/m-coriander/coriander/daifeng/testvllm/vllm/tream/actors/scheduler_actor.py`

Applied changes:
- Added env parsing and config forwarding for:
  - `TREAM_SHIFT_SHOCK_USE_JSD_ONLY`
  - `TREAM_SHIFT_LOCAL_RELAX_ENABLE`
  - `TREAM_SHIFT_LOCAL_RELAX_WINDOWS`
  - `TREAM_SHIFT_LOCAL_RELAX_RADIUS`
  - `TREAM_SHIFT_LOCAL_RELAX_DRIFT_THRESHOLD`
- In `JSD-only` mode, set shock weights to drift-only for mode control:
  - keep `drift` weight
  - set `accept/reverify/verify_ratio/waste` weights to `0` for mode switching
  - these non-drift metrics are still logged (not removed from telemetry).
- Added startup log printing of the new knobs for runtime verification.

Design intent:
- Fast experiment toggle without rewriting policy bodies.
- Preserve observability of original health signals while simplifying shift trigger semantics.

### 5.3 Launch chain for focused ABC subset (6 points)

Code:
- `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/launch_abc6_trigpu_shift_taskpool_jsdonly.sh`

Coverage set:
- `(1,1)`, `(4,4)`, `(1,8)`, `(8,1)`, `(2,8)`, `(8,2)`

Runtime knobs in launcher:
- `TREAM_SHIFT_SHOCK_USE_JSD_ONLY=1`
- `TREAM_SHIFT_LOCAL_RELAX_ENABLE=1`
- `TREAM_SHIFT_LOCAL_RELAX_WINDOWS=4`
- `TREAM_SHIFT_LOCAL_RELAX_RADIUS=1`
- `TREAM_SHIFT_LOCAL_RELAX_DRIFT_THRESHOLD=0.30`

Important chain fix:
- Added `PYTHONPATH="${TREAM_ROOT}/..:${PYTHONPATH:-}"` when launching `tream.py`.
- Root cause of first failed attempt was `ModuleNotFoundError: vllm` in container runtime path.

## 6) Knowledge Summary From This Iteration

1. `seek_anchor` works as a compression mechanism because it is still strongly constrained by safe neighborhood movement and anti-rebound guards; this is why bad starts can still be pushed down.
2. The main risk is not only phase routing, but early `mode=ADAPT` side-effects before phase policy execution:
   - candidate pool expansion
   - anchor/frontier update contamination
3. Therefore, pre-anchor clamping is critical:
   - it protects warmup/cold-start compression path from JSD-triggered ADAPT noise.
4. `JSD-only` is valid as a clean shift signal for this experiment, but should be gated by stage:
   - observe in pre-anchor
   - act in post-anchor.
5. Local relax should be temporary and local:
   - radius-limited
   - window-limited
   - trigger-thresholded
   - no global all-config reopening.
6. Operationally, launch-chain correctness matters as much as scheduler logic:
   - if runtime import path is wrong, all policy conclusions are invalid because runs never reached scheduler steps.

## 7) Current Repro Artifacts (for this round)

- Code:
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/scheduler_Shift/core.py`
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/actors/scheduler_actor.py`
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/launch_abc6_trigpu_shift_taskpool_jsdonly.sh`
- Latest 6-point run id:
  - `abc6x3_trigpu_shift_taskpool_jsdonly_20260317_223847`
- Logs:
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/abc6x3_trigpu_shift_taskpool_jsdonly_20260317_223847.status.log`
  - `/m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/abc6x3_trigpu_shift_taskpool_jsdonly_20260317_223847.tsv`
