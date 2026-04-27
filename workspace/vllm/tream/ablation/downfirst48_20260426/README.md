# DOH AB48 Dyn vs JSD Down-First Run

Date: 2026-04-26 / 2026-04-27

This directory records the validated parameter family and code path for the DOH AB48 comparison where JSD down-first improves latency over the matching Dyn(NoJSD) baseline.

## Runs

- Dyn(NoJSD): `ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td1_pre2-2_jstage1-1_cw15_df1lg0p03qd0p05cont0_20260426_083320`
- Dyn+JSD down-first: `ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_pre2-2_jstage1-1_cw15_df1lg0p03qd0p05cont0_20260426_174214`
- A05 JSD rerun: `rerun_a05_jsd_downfirst_20260427.sh`

A05 from the first JSD full run was a false success: the driver returned rc=0 after `TrainingActor.__init__` failed with `EADDRINUSE`. It is rerun separately with one lane on GPU 7 and unique Ray ports.

## Code Changes

The JSD local-probe path is changed to be latency-oriented and down-first:

- `tream/scheduler_Shift/core.py`
  - Adds `jsd_probe_down_first`, `jsd_probe_down_latency_gain_eps`, and `jsd_probe_down_quality_drop_max` config fields.
  - Starts JSD probe at the downward neighbor when `jsd_probe_down_first=1`.
  - Commits the downward move when latency improves by at least `down_latency_gain_eps` and quality drop is no more than `down_quality_drop_max`.
  - Falls back to base if downward movement is too lossy; probes upward only when down does not produce enough latency gain.
- `tream/actors/scheduler_actor.py`
  - Wires environment variables into `SchedulerConfig`.
  - Logs the active down-first thresholds in scheduler startup output.

## Runtime Parameters

Common parameters for both Dyn(NoJSD) and Dyn+JSD:

```bash
RUN_MODE=dynamic
RUN_SPEC=1
DATASET_SUBDIR=DOH
MAX_FRAMES=4000
IB=32
TB=16
GPU_MEM_UTIL=0.9
RAY_NUM_CPUS=16
WORKER_LANES=3
SHIFT_COLD_START_WINDOWS=15
SHIFT_COLD_START_MAX_WINDOWS=15
SHIFT_COLD_PROBE_EVERY=1
SHIFT_COLD_PROBE_EPS_FAST=2
SHIFT_COLD_PROBE_EPS_SLOW=0
SHIFT_COLD_AVOID_TWO_CYCLE=1
SHIFT_COLD_AXIS_ROTATION=1
SHIFT_COLD_PATIENCE_DIRECTIONS=8
SHIFT_DEFAULT_QUALITY_MIN=0.2
SHIFT_COLD_RELAX_SAFETY=1
SHIFT_COLD_POST_PAUSE_WINDOWS=1
SHIFT_MIN_COUNT_FOR_TRUST=1
SHIFT_ADAPT_PROBE_WINDOWS=0
SHIFT_COLD_WHITELIST_PROBE=1
SHIFT_COLD_WHITELIST_BUDGET=12
SHIFT_COLD_WHITELIST_LAT_SLACK=0.15
SHIFT_COLD_WHITELIST_Q_SLACK=0.60
SHIFT_COLD_WHITELIST_MAX_SWITCH=2
SHIFT_SHOCK_USE_JSD_ONLY=1
SHIFT_LOCAL_RELAX_ENABLE=1
SHIFT_LOCAL_RELAX_WINDOWS=3
SHIFT_LOCAL_RELAX_RADIUS=1
SHIFT_LOCAL_RELAX_Q_SLACK=0.05
SHIFT_LOCAL_RELAX_LAT_SLACK=0.08
SHIFT_LOCAL_RELAX_DRIFT_THRESHOLD=0.30
SHIFT_JSD_PROBE_ENABLE=1
SHIFT_JSD_PROBE_DOWN_FIRST=1
SHIFT_JSD_PROBE_DOWN_LAT_GAIN_EPS=0.03
SHIFT_JSD_PROBE_DOWN_Q_DROP_MAX=0.05
SHIFT_JSD_PROBE_CONTINUE_STEPS=0
SHIFT_PRE_JSD_CTX_MIN=2
SHIFT_PRE_JSD_INF_MIN=2
SHIFT_JSD_STAGE_CTX_MIN=1
SHIFT_JSD_STAGE_INF_MIN=1
SHIFT_COLD_DIRECTIONAL_ENABLE=0
SHIFT_COLD_PREFER_LARGE_DESCEND=0
SHIFT_COLD_PREFER_SMALL_ASCEND=0
SHIFT_COLD_SINGLE_AXIS_LOCK=0
SHIFT_COLD_EARLY_EXIT_ENABLE=0
```

Only difference between the two dynamic groups:

```bash
# Dyn(NoJSD)
SHIFT_SHOCK_DISABLE_DRIFT=1

# Dyn+JSD down-first
SHIFT_SHOCK_DISABLE_DRIFT=0
```

## Launch Commands

Full 48+48 comparison:

```bash
cd /workspace/vllm/tream
bash inference_logs/launch_downfirst48_compare_20260426.sh
```

A05 JSD single-case rerun:

```bash
cd /workspace/vllm/tream
bash inference_logs/rerun_a05_jsd_downfirst_20260427.sh
```

## Valid-47 Result Before A05 Rerun

A05 JSD is excluded from this temporary summary. The CSV is stored as `downfirst48_valid47_compare.csv`.

| group | mean latency | mean accuracy |
| --- | ---: | ---: |
| Spec-only baseline | 0.719649 | 0.428950 |
| Dyn(NoJSD) | 0.434537 | 0.500133 |
| Dyn+JSD down-first | 0.395627 | 0.497813 |

Speedups:

- Dyn(NoJSD) vs spec: `1.6561x`
- Dyn+JSD vs spec: `1.8190x`
- Dyn+JSD vs Dyn(NoJSD): `1.0984x`

Win counts on valid 47 cases:

- Dyn latency better than spec: `39/47`
- JSD latency better than Dyn: `36/47`
- JSD latency better than spec: `41/47`
- Dyn accuracy at least spec: `35/47`
- JSD accuracy at least Dyn: `19/47`

Group-level behavior:

- Small workloads: JSD is worse than Dyn due to exploration overhead/noise (`0/6` latency wins).
- Medium workloads: JSD is usually better than Dyn (`8/9` latency wins).
- Large workloads: JSD is usually better than Dyn (`28/32` latency wins).
