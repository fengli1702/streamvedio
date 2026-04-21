# Task Brief: AB48 Dynamic+Spec (LB1)

## Run ID
ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260407_133732

## Goal
Re-run 48-point AB grid on 3 GPUs with latest scheduler code where:
- Global ctx/inf lower bound is open to 1.
- Cold-start default target remains 2.

## Launch
- Script: `tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh`
- Mode: `RUN_MODE=dynamic`, `RUN_SPEC=1`
- GPUs: `GPU_A=0`, `GPU_B=1`, `GPU_C=2`
- Status log: `tream/inference_logs/ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260407_133732.status.log`
- Manifest: `tream/inference_logs/ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260407_133732.tsv`

## Scheduler semantic marker
This run is the "LB1" round (global lower bound=1, cold-start target=2).

## Core code basis
- `tream/actors/scheduler_actor.py`: bounds default `(1,8)`; cold target env defaults `2`.
- `tream/scheduler_Shift/core.py`: cold-start target defaults `2`.
