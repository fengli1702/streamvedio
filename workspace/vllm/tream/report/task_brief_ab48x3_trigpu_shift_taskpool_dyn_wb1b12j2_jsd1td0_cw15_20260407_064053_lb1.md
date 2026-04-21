# Task Brief: AB48 Dynamic+Spec (LB1)

## Run ID
ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260407_064053

## Goal
Re-run 48-point AB grid on 3 GPUs with latest scheduler code:
- Global lower bound for ctx/inf is 1.
- Cold-start default target remains 2.

## Launch
- Script: "/workspace/vllm/tream/inference_logs/launch_ab48_trigpu_shift_taskpool.sh"
- Container: "vllm-tream"
- Mode: RUN_MODE=dynamic, RUN_SPEC=1
- GPUs: 0,1,2

## Logs
- Status: /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260407_064053.status.log
- Manifest: /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/ab48x3_trigpu_shift_taskpool_dyn_wb1b12j2_jsd1td0_cw15_20260407_064053.tsv
- Launcher: /m-coriander/coriander/daifeng/testvllm/vllm/tream/inference_logs/ab48_lb1_3gpu_20260407_134053.launcher.log

## Semantic marker
This is the LB1 round.
