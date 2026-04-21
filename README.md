# Open-box Full Package (No Docker)

This package is intended to be copied to a remote machine and run directly.

## Included
- `workspace/vllm/`: runnable code snapshot (hacked vLLM + tream + SpecForge code)
- `workspace/vllm/tream/lvm-llama2-7b/`: base model
- `workspace/vllm/tream/data/streaming-lvm-dataset/`: datasets
- `workspace/vllm/SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475/`: draft checkpoint
- `tools/no_docker/`: environment setup files
- `tools/launch_ab48_8gpu.sh`: 8-GPU launcher
- `tools/run_ab48_spec_static_8gpu.sh`: one-click single experiment
- `tools/run_ab48_ablation_4modes_8gpu.sh`: one-click 4-mode ablation

## Quick start

```bash
tar -xzf openbox_full_*.tar.gz
cd openbox_full_*

# one-click: AB48 static spec, 8-GPU
bash tools/run_ab48_spec_static_8gpu.sh
```

## One-click full ablation (4 modes)

```bash
bash tools/run_ab48_ablation_4modes_8gpu.sh
```

Defaults:
- environment: `conda`, env name `tream`
- dataset: `Ego4d`
- GPUs: `0,1,2,3,4,5,6,7`
- concurrency: `8`
