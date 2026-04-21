#!/bin/bash

# Memory Budget Allocation and Hardware Quality: How does giving more/less memory/compute to each Ray actor shift the trade-offs?
CUDA_VISIBLE_DEVICES=0,1,2 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --num_workers 3 --wandb_run_name tream_gpus-3_lvm-data

# TODO: figure out distributed training
# CUDA_VISIBLE_DEVICES=5,6,7,8 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --num_workers 4 --wandb_run_name tream_gpus-4

