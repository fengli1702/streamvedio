#!/bin/bash

# Prediction Horizon: How far can we look ahead accurately? (context length 4)
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 2 --wandb_run_name tream_il-2_lvm-data &
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 4 --wandb_run_name tream_il-4_lvm-data &
CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 6 --wandb_run_name tream_il-6_lvm-data &
wait

# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 8 --wandb_run_name tream_il-8
# uv run tream_2.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 16 --wandb_run_name tream_il-16
# uv run tream_2.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 24 --wandb_run_name tream_il-24
# uv run tream_2.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 32 --wandb_run_name tream_il-32
