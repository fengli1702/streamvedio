#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 5 --wandb_run_name tream_wsf-5_lvm-data
# CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 15 --wandb_run_name tream_wsf-15_room1D
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 1   --wandb_run_name tream_cl-1_room1D
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 2   --wandb_run_name tream_cl-2_room1D
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 2 --wandb_run_name tream_il-2_lvm-data
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 8 --wandb_run_name tream_il-8_lvm-data
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset                                 --wandb_run_name neg_baseline_lvm-data
