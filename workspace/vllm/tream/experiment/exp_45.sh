#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 10 --wandb_run_name tream_wsf-10_lvm-data
# CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 5 --wandb_run_name tream_wsf-5_room1D
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 4 --wandb_run_name tream_il-4_lvm-data
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 6 --wandb_run_name tream_il-6_lvm-data
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D                                               --wandb_run_name neg_baseline_room1D
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 12 --wandb_run_name tream_cl-12_room1D

# CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 10 --wandb_run_name tream_wsf-10_lvm-data
# CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 5 --wandb_run_name tream_wsf-5_room1D
# CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 6 --wandb_run_name tream_il-6_lvm-data
# CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 4 --wandb_run_name tream_il-4_lvm-data
# CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 2 --wandb_run_name tream_il-2_lvm-data
# CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 6 --wandb_run_name tream_il-6_lvm-data