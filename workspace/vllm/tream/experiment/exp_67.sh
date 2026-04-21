#!/bin/bash

# 1. Real-time Online-vs-Offline: How much latency & throughput do we pay for simultaneous training + inference?
# uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --wandb_run_name tream_base
# uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset            --wandb_run_name no_train
# uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset            --wandb_run_name full_train

CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --wandb_run_name tream_base_lvm-data
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --wandb_run_name tream_base_room1D
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset            --wandb_run_name pos_baseline_lvm-data --model_name "../streaming-lvm/saved_models/tream_base_lvm-data"
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D                          --wandb_run_name pos_baseline_room1D   --model_name "../streaming-lvm/saved_models/tream_base_room1D"
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 8 --wandb_run_name tream_cl-8_room1D

# CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --inference_length 8 --wandb_run_name tream_il-8_room1D

# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --wandb_run_name tream_base_lvm-data
# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --wandb_run_name tream_base_room1D
# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D                          --wandb_run_name neg_baseline_room1D
# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D                          --wandb_run_name pos_baseline_room1D --model_name "../streaming-lvm/saved_models/tream_base_room1D"
# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 8 --wandb_run_name tream_il-8_lvm-data
# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --inference_length 8 --wandb_run_name tream_il-8_room1D

