#!/bin/bash

# 2. Update-Frequency Sweep: How often (in batches) should weights be published from GPU 2?
# CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 1 --wandb_run_name tream_wsf-1_lvm-data
(
    CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 5 --wandb_run_name tream_wsf-5_lvm-data &
    CUDA_VISIBLE_DEVICES=5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 10 --wandb_run_name tream_wsf-10_lvm-data &
    CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 15 --wandb_run_name tream_wsf-15_lvm-data &
    wait
)

(
    CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 5 --wandb_run_name tream_wsf-5_room1D &
    CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 10 --wandb_run_name tream_wsf-10_room1D &
    CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 15 --wandb_run_name tream_wsf-15_room1D &
    wait
)
