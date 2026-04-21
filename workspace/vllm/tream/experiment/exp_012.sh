#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --num_workers 3 --wandb_run_name tream_gpus-3_lvm-data
CUDA_VISIBLE_DEVICES=0,1,2 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --num_workers 3 --wandb_run_name tream_room1D --max_frames 50000
