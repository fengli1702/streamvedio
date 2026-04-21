#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 uv run tream.py --input_frames_path "../streaming-lvm/data/room_2D"  -gc -train --wandb_run_name tream_room2D --max_frames 50000 

# TODO: figure out distributed training
# CUDA_VISIBLE_DEVICES=5,6,7,8 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --num_workers 4 --wandb_run_name tream_gpus-4
