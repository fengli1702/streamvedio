#!/bin/bash

# Loss-Curve Dynamics: How fast do we learn vs. tokens seen?
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path "../streaming-lvm/data/room_1D"  -gc -train --wandb_run_name tream_room1D --max_frames 50000
CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path "../streaming-lvm/data/room_2D"  -gc -train --wandb_run_name tream_room2D --max_frames 50000 