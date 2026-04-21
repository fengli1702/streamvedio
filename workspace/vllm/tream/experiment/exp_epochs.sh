#!/bin/bash

# Epochs Sweep: How does the number of epochs affect training?
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --epochs 2 --wandb_run_name tream_e-2_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --epochs 4 --wandb_run_name tream_e-4_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --epochs 5 --wandb_run_name tream_e-5_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --epochs 2 --wandb_run_name tream_e-2_room1D
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --epochs 4 --wandb_run_name tream_e-4_room1D
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --epochs 5 --wandb_run_name tream_e-5_room1D
