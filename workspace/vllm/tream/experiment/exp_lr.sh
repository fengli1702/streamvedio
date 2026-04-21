#!/bin/bash

# Learning Rate Sweep: How does learning rate affect training?
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train -lr 1e-5 --wandb_run_name tream_lr-1e-5_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train -lr 5e-5 --wandb_run_name tream_lr-5e-5_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train -lr 1e-4 --wandb_run_name tream_lr-1e-4_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train -lr 5e-4 --wandb_run_name tream_lr-5e-4_lvm-data
uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train -lr 1e-3 --wandb_run_name tream_lr-1e-3_lvm-data

uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train -lr 1e-5 --wandb_run_name tream_lr-1e-5_room1D
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train -lr 5e-5 --wandb_run_name tream_lr-5e-5_room1D
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train -lr 1e-4 --wandb_run_name tream_lr-1e-4_room1D
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train -lr 5e-4 --wandb_run_name tream_lr-5e-4_room1D
uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train -lr 1e-3 --wandb_run_name tream_lr-1e-3_room1D
