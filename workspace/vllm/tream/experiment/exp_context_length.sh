#!/bin/bash

# Context-Length Scaling: What happens when we increase window size?
(
    CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --context_length 1 --wandb_run_name tream_cl-1_lvm-data &
    CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 1 --wandb_run_name tream_cl-1_room1D &
    (
        CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset/DOH      -gc -train --wandb_run_name tream_doh &&
        CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset/Ego4d    -gc -train --wandb_run_name tream_ego4d &&
        CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset/motsynth -gc -train --wandb_run_name tream_motsynth &&
        CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset/UCF101   -gc -train --wandb_run_name tream_ucf101 &&
        CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset/visdrone -gc -train --wandb_run_name tream_visdrone
    ) &
    wait
)

(
    CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --context_length 2 --wandb_run_name tream_cl-2_lvm-data &
    CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 2 --wandb_run_name tream_cl-2_room1D &
    CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 2 --wandb_run_name tream_il-2_lvm-data &
    wait
)

(
    CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --context_length 8 --wandb_run_name tream_cl-8_lvm-data &
    CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 8 --wandb_run_name tream_cl-8_room1D &
    CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 4 --wandb_run_name tream_il-4_lvm-data &
    wait
)

(
    CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --context_length 12 --wandb_run_name tream_cl-12_lvm-data &
    CUDA_VISIBLE_DEVICES=2,3 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 12 --wandb_run_name tream_cl-12_room1D &
    CUDA_VISIBLE_DEVICES=4,5 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --inference_length 6 --wandb_run_name tream_il-6_lvm-data &
    wait
)

# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --context_length 15 --wandb_run_name tream_cl-15_lvm-data
# CUDA_VISIBLE_DEVICES=6,7 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D               -gc -train --context_length 15 --wandb_run_name tream_cl-15_room1D

# uv run tream.py --input_frames_path ./data/lvm_dataset -gc -train --context_length 24 --wandb_run_name exp4_24
# uv run tream.py --input_frames_path ./data/lvm_dataset -gc -train --context_length 32 --wandb_run_name exp4_32
# uv run tream.py --input_frames_path ./data/lvm_dataset -gc -train --context_length 64 --wandb_run_name exp4_64