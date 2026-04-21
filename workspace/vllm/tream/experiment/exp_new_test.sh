#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 15 --wandb_run_name tream_wsf-15_lvm-data
# CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 10 --wandb_run_name tream_wsf-10_room1D
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py   --input_frames_path ./data/streaming-lvm-dataset/DOH   -gc   --context_length 4   --model_name saved_models/lvm-llama2-7b  --num_workers=1   --inference_length 1    -train  --wandb_run_name tream_cl-1_lvm-data-diffinfer-DOH
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py   --input_frames_path ./data/streaming-lvm-dataset/DOH   -gc   --context_length 4   --model_name saved_models/lvm-llama2-7b  --num_workers=1   --inference_length 2    -train  --wandb_run_name tream_cl-2_lvm-data-diffinfer-DOH
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py   --input_frames_path ./data/streaming-lvm-dataset/DOH   -gc   --context_length 4   --model_name saved_models/lvm-llama2-7b  --num_workers=1   --inference_length 4    -train  --wandb_run_name tream_cl-4_lvm-data-diffinfer-DOH
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py   --input_frames_path ./data/streaming-lvm-dataset/DOH   -gc   --context_length 4   --model_name saved_models/lvm-llama2-7b  --num_workers=1   --inference_length 8    -train  --wandb_run_name tream_cl-8_lvm-data-diffinfer-DOH
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py   --input_frames_path ./data/streaming-lvm-dataset/DOH   -gc   --context_length 4   --model_name saved_models/lvm-llama2-7b  --num_workers=1   --inference_length 12   -train   --wandb_run_name tream_cl-12_lvm-data-diffinf-DOH

