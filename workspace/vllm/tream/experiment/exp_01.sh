#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/streaming-lvm-dataset -gc -train --weight_sharing_freq 15 --wandb_run_name tream_wsf-15_lvm-data
# CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ../streaming-lvm/data/room_1D -gc -train --weight_sharing_freq 10 --wandb_run_name tream_wsf-10_room1D
CUDA_VISIBLE_DEVICES=1,2,3,4,5 uv run tream.py --input_frames_path ./data/streaming-lvm-dataset/DOH -gc -train --context_length 1 --wandb_run_name tream_cl-1_lvm-data   --model_name saved_models/lvm-llama2-7b
CUDA_VISIBLE_DEVICES=0 uv run tream.py --input_frames_path ./data/streaming-lvm-dataset/DOH -gc -train --context_length 2 --wandb_run_name tream_cl-2_lvm-data   --model_name saved_models/lvm-llama2-7b
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ./data/streaming-lvm-dataset/DOH -gc -train --context_length 8 --wandb_run_name tream_cl-8_lvm-data   --model_name saved_models/lvm-llama2-7b
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ./data/streaming-lvm-dataset/DOH -gc -train --context_length 12 --wandb_run_name tream_cl-12_lvm-data   --model_name saved_models/lvm-llama2-7b
CUDA_VISIBLE_DEVICES=0,1 uv run tream.py --input_frames_path ./data/streaming-lvm-dataset/DOH -gc -train --inference_length 2 --wandb_run_name tream_il-2_lvm-data   --model_name saved_models/lvm-llama2-7b


CUDA_VISIBLE_DEVICES=2 uv run tream.py  --input_frames_path ./data/streaming-lvm-dataset/DOH  -gc -train --context_length 4  --model_name saved_models/lvm-llama2-7b  --max_frames 1000  --num_workers=1  --inference_length 12
CUDA_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 uv run tream.py  --input_frames_path ./data/streaming-lvm-dataset/DOH  -gc  --context_length 4  --model_name saved_models/lvm-llama2-7b  --num_workers=1  --inference_length 12 -train
CUDA_VISIBLE_DEVICES=2 nsys profile  -t cuda,nvtx,osrt  -o profile_150_c4_infer16   --force-overwrite=true  -w true  uv run tream.py  --input_frames_path ./data/streaming-lvm-dataset/DOH  -gc  --context_length 4  --model_name saved_models/lvm-llama2-7b  --max_frames 150 --num_workers=1  --inference_length 16 -train