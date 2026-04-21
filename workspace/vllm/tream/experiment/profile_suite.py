#!/usr/bin/env python
"""Convenience runner to sweep mini_spec_benchmark across batch sizes."""

from __future__ import annotations

import argparse
import subprocess
import time
import sys
from pathlib import Path
from typing import List


def build_common_args(args: argparse.Namespace, dataset_root: Path) -> List[str]:
    cmd: List[str] = [
        "--dataset-root", str(dataset_root),
        "--teacher-model", str(args.teacher_model),
        "--context-length", str(args.context_length),
        "--inference-length", str(args.inference_length),
        "--num-prompts", str(args.num_prompts),
        "--max-frames", str(args.max_frames),
        "--vqgan-batch", str(args.vqgan_batch),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-mem-util", str(args.gpu_mem_util),
        "--num-spec-tokens", str(args.num_spec_tokens),
        "--dtype", args.dtype,
    ]

    if args.draft_model is not None:
        cmd.extend(["--draft-model", str(args.draft_model)])
    if args.disable_lora:
        cmd.append("--disable-lora")
    if args.log_file is not None:
        cmd.extend(["--log-file", str(args.log_file)])
    if args.output_dir is not None:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.max_concurrent_prompts is not None:
        cmd.extend(["--max-concurrent-prompts", str(args.max_concurrent_prompts)])
    if args.streaming:
        cmd.append("--streaming")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep mini_spec_benchmark across batch sizes")
    parser.add_argument("--dataset-root", type=Path, required=True, action='append',
                        help="Dataset root directory. Can be specified multiple times for multiple datasets.")
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--inference-length", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--vqgan-batch", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--num-spec-tokens", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--max-concurrent-prompts", type=int)
    parser.add_argument("--streaming", action="store_true",
                        help="Run mini benchmark in streaming mode (encode + decode interleaved) like InferenceActor.")
    parser.add_argument("--sleep-seconds", type=int, default=8,
                        help="Sleep between runs to let GPU memory fully release (seconds).")
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True,
                        help="List of inference batch sizes to evaluate")
    default_mini = (Path(__file__).resolve().parent / "mini_spec_benchmark.py")
    parser.add_argument("--mini-script", type=Path, default=default_mini)

    args = parser.parse_args()

    # 循环测试每个数据集
    for dataset_root in args.dataset_root:
        print(f"[SUITE] Testing dataset: {dataset_root}")
        common_args = build_common_args(args, dataset_root)
        
        # 对每个数据集测试不同的batch sizes
        for batch_size in args.batch_sizes:
            if not args.streaming and args.num_prompts % batch_size != 0:
                raise ValueError(f"num_prompts({args.num_prompts}) must be divisible by batch_size({batch_size})")
            cmd = [sys.executable, str(args.mini_script.resolve())]
            cmd.extend(common_args)
            cmd.extend(["--inference-batch-size", str(batch_size)])
            print(f"[SUITE] Running mini_spec_benchmark with dataset={dataset_root.name} batch_size={batch_size}")
            subprocess.run(cmd, check=True)
            # Allow driver/runtime to reclaim GPU memory between runs
            if args.sleep_seconds and args.sleep_seconds > 0:
                print(f"[SUITE] Sleeping {args.sleep_seconds}s to let GPU memory settle…")
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
