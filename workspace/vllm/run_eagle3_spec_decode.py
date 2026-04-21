#!/usr/bin/env python3
import os
import argparse

# 在导入 vllm 之前，关掉 DeepGEMM 相关的东西
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

from vllm import LLM
from vllm.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="Minimal EAGLE3 speculative decoding sanity script",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="Base teacher model path, e.g. /workspace/tream/lvm-llama2-7b",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        required=True,
        help="EAGLE3 draft model path (can be your LoRA-trained draft), "
             "e.g. /workspace/SpecForge/output/lvm_eagle3_lora_v1/epoch_0_step_3475",
    )
    parser.add_argument(
        "--draft-vocab-mapping-path",
        type=str,
        default=None,
        help="Optional vocab_mapping .pt path if you有，用来对齐 draft vocab",
    )
    parser.add_argument(
        "--num-spec-tokens",
        type=int,
        default=2,
        help="num_speculative_tokens for EAGLE3",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        help="Prompt 文本，可以多次指定；不指定就用内置样例",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="每个请求生成的最大 tokens",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "half", "float16", "fp16", "bf16", "bfloat16", "float32", "fp32"],
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    if args.prompt:
        prompts = args.prompt
    else:
        prompts = [
            "You are an assistant. Explain what EAGLE3 speculative decoding does in one paragraph.",
            "List three differences between standard decoding and speculative decoding.",
        ]

    # 构建 EAGLE3 的 speculative_config，model 直接指向你的 EAGLE3 LoRA 目录
    spec_cfg = {
        "method": "eagle3",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_spec_tokens,
    }
    if args.draft_vocab_mapping_path:
        spec_cfg["vocab_mapping_path"] = args.draft_vocab_mapping_path

    print("[INFO] Initializing LLM with EAGLE3 drafter ...")
    llm = LLM(
        model=args.teacher_model,
        tokenizer=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        speculative_config=spec_cfg,
        enforce_eager=False,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    print("[INFO] Running speculative decoding for", len(prompts), "prompts ...")
    outputs = llm.generate(prompts, sampling)

    for i, out in enumerate(outputs):
        print("=" * 80)
        print(f"[PROMPT {i}] {prompts[i]!r}")
        if out.outputs:
            print(f"[SPEC OUTPUT {i}] {out.outputs[0].text!r}")
        else:
            print(f"[SPEC OUTPUT {i}] <empty>")

    print("=" * 80)
    print("[DONE] Speculative decoding finished without DeepGEMM.")


if __name__ == "__main__":
    main()
