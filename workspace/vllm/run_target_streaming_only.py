#!/usr/bin/env python3
"""
Streaming image experiment: baseline (no spec) vs EAGLE3 speculative decoding.

一个脚本里同时跑两条路径：
- baseline：只启 target 模型（无推测解码）
- spec：target + EAGLE3 draft 模型（支持 LoRA 版 drafter）

输入是已经解码好的图片帧目录，内部用 VQGAN 编码成 token，并按 tream 的
流式策略构造 prompts。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import List, Sequence

import torch
from PIL import Image
from torchvision import transforms

# 在导入 vllm 之前，关掉 DeepGEMM 相关的东西，避免 deep_gemm_cpp ABI 错误。
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

from tream.lvm_tokenizer.muse import VQGANModel
import tream.lvm_tokenizer.utils as vq_utils
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams


def list_image_paths(video_dir: Path, max_frames: int | None = None) -> list[Path]:
    """遍历目录下的所有图片帧（jpg/jpeg/png），按路径排序。"""
    exts = (".jpg", ".jpeg", ".png")
    immediate = [p for p in video_dir.glob("*") if p.suffix.lower() in exts]
    if immediate:
        paths = immediate
    else:
        paths: list[Path] = []
        for ext in exts:
            paths.extend(video_dir.glob(f"**/*{ext}"))
    paths = [p for p in paths if p.is_file()]
    paths = sorted(paths, key=lambda p: (p.parent.as_posix(), p.name))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"{video_dir} 下未找到图像帧（支持扩展名：{exts}）")
    return paths


def encode_frames(
    frame_paths: Sequence[Path],
    encoder: VQGANModel,
    image_transform,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """使用 VQGAN 将帧编码为 token 序列。"""
    tokens_list: list[torch.Tensor] = []
    for start in range(0, len(frame_paths), batch_size):
        batch = frame_paths[start : start + batch_size]
        images = []
        for path in batch:
            with Image.open(path) as img:
                images.append(image_transform(img.convert("RGB")))
        image_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            _, tokens = encoder.encode(image_tensor)
        tokens_list.append(tokens.long().cpu())
    return torch.cat(tokens_list, dim=0)


def build_prompts_streaming(
    frame_tokens: Sequence[list[int]],
    context_length: int,
    batch_size: int,
    num_prompts: int,
) -> list[list[int]]:
    """
    按照 tream.InferenceActor 的逻辑构造“流式” prompt：
    - 逐帧积累，当帧数达到 batch_size 的倍数时，
      对最近 batch_size 帧构造 prompts，
      每条 prompt 由前 context_length 帧的 token 拼接。
    - 返回拍平成一维 list[list[int]]，方便依次送入 LLMEngine。
    """
    if num_prompts % batch_size != 0:
        raise ValueError("--num-prompts 必须是 --inference-batch-size 的整数倍")

    token_buffer: list[list[int]] = []
    prompt_batches: list[list[list[int]]] = []
    total_prompts = 0

    for idx, frame_token in enumerate(frame_tokens):
        token_buffer.append(frame_token)

        if (idx + 1) % batch_size != 0:
            continue

        buffer_length = len(token_buffer)
        start_idx = buffer_length - batch_size + 1
        batch_prompts: list[list[int]] = []
        for i in range(start_idx, buffer_length + 1):
            ctx_start = max(0, i - context_length)
            prompt_frames = token_buffer[ctx_start:i]
            prompt_tokens: list[int] = []
            for ft in prompt_frames:
                prompt_tokens.extend(ft)
            batch_prompts.append(prompt_tokens)
        prompt_batches.append(batch_prompts)
        total_prompts += len(batch_prompts)
        if total_prompts >= num_prompts:
            overflow = total_prompts - num_prompts
            if overflow > 0:
                prompt_batches[-1] = prompt_batches[-1][:-overflow]
            break

    flat_prompts: list[list[int]] = []
    for batch in prompt_batches:
        flat_prompts.extend(batch)

    if len(flat_prompts) != num_prompts:
        raise ValueError(f"仅构造出 {len(flat_prompts)} 条提示，少于期望 {num_prompts}")
    return flat_prompts


def prepare_prompts(args: argparse.Namespace) -> list[list[int]]:
    """从数据集目录读取帧，编码为 token，并构造流式 prompts。"""
    if args.num_prompts % args.inference_batch_size != 0:
        raise ValueError("--num-prompts 必须是 --inference-batch-size 的整数倍")

    frame_paths = list_image_paths(args.dataset_root, max_frames=args.max_frames)
    image_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )
    enc_device = torch.device(
        args.encoder_device
        if args.encoder_device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # 以模块文件所在目录为基准，定位 ckpt/laion
    vq_root = Path(vq_utils.__file__).parent
    ckpt_dir = vq_root / "ckpt" / "laion"
    encoder = VQGANModel.from_pretrained(str(ckpt_dir)).to(enc_device).eval()
    if args.encoder_dtype in {"fp16", "half"}:
        encoder = encoder.to(dtype=torch.float16)
    elif args.encoder_dtype in {"bf16", "bfloat16"}:
        encoder = encoder.to(dtype=torch.bfloat16)

    with torch.no_grad():
        frame_tokens = encode_frames(
            frame_paths, encoder, image_transform, enc_device, args.vqgan_batch
        )

    frame_tokens_list: list[list[int]] = frame_tokens.tolist()
    prompts = build_prompts_streaming(
        frame_tokens_list,
        args.context_length,
        args.inference_batch_size,
        args.num_prompts,
    )
    return prompts


def run_target_only(
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float]]:
    engine_args = EngineArgs(
        model=str(args.teacher_model),
        tokenizer=str(args.teacher_model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        enforce_eager=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    request_ids: List[str] = [f"target-only-{i}" for i in range(len(prompts))]
    pending = set(request_ids)
    texts: dict[str, str] = {}

    start = time.time()
    try:
        for rid, prompt in zip(request_ids, prompts):
            engine.add_request(
                request_id=rid,
                prompt=TokensPrompt(prompt_token_ids=prompt),
                params=sampling_params,
                lora_request=None,
            )

        while pending:
            outputs = engine.step()
            for output in outputs:
                if output.request_id in pending and output.finished:
                    text = output.outputs[0].text if output.outputs else ""
                    texts[output.request_id] = text
                    pending.remove(output.request_id)
    finally:
        try:
            engine.model_executor.shutdown()
        except Exception:
            pass
        del engine
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed = time.time() - start
    # 统计生成 token 数
    total_tokens = 0
    for rid in request_ids:
        # 文本级别只能粗略估计，这里不做 tokenizer 反向 encode，直接记录时间
        if rid in texts:
            total_tokens += len(texts[rid].split())

    outputs = [texts[rid] for rid in request_ids]
    stats = {
        "elapsed": float(elapsed),
        "approx_tokens": float(total_tokens),
        "throughput_tok_s": float(total_tokens / max(elapsed, 1e-6)),
    }
    return outputs, stats


def run_spec_eagle3(
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float]]:
    spec_cfg: dict[str, object] = {
        "method": "eagle3",
        "model": str(args.draft_model),
        "num_speculative_tokens": args.spec_num_tokens,
    }
    if args.draft_vocab_mapping_path is not None:
        spec_cfg["vocab_mapping_path"] = str(args.draft_vocab_mapping_path)

    engine_args = EngineArgs(
        model=str(args.teacher_model),
        tokenizer=str(args.teacher_model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        enforce_eager=False,
        speculative_config=spec_cfg,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    request_ids: List[str] = [f"spec-eagle3-{i}" for i in range(len(prompts))]
    pending = set(request_ids)
    texts: dict[str, str] = {}

    start = time.time()
    try:
        for rid, prompt in zip(request_ids, prompts):
            engine.add_request(
                request_id=rid,
                prompt=TokensPrompt(prompt_token_ids=prompt),
                params=sampling_params,
                lora_request=None,
            )

        while pending:
            outputs = engine.step()
            for output in outputs:
                if output.request_id in pending and output.finished:
                    text = output.outputs[0].text if output.outputs else ""
                    texts[output.request_id] = text
                    pending.remove(output.request_id)
    finally:
        try:
            engine.model_executor.shutdown()
        except Exception:
            pass
        del engine
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed = time.time() - start
    total_tokens = 0
    for rid in request_ids:
        if rid in texts:
            total_tokens += len(texts[rid].split())

    outputs = [texts[rid] for rid in request_ids]
    stats = {
        "elapsed": float(elapsed),
        "approx_tokens": float(total_tokens),
        "throughput_tok_s": float(total_tokens / max(elapsed, 1e-6)),
    }
    return outputs, stats


def run_spec_eagle3_lora_path(
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float]]:
    """Speculative decoding with EAGLE3 + LoRA path enabled but no adapter."""
    spec_cfg: dict[str, object] = {
        "method": "eagle3",
        "model": str(args.draft_model),
        "num_speculative_tokens": args.spec_num_tokens,
    }
    if args.draft_vocab_mapping_path is not None:
        spec_cfg["vocab_mapping_path"] = str(args.draft_vocab_mapping_path)

    engine_args = EngineArgs(
        model=str(args.teacher_model),
        tokenizer=str(args.teacher_model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        enforce_eager=False,
        speculative_config=spec_cfg,
        enable_lora=True,  # 开启 LoRA 基础设施，但不传 LoRARequest
    )
    engine = LLMEngine.from_engine_args(engine_args)

    request_ids: List[str] = [f"spec-eagle3-lorapath-{i}" for i in range(len(prompts))]
    pending = set(request_ids)
    texts: dict[str, str] = {}

    start = time.time()
    try:
        for rid, prompt in zip(request_ids, prompts):
            engine.add_request(
                request_id=rid,
                prompt=TokensPrompt(prompt_token_ids=prompt),
                params=sampling_params,
                lora_request=None,
            )

        while pending:
            outputs = engine.step()
            for output in outputs:
                if output.request_id in pending and output.finished:
                    text = output.outputs[0].text if output.outputs else ""
                    texts[output.request_id] = text
                    pending.remove(output.request_id)
    finally:
        try:
            engine.model_executor.shutdown()
        except Exception:
            pass
        del engine
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    elapsed = time.time() - start
    total_tokens = 0
    for rid in request_ids:
        if rid in texts:
            total_tokens += len(texts[rid].split())

    outputs = [texts[rid] for rid in request_ids]
    stats = {
        "elapsed": float(elapsed),
        "approx_tokens": float(total_tokens),
        "throughput_tok_s": float(total_tokens / max(elapsed, 1e-6)),
    }
    return outputs, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming image prompts: compare target-only vs EAGLE3 speculative decoding."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="包含解码后图像帧的目录（支持 jpg/jpeg/png）。",
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        required=True,
        help="目标 LVM 模型路径 (e.g., /workspace/tream/lvm-llama2-7b)。",
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        required=True,
        help="EAGLE3 draft 模型路径（可为 LoRA 版 drafter，例如 SpecForge 训练输出目录）。",
    )
    parser.add_argument(
        "--draft-vocab-mapping-path",
        type=Path,
        default=None,
        help="可选的 vocab_mapping .pt 路径，用于对齐 draft vocab。",
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--vqgan-batch", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.75)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[
            "auto",
            "half",
            "float16",
            "fp16",
            "bf16",
            "bfloat16",
            "float32",
            "fp32",
        ],
    )
    parser.add_argument("--encoder-device", type=str, default=None)
    parser.add_argument(
        "--encoder-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "half", "bf16", "bfloat16"],
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--spec-num-tokens",
        type=int,
        default=2,
        help="EAGLE3 num_speculative_tokens。",
    )

    args = parser.parse_args()

    print("[INFO] 准备流式 prompts（来自图片帧） ...")
    prompts = prepare_prompts(args)
    print(f"[INFO] 共生成 {len(prompts)} 条 prompts。")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    # Baseline: target-only
    print("\n" + "=" * 80)
    print("[PHASE] Target-only (no speculative decoding)")
    base_texts, base_stats = run_target_only(prompts, sampling_params, args)
    print(
        f"[BASE] elapsed={base_stats['elapsed']:.2f}s, "
        f"approx_tokens={base_stats['approx_tokens']:.0f}, "
        f"throughput≈{base_stats['throughput_tok_s']:.2f} tok/s"
    )

    # B1: Speculative: EAGLE3（不启 LoRA）
    print("\n" + "=" * 80)
    print("[PHASE] B1: EAGLE3 speculative decoding (no LoRA)")
    spec_texts, spec_stats = run_spec_eagle3(prompts, sampling_params, args)
    print(
        f"[SPEC] elapsed={spec_stats['elapsed']:.2f}s, "
        f"approx_tokens={spec_stats['approx_tokens']:.0f}, "
        f"throughput≈{spec_stats['throughput_tok_s']:.2f} tok/s"
    )

    # B2: Speculative: EAGLE3 + LoRA path enabled, but no adapter
    print("\n" + "=" * 80)
    print("[PHASE] B2: EAGLE3 speculative decoding (LoRA path, no adapter)")
    spec_lora_texts, spec_lora_stats = run_spec_eagle3_lora_path(
        prompts, sampling_params, args
    )
    print(
        f"[SPEC+LoRA] elapsed={spec_lora_stats['elapsed']:.2f}s, "
        f"approx_tokens={spec_lora_stats['approx_tokens']:.0f}, "
        f"throughput≈{spec_lora_stats['throughput_tok_s']:.2f} tok/s"
    )

    print("\n" + "=" * 80)
    # 只关心加速比
    def _spd(base: float, other: float) -> float:
        return base / other if other > 0 else float("inf")

    speedup_spec = _spd(base_stats["elapsed"], spec_stats["elapsed"])
    speedup_spec_lora = _spd(base_stats["elapsed"], spec_lora_stats["elapsed"])

    print(
        "[SPEEDUP] base / spec       = "
        f"{base_stats['elapsed']:.2f}s / {spec_stats['elapsed']:.2f}s "
        f"= {speedup_spec:.2f}x"
    )
    print(
        "[SPEEDUP] base / spec_lorap = "
        f"{base_stats['elapsed']:.2f}s / {spec_lora_stats['elapsed']:.2f}s "
        f"= {speedup_spec_lora:.2f}x"
    )


if __name__ == "__main__":
    main()
