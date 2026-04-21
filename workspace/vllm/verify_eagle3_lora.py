#!/usr/bin/env python3
"""
EAGLE3 LoRA Stability Verification Script

验证 EAGLE3 LoRA 训练是否成功。通过对比以下两个阶段来评估 LoRA 对模型行为的影响：
  B1 (base):  EAGLE3 without LoRA  （enable_lora=False，且不创建 LoRARequest）
  B2 (lora):  EAGLE3 with LoRA path
      - 如果不给 --lora-adapter-dir：只打开 LoRA 路径（enable_lora=True），不挂载任何权重；
      - 如果传入 --lora-adapter-dir：在 generate 时真正带上 LoRARequest。

支持两种输入：
  - 默认：随机 token prompts；
  - 如果提供 --dataset-root：先用 VQGAN 把图片解码成 tokens，再用这些 tokens 作为 prompts。

使用 VLLM_DRAFT_TOP1_LOG 来捕获每一步的草稿预测，然后计算 token 级别的匹配率。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from torchvision import transforms

from tream.lvm_tokenizer.muse import VQGANModel
import tream.lvm_tokenizer.utils as vq_utils

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


@contextmanager
def eagle_debug_context(tag: str, debug_dir: Optional[Path]):
    """设置 EAGLE3 调试环境"""
    if debug_dir is None:
        yield
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"{tag}_eagle.jsonl"
    old_phase = os.environ.get("PHASE_NAME")
    old_flag = os.environ.get("VLLM_EAGLE_DEBUG")
    old_path = os.environ.get("VLLM_EAGLE_DEBUG_PATH")
    os.environ["PHASE_NAME"] = tag
    os.environ["VLLM_EAGLE_DEBUG"] = "1"
    os.environ["VLLM_EAGLE_DEBUG_PATH"] = str(debug_file)
    try:
        yield
    finally:
        if old_phase is None:
            os.environ.pop("PHASE_NAME", None)
        else:
            os.environ["PHASE_NAME"] = old_phase
        if old_flag is None:
            os.environ.pop("VLLM_EAGLE_DEBUG", None)
        else:
            os.environ["VLLM_EAGLE_DEBUG"] = old_flag
        if old_path is None:
            os.environ.pop("VLLM_EAGLE_DEBUG_PATH", None)
        else:
            os.environ["VLLM_EAGLE_DEBUG_PATH"] = old_path


def list_image_paths(root: Path, max_frames: int | None = None) -> list[Path]:
    """递归枚举 root 下所有 jpg/jpeg/png 帧，按路径排序。"""
    exts = (".jpg", ".jpeg", ".png")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(root.glob(f"**/*{ext}"))
    paths = [p for p in paths if p.is_file()]
    paths = sorted(paths, key=lambda p: (p.parent.as_posix(), p.name))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"{root} 下未找到图像帧（支持扩展名：{exts}）")
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


def prepare_image_prompts(args: argparse.Namespace) -> list[list[int]]:
    """从图片目录解码出 VQ tokens 作为 prompts。"""
    assert args.dataset_root is not None
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
    if len(frame_tokens_list) < args.num_prompts:
        raise ValueError(
            f"可用帧数量 {len(frame_tokens_list)} 小于 num_prompts={args.num_prompts}"
        )

    prompts = frame_tokens_list[: args.num_prompts]
    print(
        f"[INFO] 从 {args.dataset_root} 解码得到 {len(prompts)} 个 prompts "
        f"(每个长度约 {len(prompts[0])} tokens)"
    )
    return prompts


def prepare_random_prompts(
    num_prompts: int, seq_length: int, vocab_size: int = 32000
) -> list[list[int]]:
    """生成随机 token 序列作为测试 prompts"""
    import random

    prompts = []
    for _ in range(num_prompts):
        prompt = [random.randint(1, vocab_size - 1) for _ in range(seq_length)]
        prompts.append(prompt)
    return prompts


def build_engine_args(
    model_dir: Path,
    tokenizer_dir: Path,
    tensor_parallel_size: int,
    gpu_mem_util: float,
    dtype: str,
    enable_lora: bool,
    max_loras: int,
    max_lora_rank: int,
    speculative_config: dict,
) -> EngineArgs:
    """构建 vLLM 引擎参数"""
    kwargs = dict(
        model=str(model_dir),
        tokenizer=str(tokenizer_dir),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
        dtype=dtype,
        enforce_eager=False,
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        speculative_config=speculative_config,
    )
    return EngineArgs(**kwargs)


def run_spec(
    tag: str,
    engine_args: EngineArgs,
    prompts: Sequence[Sequence[int]],
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest],
    *,
    eagle_debug_dir: Optional[Path] = None,
    sleep_after: float = 10.0,
) -> list[list[int]]:
    """运行推理并捕获草稿 tokens"""
    engine_args_copy = copy.deepcopy(engine_args)
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", prefix=f"{tag}_draft_", delete=False
    ) as tmp:
        log_path = Path(tmp.name)
    old_env = os.environ.get("VLLM_DRAFT_TOP1_LOG")
    os.environ["VLLM_DRAFT_TOP1_LOG"] = str(log_path)

    def _restore_env():
        if old_env is None:
            os.environ.pop("VLLM_DRAFT_TOP1_LOG", None)
        else:
            os.environ["VLLM_DRAFT_TOP1_LOG"] = old_env

    print(f"[PHASE] Starting {tag}...")
    with eagle_debug_context(tag, eagle_debug_dir):
        engine = LLMEngine.from_engine_args(engine_args_copy)
        request_ids = [f"{tag}-req-{idx}" for idx in range(len(prompts))]
        pending = set(request_ids)
        try:
            for idx, prompt in enumerate(prompts):
                engine.add_request(
                    request_id=request_ids[idx],
                    prompt=TokensPrompt(prompt_token_ids=list(prompt)),
                    params=sampling_params,
                    lora_request=lora_request,
                )
            while pending:
                outputs = engine.step()
                for output in outputs:
                    if output.request_id in pending and output.finished:
                        pending.remove(output.request_id)
        finally:
            try:
                engine.model_executor.shutdown()
            except Exception:
                pass
            try:
                engine.vllm_config.compilation_config.static_forward_context.clear()
            except Exception:
                pass
            del engine
    _restore_env()

    sequences = parse_log(log_path, request_ids)
    log_path.unlink(missing_ok=True)

    if sleep_after and sleep_after > 0:
        print(f"[PHASE] {tag} completed. Sleeping {sleep_after} seconds...")
        time.sleep(sleep_after)
    return sequences


def parse_log(log_path: Path, request_ids: Sequence[str]) -> list[list[int]]:
    """从日志中解析草稿 tokens"""
    seqs = {rid: [] for rid in request_ids}
    if not log_path.exists():
        print(f"[WARN] Log file not found: {log_path}")
        return [seqs[rid] for rid in request_ids]

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                for entry in record.get("requests", []):
                    rid = entry.get("request_id")
                    tokens = entry.get("draft_tokens", [])
                    if rid not in seqs or not tokens:
                        continue
                    seqs[rid].extend(tokens)
            except json.JSONDecodeError:
                continue
    return [seqs[rid] for rid in request_ids]


def compare_sequences(
    seqs_a: Sequence[Sequence[int]],
    seqs_b: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    """比较两组序列的相似性"""
    assert len(seqs_a) == len(seqs_b)
    total_positions = 0
    match_positions = 0
    exact_matches = 0

    for a, b in zip(seqs_a, seqs_b):
        L = min(len(a), len(b))
        if L == 0:
            continue
        total_positions += L
        match_positions += sum(1 for i in range(L) if a[i] == b[i])
        if a[:L] == b[:L]:
            exact_matches += 1

    prompts = len(seqs_a)
    valid = sum(1 for a, b in zip(seqs_a, seqs_b) if min(len(a), len(b)) > 0)

    return {
        "num_prompts": prompts,
        "valid_prompts": valid,
        "token_match_rate": (match_positions / total_positions)
        if total_positions
        else 0.0,
        "exact_prefix_match_rate": (exact_matches / valid) if valid else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-model",
        type=Path,
        required=True,
        help="目标模型路径 (e.g., /workspace/tream/lvm-llama2-7b)",
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        required=True,
        help="EAGLE3 draft 模型目录（例如 SpecForge 训练输出的某个 epoch 目录）",
    )
    parser.add_argument(
        "--lora-adapter-dir",
        type=Path,
        default=None,
        help="LoRA 适配器目录 (可选；不提供则表示\"没有 LoRA 权重\"的场景)",
    )
    parser.add_argument("--num-prompts", type=int, default=32, help="测试 prompt 数量")
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="每个 prompt 的 token 长度（仅用于随机模式）",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000, help="随机模式下的词汇表大小"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.75)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "half", "float16", "fp16", "bf16", "bfloat16", "float32", "fp32"],
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="每个请求生成的最大 tokens"
    )
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--spec-method", type=str, default="eagle3")
    parser.add_argument("--spec-num-tokens", type=int, default=2)
    parser.add_argument(
        "--draft-vocab-mapping-path",
        type=Path,
        default=None,
        help="词表映射文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eagle3_lora_verification.json"),
    )
    parser.add_argument("--save-sequences", action="store_true")
    parser.add_argument(
        "--eagle-debug-dir",
        type=Path,
        default=None,
        help="保存 EAGLE3 调试日志的目录",
    )
    # 可选：从图片目录解码 prompts
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="如果提供，则从该目录解码图片为 VQ tokens 作为 prompts",
    )
    parser.add_argument("--vqgan-batch", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--encoder-device", type=str, default=None)
    parser.add_argument(
        "--encoder-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "half", "bf16", "bfloat16"],
    )

    args = parser.parse_args()

    # 构造 prompts：优先使用图片解码，其次随机 tokens
    if args.dataset_root is not None:
        prompts = prepare_image_prompts(args)
    else:
        prompts = prepare_random_prompts(
            args.num_prompts, args.prompt_length, args.vocab_size
        )
        print(
            f"[INFO] 生成了 {len(prompts)} 个随机测试 prompts (长度: {args.prompt_length})"
        )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    # 构建 EAGLE3 配置
    spec_cfg = {
        "method": args.spec_method,
        "model": str(args.draft_model),
        "num_speculative_tokens": args.spec_num_tokens,
    }
    if args.draft_vocab_mapping_path:
        spec_cfg["vocab_mapping_path"] = str(args.draft_vocab_mapping_path)

    # B1: 基础 EAGLE3 (无 LoRA)
    print("\n" + "=" * 60)
    print("[PHASE] ===== B1: EAGLE3 Base (without LoRA) =====")
    print("=" * 60)
    base_args = build_engine_args(
        args.teacher_model,
        args.teacher_model,
        args.tensor_parallel_size,
        args.gpu_mem_util,
        args.dtype,
        enable_lora=False,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        speculative_config=spec_cfg,
    )
    draft_base = run_spec(
        "eagle3-base",
        base_args,
        prompts,
        sampling_params,
        None,
        eagle_debug_dir=args.eagle_debug_dir,
    )

    # B2: EAGLE3 with LoRA path (enable_lora=True，LoRARequest 可选)
    print("\n" + "=" * 60)
    print("[PHASE] ===== B2: EAGLE3 with LoRA path =====")
    print("=" * 60)

    lora_args = copy.deepcopy(base_args)
    lora_args.enable_lora = True

    if args.lora_adapter_dir is not None:
        lora_request = LoRARequest(
            lora_name="eagle3-lora",
            lora_int_id=1,
            lora_local_path=str(args.lora_adapter_dir),
        )
        print(f"[INFO] LoRA adapter dir configured: {args.lora_adapter_dir}")
    else:
        lora_request = None
        print(
            "[INFO] B2: no --lora-adapter-dir provided -> enable_lora=True 但不挂载任何 LoRA 权重"
        )

    draft_lora = run_spec(
        "eagle3-lora",
        lora_args,
        prompts,
        sampling_params,
        lora_request,
        eagle_debug_dir=args.eagle_debug_dir,
    )

    # 比较结果
    print("\n" + "=" * 60)
    print("[RESULT] Verification Results")
    print("=" * 60)

    comparison = compare_sequences(draft_base, draft_lora)
    print(f"[METRIC] Token Match Rate: {comparison['token_match_rate']:.4f}")
    print(
        f"[METRIC] Exact Prefix Match Rate: {comparison['exact_prefix_match_rate']:.4f}"
    )
    print(
        f"[METRIC] Valid Prompts: {comparison['valid_prompts']} / {comparison['num_prompts']}"
    )

    # 保存结果
    results = {
        "config": {
            "num_prompts": args.num_prompts,
            "prompt_length": args.prompt_length,
            "spec_method": args.spec_method,
            "spec_num_tokens": args.spec_num_tokens,
            "max_tokens": args.max_tokens,
            "draft_model": str(args.draft_model),
            "lora_adapter_dir": str(args.lora_adapter_dir),
            "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        },
        "comparison": {
            "B1_base_vs_B2_lora_path": comparison,
        },
    }

    if args.save_sequences:
        results["sequences"] = {
            "B1_base": draft_base,
            "B2_lora_path": draft_lora,
        }

    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Results saved to {args.output}")

    if comparison["token_match_rate"] > 0.99:
        print(
            "✅ [SUCCESS] LoRA path does NOT significantly affect EAGLE3 draft predictions"
        )
    else:
        print(
            "⚠️  [WARNING] LoRA path may affect EAGLE3 draft predictions"
        )


if __name__ == "__main__":
    main()
