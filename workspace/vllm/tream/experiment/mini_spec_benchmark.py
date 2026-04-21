#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化推测解码基准测试脚本，与 tream 推理路径配置保持一致
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lvm_tokenizer.muse import VQGANModel
from lvm_tokenizer.utils import ENCODING_SIZE, RAW_VQGAN_PATH
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
import torch.cuda.nvtx as nvtx
from vllm.lora.request import LoRARequest
# vLLM 0.8.x exposes internal SpecMetrics; 0.10.x removed/relocated it.
# Try to import and fallback to a no-op shim for compatibility.
try:
    from vllm.spec_decode._spec_metrics import SpecMetrics  # type: ignore
except Exception:
    class SpecMetrics:  # type: ignore
        _path: str | None = None

        @staticmethod
        def flush() -> None:
            return

def list_image_paths(video_dir: Path, max_frames: int | None = None) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png")
    immediate = [p for p in video_dir.glob("*") if p.suffix.lower() in exts]
    if immediate:
        paths = immediate
    else:
        paths = []
        for ext in exts:
            paths.extend(video_dir.glob(f"**/*{ext}"))
    paths = [p for p in paths if p.is_file()]
    paths = sorted(paths, key=lambda p: (p.parent.as_posix(), p.name))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"{video_dir} 下未找到图像帧（支持扩展名：{exts}）")
    return paths


def encode_frames(frame_paths: Sequence[Path],
                  encoder: VQGANModel,
                  image_transform,
                  device: torch.device,
                  batch_size: int) -> torch.Tensor:
    tokens_list: list[torch.Tensor] = []
    for start in range(0, len(frame_paths), batch_size):
        batch = frame_paths[start:start + batch_size]
        images = []
        for path in batch:
            with Image.open(path) as img:
                images.append(image_transform(img.convert("RGB")))
        image_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            _, tokens = encoder.encode(image_tensor)
        tokens_list.append(tokens.long().cpu())
    return torch.cat(tokens_list, dim=0)


def build_prompts_streaming(frame_tokens: Sequence[list[int]],
                            context_length: int,
                            batch_size: int,
                            num_prompts: int) -> list[list[list[int]]]:
    """
    按照 tream.InferenceActor 的逻辑：逐帧积累，当帧数达到 batch_size 的倍数时，
    对最近 batch_size 帧构造 prompt，每条 prompt 由前 context_length 帧的 token 拼接。
    """
    if num_prompts % batch_size != 0:
        raise ValueError("num_prompts 必须是 inference_batch_size 的整数倍。")

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
            # Trim excess prompts from the last batch if needed
            overflow = total_prompts - num_prompts
            if overflow > 0:
                prompt_batches[-1] = prompt_batches[-1][:-overflow]
            return prompt_batches

    raise ValueError(f"仅构造出 {total_prompts} 条提示，少于期望 {num_prompts}")


@dataclass
class RunResult:
    mode: str
    total_time: float
    prompt_tokens: int
    generated_tokens: int
    throughput: float
    avg_latency: float
    acceptance_rate: float | None = None
    accepted_tokens: int | None = None
    proposed_tokens: int | None = None
    profiling: dict[str, Any] | None = None


def run_inference(
    mode: str,
    prompt_batches: Sequence[Sequence[Sequence[int]]],
    sampling_params: SamplingParams,
    engine_args: EngineArgs,
    spec_metrics_path: Path | None = None,
    chunk_size: int | None = None,
    lora_request: LoRARequest | None = None,
) -> RunResult:
    if spec_metrics_path is not None:
        if spec_metrics_path.exists():
            spec_metrics_path.unlink()
        SpecMetrics.flush()
        SpecMetrics._path = str(spec_metrics_path)  # type: ignore[attr-defined]

    # Time engine initialization to include in profiling
    with nvtx.range(f"{mode}/engine_init"):
        t_engine_init0 = time.time()
        engine = LLMEngine.from_engine_args(engine_args)
        t_engine_init = time.time() - t_engine_init0
    request_ids: list[str] = []

    def process_chunk(chunk_prompts: Sequence[Sequence[int]],
                      start_idx: int) -> tuple[int, int, dict[str, Any]]:
        chunk_start = time.time()
        add_start = time.time()
        local_prompt_tokens = sum(len(p) for p in chunk_prompts)
        local_generated_tokens = 0
        chunk_request_ids: list[str] = []
        for j, prompt in enumerate(chunk_prompts):
            req_id = f"{mode}-req-{start_idx + j}"
            request_ids.append(req_id)
            chunk_request_ids.append(req_id)
            engine.add_request(
                request_id=req_id,
                prompt=TokensPrompt(prompt_token_ids=prompt),
                params=sampling_params,
                lora_request=lora_request,
            )
        add_time = time.time() - add_start

        completed = 0
        step_calls = 0
        decode_time = 0.0
        while completed < len(chunk_prompts):
            step_start = time.time()
            outputs = engine.step()
            decode_time += time.time() - step_start
            step_calls += 1
            for output in outputs:
                if output.request_id in chunk_request_ids and output.finished:
                    local_generated_tokens += len(output.outputs[0].token_ids)
                    completed += 1

        chunk_elapsed = time.time() - chunk_start
        chunk_entry = {
            "chunk_size": len(chunk_prompts),
            "prompt_tokens": local_prompt_tokens,
            "generated_tokens": local_generated_tokens,
            "add_time": add_time,
            "decode_time": decode_time,
            "chunk_time": chunk_elapsed,
            "step_calls": step_calls,
        }
        return local_prompt_tokens, local_generated_tokens, chunk_entry

    start = time.time()
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_requests = 0
    chunk_stats: list[dict[str, Any]] = []

    try:
        for batch_prompts in prompt_batches:
            if chunk_size is None or chunk_size >= len(batch_prompts):
                local_prompt_tokens, local_generated_tokens, chunk_entry = process_chunk(
                    batch_prompts, total_requests)
                chunk_stats.append(chunk_entry)
                total_prompt_tokens += local_prompt_tokens
                total_generated_tokens += local_generated_tokens
                total_requests += len(batch_prompts)
            else:
                for offset in range(0, len(batch_prompts), chunk_size):
                    chunk = batch_prompts[offset:offset + chunk_size]
                    local_prompt_tokens, local_generated_tokens, chunk_entry = process_chunk(
                        chunk, total_requests)
                    chunk_stats.append(chunk_entry)
                    total_prompt_tokens += local_prompt_tokens
                    total_generated_tokens += local_generated_tokens
                    total_requests += len(chunk)
    finally:
        # Ensure baseline engine releases memory before speculative run.
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
    throughput = total_generated_tokens / max(elapsed, 1e-6)
    avg_latency = elapsed / max(total_requests, 1)

    acceptance_rate = None
    accepted = None
    proposed = None
    if spec_metrics_path is not None and spec_metrics_path.exists():
        accepted, proposed = parse_spec_metrics(spec_metrics_path)
        if proposed:
            acceptance_rate = accepted / proposed

    profiling_summary = {
        "num_chunks": len(chunk_stats),
        "total_add_time": sum(cs["add_time"] for cs in chunk_stats),
        "total_decode_time": sum(cs["decode_time"] for cs in chunk_stats),
        "total_chunk_time": sum(cs["chunk_time"] for cs in chunk_stats),
        "total_step_calls": sum(cs["step_calls"] for cs in chunk_stats),
        "engine_init_time": t_engine_init,
        "chunk_stats": chunk_stats,
    }

    return RunResult(
        mode=mode,
        total_time=elapsed,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        throughput=throughput,
        avg_latency=avg_latency,
        acceptance_rate=acceptance_rate,
        accepted_tokens=accepted,
        proposed_tokens=proposed,
        profiling=profiling_summary,
    )


def run_inference_streaming(
    mode: str,
    frame_paths: Sequence[Path],
    encoder: VQGANModel,
    image_transform,
    encoder_device: torch.device,
    engine_device: torch.device,
    context_length: int,
    batch_size: int,
    inference_length: int,
    vqgan_batch: int,
    sampling_params: SamplingParams,
    engine_args: EngineArgs,
    spec_metrics_path: Path | None = None,
    lora_request: LoRARequest | None = None,
) -> RunResult:
    if spec_metrics_path is not None:
        if spec_metrics_path.exists():
            spec_metrics_path.unlink()
        SpecMetrics.flush()
        SpecMetrics._path = str(spec_metrics_path)  # type: ignore[attr-defined]

    # Time engine initialization for streaming path
    t_engine_init0 = time.time()
    engine = LLMEngine.from_engine_args(engine_args)
    t_engine_init = time.time() - t_engine_init0

    token_buffer: list[list[int]] = []
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_requests = 0

    # Simple profiling per macro cycle
    chunk_stats: list[dict[str, Any]] = []

    # Accuracy measurement cache and counters (align with InferenceActor)
    prediction_cache: list[dict[str, Any]] = []
    total_matches = 0
    total_compared = 0

    start = time.time()

    def process_macro_cycle() -> None:
        nonlocal total_prompt_tokens, total_generated_tokens, total_requests
        nonlocal total_matches, total_compared
        if len(token_buffer) < batch_size:
            return
        buffer_length = len(token_buffer)
        batch_prompts: list[list[int]] = []
        for i in range(buffer_length - batch_size + 1, buffer_length + 1):
            ctx_start = max(0, i - context_length)
            prompt_tokens: list[int] = []
            for ft in token_buffer[ctx_start:i]:
                prompt_tokens.extend(ft)
            batch_prompts.append(prompt_tokens)

        # Time one macro cycle similar to actor
        with nvtx.range(f"{mode}/macro_cycle"):
            cycle_start = time.time()
            with nvtx.range(f"{mode}/add_requests"):
                add_start = time.time()
                request_ids: list[str] = []
                local_prompt_tokens = sum(len(p) for p in batch_prompts)
                local_generated_tokens = 0
                for j, prompt in enumerate(batch_prompts):
                    req_id = f"{mode}-stream-req-{total_requests + j}"
                    request_ids.append(req_id)
                    engine.add_request(
                        request_id=req_id,
                        prompt=TokensPrompt(prompt_token_ids=prompt),
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                add_time = time.time() - add_start

            completed = 0
            step_calls = 0
            decode_time = 0.0
            # collect generated tokens per request id
            req_tokens: dict[str, list[int]] = {}
            with nvtx.range(f"{mode}/decode_loop"):
                while completed < len(batch_prompts):
                    t0 = time.time()
                    with nvtx.range(f"{mode}/engine_step"):
                        outputs = engine.step()
                    decode_time += time.time() - t0
                    step_calls += 1
                    for output in outputs:
                        if output.request_id in request_ids and output.finished:
                            gen_ids = list(output.outputs[0].token_ids)
                            req_tokens[output.request_id] = gen_ids
                            local_generated_tokens += len(gen_ids)
                            completed += 1

            cycle_elapsed = time.time() - cycle_start
            chunk_stats.append({
                "chunk_size": len(batch_prompts),
                "prompt_tokens": local_prompt_tokens,
                "generated_tokens": local_generated_tokens,
                "add_time": add_time,
                "decode_time": decode_time,
                "chunk_time": cycle_elapsed,
                "step_calls": step_calls,
            })
            total_prompt_tokens += local_prompt_tokens
            total_generated_tokens += local_generated_tokens
            total_requests += len(batch_prompts)

        # Cache predictions for later accuracy check when ground truth arrives
        macro_start = buffer_length - batch_size + 1  # frame index (1-based style)
        for j, req_id in enumerate(request_ids):
            pred = req_tokens.get(req_id)
            if pred is None:
                continue
            # required frames are [start_idx, start_idx + inference_length)
            required_start_idx = macro_start + j
            prediction_cache.append({
                "predicted_tokens": pred,
                "required_start_idx": required_start_idx,
                "required_frame_count": inference_length,
            })

        # Try to evaluate any cached predictions whose ground truth is available now
        eval_any = True
        while eval_any:
            eval_any = False
            cur_len = len(token_buffer)
            for entry in list(prediction_cache):
                rs = entry["required_start_idx"]
                rc = entry["required_frame_count"]
                if rs + rc <= cur_len:
                    # build ground truth tokens from frames [rs, rs+rc)
                    gt: list[int] = []
                    for fi in range(rs, rs + rc):
                        gt.extend(token_buffer[fi - 1])
                    pred_tokens: list[int] = entry["predicted_tokens"]
                    if pred_tokens and gt:
                        mlen = min(len(pred_tokens), len(gt))
                        matches = sum(1 for i in range(mlen) if pred_tokens[i] == gt[i])
                        total_matches += matches
                        total_compared += mlen
                    prediction_cache.remove(entry)
                    eval_any = True

    # Stream frames EXACTLY like InferenceActor: encode one image at a time
    # and trigger a macro cycle whenever the buffer reaches a multiple of
    # batch_size.
    try:
        for path in frame_paths:
            with Image.open(path) as img:
                img_tensor = image_transform(img.convert("RGB")).unsqueeze(0)
            img_tensor = img_tensor.to(encoder_device, non_blocking=True)
            with torch.no_grad():
                with nvtx.range("vqgan/encode_1"):
                    _, t = encoder.encode(img_tensor)
            token_buffer.append(t[0].long().cpu().tolist())
            if len(token_buffer) % batch_size == 0:
                process_macro_cycle()
    finally:
        # Ensure this engine is released before returning.
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
    throughput = total_generated_tokens / max(elapsed, 1e-6)
    avg_latency = elapsed / max(total_requests, 1)

    acceptance_rate = None
    accepted = None
    proposed = None
    if spec_metrics_path is not None and spec_metrics_path.exists():
        accepted, proposed = parse_spec_metrics(spec_metrics_path)
        if proposed:
            acceptance_rate = accepted / proposed

    profiling_summary = {
        "num_chunks": len(chunk_stats),
        "total_add_time": sum(cs["add_time"] for cs in chunk_stats) if chunk_stats else 0.0,
        "total_decode_time": sum(cs["decode_time"] for cs in chunk_stats) if chunk_stats else 0.0,
        "total_chunk_time": sum(cs["chunk_time"] for cs in chunk_stats) if chunk_stats else 0.0,
        "total_step_calls": sum(cs["step_calls"] for cs in chunk_stats) if chunk_stats else 0,
        "engine_init_time": t_engine_init,
        "chunk_stats": chunk_stats,
        "acc_total_matches": total_matches,
        "acc_total_compared": total_compared,
        "acc_accuracy": (total_matches / total_compared) if total_compared > 0 else None,
    }

    return RunResult(
        mode=mode,
        total_time=elapsed,
        prompt_tokens=total_prompt_tokens,
        generated_tokens=total_generated_tokens,
        throughput=throughput,
        avg_latency=avg_latency,
        acceptance_rate=acceptance_rate,
        accepted_tokens=accepted,
        proposed_tokens=proposed,
        profiling=profiling_summary,
    )


def parse_spec_metrics(spec_path: Path) -> tuple[int, int]:
    accepted = 0
    proposed = 0
    with spec_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            accepted += int(record.get("accepted", 0))
            proposed += int(record.get("proposed", 0))
    return accepted, proposed


def format_result(result: RunResult) -> str:
    base = [
        f"模式: {result.mode}",
        f"总耗时: {result.total_time:.3f}s",
        f"提示 token: {result.prompt_tokens}",
        f"生成 token: {result.generated_tokens}",
        f"吞吐: {result.throughput:.2f} tokens/s",
        f"平均单请求延迟: {result.avg_latency:.3f}s",
    ]
    if result.acceptance_rate is not None:
        base.append(f"草稿接受率: {result.acceptance_rate * 100:.2f}% "
                    f"(accepted={result.accepted_tokens}, proposed={result.proposed_tokens})")
    return " | ".join(base)


def save_detailed_logs(output_dir: Path,
                       dataset_name: str,
                       args_record: dict[str, Any],
                       baseline: RunResult | None,
                       spec: RunResult | None) -> None:
    """Save richer, machine-readable logs alongside the summary log.

    - Writes per-chunk stats to JSONL files for baseline/spec.
    - Writes profile summaries to JSON files.
    - Writes a combined compare summary JSON for quick analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save baseline chunks (if baseline ran)
    if baseline and baseline.profiling and "chunk_stats" in baseline.profiling:
        with (output_dir / "baseline_chunks.jsonl").open("w", encoding="utf-8") as f:
            for idx, cs in enumerate(baseline.profiling.get("chunk_stats", [])):
                rec = {"mode": "baseline", "dataset": dataset_name, "chunk_idx": idx}
                rec.update(cs)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Save spec chunks
    if spec and spec.profiling and "chunk_stats" in spec.profiling:
        with (output_dir / "spec_chunks.jsonl").open("w", encoding="utf-8") as f:
            for idx, cs in enumerate(spec.profiling.get("chunk_stats", [])):
                rec = {"mode": "spec", "dataset": dataset_name, "chunk_idx": idx}
                rec.update(cs)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Profile summaries
    baseline_prof = baseline.profiling if baseline and baseline.profiling else None
    spec_prof = spec.profiling if spec and spec.profiling else None
    with (output_dir / "baseline_profile.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "args": args_record,
            "result": {
                "throughput": (baseline.throughput if baseline else None),
                "avg_latency": (baseline.avg_latency if baseline else None),
                "prompt_tokens": (baseline.prompt_tokens if baseline else None),
                "generated_tokens": (baseline.generated_tokens if baseline else None),
            },
            "profiling": baseline_prof,
        }, f, ensure_ascii=False)

    if spec:
        with (output_dir / "spec_profile.json").open("w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset_name,
                "args": args_record,
                "result": {
                    "throughput": spec.throughput,
                    "avg_latency": spec.avg_latency,
                    "prompt_tokens": spec.prompt_tokens,
                    "generated_tokens": spec.generated_tokens,
                    "acceptance_rate": spec.acceptance_rate,
                    "accepted": spec.accepted_tokens,
                    "proposed": spec.proposed_tokens,
                },
                "profiling": spec_prof,
            }, f, ensure_ascii=False)

    # Compare summary (only when both baseline and spec are available)
    if spec and baseline:
        compare_payload = {
            "dataset": dataset_name,
            "args": args_record,
            "baseline": {
                "throughput": baseline.throughput,
                "avg_latency": baseline.avg_latency,
                "engine_init_time": (baseline.profiling or {}).get("engine_init_time") if baseline.profiling else None,
                "total_decode_time": (baseline.profiling or {}).get("total_decode_time") if baseline.profiling else None,
            },
            "spec": {
                "throughput": spec.throughput,
                "avg_latency": spec.avg_latency,
                "engine_init_time": (spec.profiling or {}).get("engine_init_time") if spec.profiling else None,
                "total_decode_time": (spec.profiling or {}).get("total_decode_time") if spec.profiling else None,
                "acceptance_rate": spec.acceptance_rate,
                "accepted": spec.accepted_tokens,
                "proposed": spec.proposed_tokens,
            },
            "delta": {
                "throughput": spec.throughput - baseline.throughput,
                "throughput_ratio": (spec.throughput / max(baseline.throughput, 1e-6) - 1.0)
                if baseline.throughput else None,
                "avg_latency": spec.avg_latency - baseline.avg_latency,
            },
        }
        with (output_dir / "compare_summary.json").open("w", encoding="utf-8") as f:
            json.dump(compare_payload, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speculative decoding mini benchmark")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--inference-length", type=int, default=2)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=32,
                        help="单次推理批大小（与 tream inference_batch_size 对齐）。")
    parser.add_argument("--max-concurrent-prompts", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--vqgan-batch", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("./inference_logs"))
    parser.add_argument("--num-spec-tokens", type=int, default=5)
    parser.add_argument("--spec-method", type=str, choices=["ngram", "mqa", "eagle3"], default="ngram",
                        help="推测方法：ngram（图友好）或 mqa/eagle3（需草稿模型）。")
    parser.add_argument("--vocab-mapping-path", type=Path, default=None,
                        help="EAGLE-3 草稿与目标词表映射 (.pt)，可选。")
    parser.add_argument("--prompt-lookup-min", type=int, default=2,
                        help="N-gram: 最小回看窗口。")
    parser.add_argument("--prompt-lookup-max", type=int, default=5,
                        help="N-gram: 最大回看窗口。")
    parser.add_argument("--spec-disable-mqa-scorer", action="store_true",
                        help="禁用 MQA scorer，回退为 batch expansion 评分路径（更易图化）。")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-adapter-dir", type=Path, default=None,
                        help="如果提供，将以该目录构造 LoRARequest 进行运行时 LoRA 应用（仅 inference 侧）。")
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--dtype",
                        type=str,
                        default="auto",
                        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"])

    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Interleave tokenize (VQGAN) and generation like the InferenceActor.")
    parser.add_argument("--sleep-seconds", type=float, default=3.0,
                        help="Sleep between baseline and speculative runs to allow GPU memory to be reclaimed.")
    parser.add_argument("--run-mode", type=str, choices=["baseline", "spec", "both"], default="baseline",
                        help="只跑 baseline、只跑 spec，或都跑（默认 baseline）。")
    parser.add_argument("--encoder-device", type=str, default=None,
                        help="Device for VQGAN encoder (e.g., cuda:1 or cpu). Defaults to engine device when omitted.")
    parser.add_argument("--encoder-dtype", type=str, default="auto",
                        choices=["auto", "fp32", "fp16", "bf16"],
                        help="Computation dtype for VQGAN encoder to reduce memory in streaming mode.")

    args = parser.parse_args()

    engine_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {engine_device}")

    enable_lora = not args.disable_lora

    list_start = time.time()
    frame_paths = list_image_paths(args.dataset_root, max_frames=args.max_frames)
    list_time = time.time() - list_start
    print(f"[INFO] 采样帧数: {len(frame_paths)}")
    image_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    # Decide encoder device and dtype
    # In streaming mode, force encoder to run on the SAME GPU as engine to
    # mirror InferenceActor behavior precisely.
    if args.streaming:
        enc_dev = engine_device
    else:
        enc_dev = engine_device if args.encoder_device is None else torch.device(args.encoder_device)
    enc_dtype_map = {"auto": None, "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    enc_dtype = enc_dtype_map.get(args.encoder_dtype, None)
    encoder = VQGANModel.from_pretrained(RAW_VQGAN_PATH)
    if enc_dtype is not None:
        encoder = encoder.to(dtype=enc_dtype)
    encoder = encoder.to(enc_dev).eval()
    if args.streaming and enc_dev.type == "cuda" and enc_dev == engine_device and args.gpu_mem_util > 0.9:
        print("[WARN] Streaming with encoder on same GPU and gpu_mem_util>0.9 may cause OOM; consider --encoder-device cpu/cuda:X or lower --gpu-mem-util.")

    if not args.streaming:
        encode_start = time.time()
        # Encode frames on the chosen encoder device (enc_dev)
        frame_tokens = encode_frames(frame_paths, encoder, image_transform, enc_dev, args.vqgan_batch)
        encode_time = time.time() - encode_start
        print(f"[INFO] frame_tokens 形状: {tuple(frame_tokens.shape)} (每帧 {ENCODING_SIZE} token)")

        if args.num_prompts % args.inference_batch_size != 0:
            adjusted = (args.num_prompts // args.inference_batch_size) * args.inference_batch_size
            if adjusted <= 0:
                adjusted = args.inference_batch_size
            print(f"[WARN] --num-prompts={args.num_prompts} 不是 --inference-batch-size={args.inference_batch_size} 的整数倍；自动调整为 {adjusted}。")
            args.num_prompts = adjusted

        frame_tokens_list: list[list[int]] = frame_tokens.tolist()
        prompt_build_start = time.time()
        prompt_batches = build_prompts_streaming(frame_tokens_list,
                                                 args.context_length,
                                                 args.inference_batch_size,
                                                 args.num_prompts)
        prompt_build_time = time.time() - prompt_build_start
        total_prompts = sum(len(batch) for batch in prompt_batches)
        print(f"[INFO] 构造批次 {len(prompt_batches)} 个，总提示数 {total_prompts}。"
              f"每批大小 {len(prompt_batches[0]) if prompt_batches else 0}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.inference_length * ENCODING_SIZE,
        logprobs=1,
    )

    dtype_alias_map = {
        "fp16": "half",
        "float16": "half",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "auto": "auto",
    }
    engine_dtype = dtype_alias_map.get(args.dtype.lower(), "auto")

    # Baseline uses CUDA Graphs (V0 default) for fair parity with InferenceActor
    # Set enforce_eager=False here. Speculative run will override to True.
    base_engine_kwargs: dict[str, Any] = dict(
        model=str(args.teacher_model),
        tokenizer=str(args.teacher_model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=engine_dtype,
        enforce_eager=False,
    )
    # Streaming on a single GPU can OOM with high mem util. We no longer
    # override the value, but warn so the caller can choose. For full parity
    # with InferenceActor, pass the same --gpu-mem-util here.
    if args.streaming and base_engine_kwargs["gpu_memory_utilization"] > 0.95:
        print("[WARN] Streaming with gpu_mem_util>0.95 may cause OOM; consider 0.92–0.95 if unstable.")
    if enable_lora:
        base_engine_kwargs.update(
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
            max_loras=args.max_loras,
        )
    base_engine_args = EngineArgs(**base_engine_kwargs)

    max_concurrent = args.max_concurrent_prompts or args.inference_batch_size

    # 构造可选的 LoRARequest（若提供适配器目录）
    lora_req = None
    if enable_lora and args.lora_adapter_dir and args.lora_adapter_dir.exists():
        lora_req = LoRARequest(
            lora_name=f"bench-{int(time.time())}",
            lora_int_id=1,
            lora_local_path=str(args.lora_adapter_dir),
        )

    baseline_result: RunResult | None = None
    if args.run_mode in ("baseline", "both"):
        if not args.streaming:
            baseline_result = run_inference(
                mode="baseline",
                prompt_batches=prompt_batches,
                sampling_params=sampling_params,
                engine_args=base_engine_args,
                chunk_size=max_concurrent,
                lora_request=lora_req,
            )
        else:
            baseline_result = run_inference_streaming(
                mode="baseline",
                frame_paths=frame_paths,
                encoder=encoder,
                image_transform=image_transform,
                encoder_device=enc_dev,
                engine_device=engine_device,
                context_length=args.context_length,
                batch_size=args.inference_batch_size,
                inference_length=args.inference_length,
                vqgan_batch=args.vqgan_batch,
                sampling_params=sampling_params,
                engine_args=base_engine_args,
                lora_request=lora_req,
            )
        print(format_result(baseline_result))

    spec_result: RunResult | None = None
    # 仅在 run-mode 需要时运行推测；eagle3/mqa 需要提供 --draft-model
    run_spec = args.run_mode in ("spec", "both")
    if run_spec:
        if args.spec_method in ("mqa", "eagle3") and args.draft_model is None:
            raise ValueError("spec_method=mqa/eagle3 需要提供 --draft-model")
        # Give CUDA a moment to release memory from baseline engine
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        if args.sleep_seconds and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        spec_metrics_path = args.output_dir / "mini_spec_metrics.jsonl"
        spec_engine_kwargs = dict(base_engine_kwargs)
        if args.spec_method == "ngram":
            spec_engine_kwargs.update(
                speculative_config={
                    "method": "ngram",
                    "num_speculative_tokens": args.num_spec_tokens,
                    "prompt_lookup_min": args.prompt_lookup_min,
                    "prompt_lookup_max": args.prompt_lookup_max,
                },
            )
        else:
            # Draft-model based speculative decoding (e.g., EAGLE-3)
            # vLLM 0.10.1.1 requires the method to be explicitly specified
            # unless the model path contains an eagle* hint. Set it here.
            spec_engine_kwargs.update(
                speculative_config={
                    "method": "eagle3" if args.spec_method == "eagle3" else args.spec_method,
                    "model": str(args.draft_model),
                    "num_speculative_tokens": args.num_spec_tokens,
                    "draft_tensor_parallel_size": 1,
                    # 当禁用 MQA scorer 时，回退到 batch expansion
                    "disable_mqa_scorer": bool(args.spec_disable_mqa_scorer),
                },
            )
            if args.vocab_mapping_path is not None:
                spec_engine_kwargs["speculative_config"]["vocab_mapping_path"] = str(args.vocab_mapping_path)
            if not args.spec_disable_mqa_scorer:
                spec_engine_kwargs["enforce_eager"] = True
        spec_engine_args = EngineArgs(**spec_engine_kwargs)
        if not args.streaming:
            spec_result = run_inference(
                mode="speculative",
                prompt_batches=prompt_batches,
                sampling_params=sampling_params,
                engine_args=spec_engine_args,
                spec_metrics_path=spec_metrics_path,
                chunk_size=max_concurrent,
                lora_request=lora_req,
            )
        else:
            spec_result = run_inference_streaming(
                mode="speculative",
                frame_paths=frame_paths,
                encoder=encoder,
                image_transform=image_transform,
                encoder_device=enc_dev,
                engine_device=engine_device,
                context_length=args.context_length,
                batch_size=args.inference_batch_size,
                inference_length=args.inference_length,
                vqgan_batch=args.vqgan_batch,
                sampling_params=sampling_params,
                engine_args=spec_engine_args,
                spec_metrics_path=spec_metrics_path,
                lora_request=lora_req,
            )
        print(format_result(spec_result))

    log_path = getattr(args, "log_file", None) or (args.output_dir / "mini_spec_benchmark.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().isoformat(timespec="seconds")

    args_record = {}
    for key, value in vars(args).items():
        args_record[key] = str(value) if isinstance(value, Path) else value
    params_json = json.dumps(args_record, ensure_ascii=False, sort_keys=True)
    
    # 添加数据集名称标记
    dataset_name = args.dataset_root.name
    
    profile_payload = {
        "dataset_name": dataset_name,
        "frame_count": len(frame_paths),
        "list_time": list_time,
        "encode_time": (encode_time if not args.streaming else None),
        "prompt_build_time": (prompt_build_time if not args.streaming else None),
        "baseline": (baseline_result.profiling if baseline_result is not None else None),
        "speculative": (spec_result.profiling if spec_result is not None else None),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] dataset={dataset_name} params={params_json}\n")
        f.write(f"[{timestamp}] dataset={dataset_name} profile={json.dumps(profile_payload, ensure_ascii=False)}\n")
        if baseline_result is not None:
            f.write(f"[{timestamp}] dataset={dataset_name} {format_result(baseline_result)}\n")
        if spec_result is not None:
            f.write(f"[{timestamp}] dataset={dataset_name} {format_result(spec_result)}\n")
    # Save machine-readable detailed logs
    save_detailed_logs(args.output_dir, dataset_name, args_record, baseline_result, spec_result)


if __name__ == "__main__":
    main()
