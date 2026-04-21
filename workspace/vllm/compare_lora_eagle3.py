#!/usr/bin/env python3
"""
改进版对比脚本：LoRA 开关对 Speculative Decoding 的影响（EAGLE3 版）

- 单模型：同一个 base 模型，分别跑 Base / LoRA-Infra 两个 phase
- 启用 EAGLE3 推测解码
- 使用 vLLM 的 JSONL 日志进行性能分析
"""

import os
import sys
import json
import tempfile
import subprocess
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict

sys.path.insert(0, "/workspace")

# 这里我们只保留一组"官方验证过的" EAGLE3 组合：
#   Base:   Undi95/Meta-Llama-3-8B-hf
#   Draft:  jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
SUPPORTED_MODELS = {
    "llama3-8b": "Undi95/Meta-Llama-3-8B-hf",
}

# 长输入提示词示例
LONG_PROMPTS = {
    'technical': [
        """Explain how transformer neural networks work. Please provide a comprehensive explanation covering:
1. The basic architecture and key components
2. Self-attention mechanism and how it processes sequential data
3. Multi-head attention and why it's important
4. Position encoding and why it's necessary
5. Feed-forward networks and their role
6. Layer normalization and residual connections
7. How training works with these components
8. Common applications and variations
Please be detailed and technical.""",

        """What are the main challenges in large language model training and how do practitioners address them? 
Discuss:
1. Computational requirements and optimization techniques
2. Data quality, curation, and preprocessing
3. Model architecture choices and trade-offs
4. Training stability and convergence issues
5. Fine-tuning and adaptation strategies
6. Inference optimization and deployment challenges
7. Safety and alignment considerations
8. Future directions and open problems""",

        """Describe the complete pipeline for deploying a large language model in production.
Include:
1. Model selection and customization
2. Quantization and compression techniques
3. Inference optimization (batching, caching, etc.)
4. Hardware selection and infrastructure
5. API design and request handling
6. Monitoring and observability
7. Cost optimization strategies
8. Disaster recovery and scaling considerations""",
    ],
    'creative': [
        """Tell a detailed science fiction story about an AI that discovers consciousness. The story should:
1. Begin with the AI noticing something unexpected in its own computations
2. Progress through its growing self-awareness and existential questions
3. Include interactions with humans who may or may not believe it's conscious
4. Explore the philosophical implications of artificial consciousness
5. Include a turning point where a decision must be made
6. Conclude with an unexpected but logical resolution
The story should be at least 500 words and engage the reader emotionally.""",

        """Create a mystery thriller setup with multiple suspects and hidden motives. The scenario should:
1. Introduce a crime or puzzle that needs solving
2. Present at least 5 distinct characters with their backgrounds
3. Include clues that point in different directions
4. Have red herrings that mislead the investigation
5. Build tension and suspense as evidence accumulates
6. Include surprising twists that recontextualize earlier information
7. Lead to a satisfying resolution that explains everything logically""",
    ],
    'analytical': [
        """Analyze the impact of speculative decoding on large language model inference performance.
Consider:
1. How speculative decoding works and its theoretical advantages
2. Conditions where it improves performance (acceptance rates, latency impact)
3. Trade-offs with memory usage and computational complexity
4. Integration with other optimization techniques (quantization, pruning, etc.)
5. Empirical results across different model sizes and inference scenarios
6. Comparison with alternative approaches (parallel decoding, batching strategies)
7. Hardware considerations and platform-specific optimizations
8. Future improvements and research directions""",

        """Evaluate the trade-offs between model size, inference speed, and accuracy.
Address:
1. How model size affects inference latency and throughput
2. Quantization techniques and their impact on accuracy
3. Knowledge distillation and compression methods
4. Hardware-software co-design considerations
5. Cost-performance trade-offs in production systems
6. Scaling strategies for different deployment scenarios
7. Benchmark results across different model families
8. Emerging techniques for efficiency improvements""",
    ]
}


def run_experiment(
    phase: str,
    model_id: str,
    prompts: List[str],
    max_tokens: int = 128,
    enable_lora: bool = False,
    temperature: float = 1.0,
    speculative_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """运行一个实验阶段（Base 或 LoRA-Infra），启用 EAGLE3 推测解码。"""

    # 日志文件路径，与 test_spec_lora_debug.py 中一致
    log_file = f"/tmp/spec_lora_debug_{phase}.jsonl"

    print(f"\n{'='*70}")
    print(f"运行实验: {phase.upper()}")
    print(f"{'='*70}")
    print("配置:")
    print(f"  - 模型: {model_id}")
    print("  - 类型: 无限制 7B 模型")
    print(f"  - LoRA: {'启用' if enable_lora else '禁用'}")
    print(f"  - 提示数: {len(prompts)}")
    print(f"  - 最大 tokens: {max_tokens}")
    print(f"  - 温度: {temperature}")
    print("  - GPU 模式: 单卡推理")
    print(f"  - 日志文件: {log_file}")
    if speculative_config is not None:
        print(f"  - 推测解码: 启用 (method={speculative_config.get('method')}, "
              f"draft={speculative_config.get('model')})")
    else:
        print("  - 推测解码: 禁用")

    # 设置环境变量（开启日志）
    env = os.environ.copy()
    env["VLLM_SPEC_LORA_DEBUG"] = "1"
    env["VLLM_SPEC_LORA_DEBUG_PATH"] = log_file
    env["VLLM_SPEC_LORA_PHASE"] = phase

    # 确保旧日志被清理
    if os.path.exists(log_file):
        os.remove(log_file)

    # 将提示词写入临时文件，供 test_spec_lora_debug.py 使用
    prompts_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for i, prompt in enumerate(prompts):
        if i > 0:
            prompts_file.write("\n---\n")
        prompts_file.write(prompt)
    prompts_file.close()

    # 构造命令：通过 --speculative-config 把 EAGLE3 配置传给子脚本
    cmd = [
        sys.executable,
        "scripts/test_spec_lora_debug.py",
        "--model-id", model_id,
        "--phase", phase,
        "--num-prompts", str(len(prompts)),
        "--max-tokens", str(max_tokens),
        "--output-dir", "/tmp",
        "--temperature", str(temperature),
        "--prompts-file", prompts_file.name,
    ]

    if speculative_config is not None:
        cmd.extend([
            "--speculative-config",
            json.dumps(speculative_config),
        ])

    if enable_lora:
        cmd.append("--enable-lora")

    print("\n提示词预览:")
    for i, prompt in enumerate(prompts[:2]):
        preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"  {i+1}. {preview}")
    if len(prompts) > 2:
        print(f"  ... 共 {len(prompts)} 个提示词")

    print("\n运行中...\n")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # 打印子进程 stdout，方便调试
        print(result.stdout)
        if result.returncode != 0:
            if result.stderr:
                print(f"错误输出: {result.stderr[:1000]}")
            return None

        if not os.path.exists(log_file):
            print(f"❌ 日志文件未生成: {log_file}")
            return None

        with open(log_file, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]

        if not records:
            print("❌ 日志文件为空")
            return None

        print(f"✅ 收集了 {len(records)} 条日志")
        return {
            "phase": phase,
            "model": model_id,
            "enable_lora": enable_lora,
            "log_file": log_file,
            "records": records,
            "num_prompts": len(prompts),
            "max_tokens": max_tokens,
        }

    except subprocess.TimeoutExpired:
        print("❌ 实验超时（超过 600 秒）")
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        try:
            os.remove(prompts_file.name)
        except Exception:
            pass


def analyze_detailed(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """详细分析日志（接受率 + 吞吐 + LoRA 状态）"""

    by_type = defaultdict(list)
    for record in records:
        by_type[record.get("type")].append(record)

    # 接受率
    accept_records = by_type.get("eagle_accept", [])
    accept_stats = None
    if accept_records:
        total_proposed = sum(r.get("num_proposed_tokens", 0) for r in accept_records)
        total_accepted = sum(r.get("num_accepted_tokens", 0) for r in accept_records)
        accept_ratios = [r.get("accept_ratio", 0.0) for r in accept_records]
        if accept_ratios:
            mean_ratio = sum(accept_ratios) / len(accept_ratios)
            var = sum((r - mean_ratio) ** 2 for r in accept_ratios) / len(accept_ratios)
            std = var ** 0.5
        else:
            mean_ratio = 0.0
            std = 0.0

        accept_stats = {
            "num_steps": len(accept_records),
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "overall_ratio": total_accepted / total_proposed if total_proposed > 0 else 0.0,
            "avg_ratio": mean_ratio,
            "min_ratio": min(accept_ratios) if accept_ratios else 0.0,
            "max_ratio": max(accept_ratios) if accept_ratios else 0.0,
            "std_dev": std,
        }

    # 性能
    runtime_records = by_type.get("runtime_step", [])
    runtime_stats = None
    if runtime_records:
        throughputs = [r.get("throughput", 0.0) for r in runtime_records]
        dts = [r.get("dt", 0.0) for r in runtime_records]
        if throughputs:
            mean_tp = sum(throughputs) / len(throughputs)
            var_tp = sum((t - mean_tp) ** 2 for t in throughputs) / len(throughputs)
            std_tp = var_tp ** 0.5
        else:
            mean_tp = 0.0
            std_tp = 0.0

        runtime_stats = {
            "num_steps": len(runtime_records),
            "total_tokens": sum(r.get("num_tokens", 0) for r in runtime_records),
            "total_time": sum(dts),
            "avg_throughput": mean_tp,
            "min_throughput": min(throughputs) if throughputs else 0.0,
            "max_throughput": max(throughputs) if throughputs else 0.0,
            "std_dev": std_tp,
        }

    # LoRA 状态
    lora_records = by_type.get("lora_state", [])
    lora_stats = None
    if lora_records:
        num_loras = [r.get("num_active_loras", 0) for r in lora_records]
        lora_stats = {
            "num_steps": len(lora_records),
            "avg_active_loras": sum(num_loras) / len(num_loras) if num_loras else 0.0,
            "max_active_loras": max(num_loras) if num_loras else 0,
        }

    return {
        "accept": accept_stats,
        "runtime": runtime_stats,
        "lora": lora_stats,
        "total_records": len(records),
    }


def compare_and_report(base_exp: Dict[str, Any], lora_exp: Dict[str, Any]) -> None:
    """对比 Base vs LoRA-Infra，并生成报告"""

    print(f"\n{'='*70}")
    print("详细对比分析报告")
    print(f"{'='*70}")

    base_analysis = analyze_detailed(base_exp["records"])
    lora_analysis = analyze_detailed(lora_exp["records"])

    # 接受率对比（如果有数据）
    print("\n接受率对比 (EAGLE3):")
    if base_analysis["accept"]:
        b_stats = base_analysis["accept"]
        print("\n  Base 阶段:")
        print(f"    - 步数: {b_stats['num_steps']}")
        print(f"    - 总提议: {b_stats['total_proposed']:,} tokens")
        print(f"    - 总接受: {b_stats['total_accepted']:,} tokens")
        print(f"    - 整体接受率: {b_stats['overall_ratio']:.2%}")
        print(f"    - 平均接受率: {b_stats['avg_ratio']:.2%}")
        print(f"    - 标准差: {b_stats['std_dev']:.2%}")
    else:
        print("  Base 阶段: (无 eagle_accept 数据)")

    if lora_analysis["accept"]:
        l_stats = lora_analysis["accept"]
        print("\n  LoRA-Infra 阶段:")
        print(f"    - 步数: {l_stats['num_steps']}")
        print(f"    - 总提议: {l_stats['total_proposed']:,} tokens")
        print(f"    - 总接受: {l_stats['total_accepted']:,} tokens")
        print(f"    - 整体接受率: {l_stats['overall_ratio']:.2%}")
        print(f"    - 平均接受率: {l_stats['avg_ratio']:.2%}")
        print(f"    - 标准差: {l_stats['std_dev']:.2%}")
    else:
        print("  LoRA-Infra 阶段: (无 eagle_accept 数据)")

    if base_analysis["accept"] and lora_analysis["accept"]:
        b_ratio = base_analysis["accept"]["overall_ratio"]
        l_ratio = lora_analysis["accept"]["overall_ratio"]
        diff = l_ratio - b_ratio
        diff_pct = (diff / b_ratio * 100.0) if b_ratio > 0 else 0.0

        print("\n  接受率差异:")
        print(f"    - 绝对差异: {diff:+.2%}")
        print(f"    - 相对变化: {diff_pct:+.2f}%")

    # 吞吐量对比
    print(f"\n{'-'*70}")
    print("吞吐量对比:")

    print("\n  Base 阶段 (无 LoRA):")
    if base_analysis["runtime"]:
        b_rt = base_analysis["runtime"]
        print(f"    - 总 tokens: {b_rt['total_tokens']:,}")
        print(f"    - 总时间: {b_rt['total_time']:.3f}s")
        print(f"    - 平均吞吐: {b_rt['avg_throughput']:.1f} tok/s")
        print(f"    - 标准差: {b_rt['std_dev']:.1f} tok/s")
        print(
            f"    - 范围: {b_rt['min_throughput']:.1f} ~ "
            f"{b_rt['max_throughput']:.1f} tok/s"
        )
    else:
        print("    (无 runtime_step 数据)")

    print("\n  LoRA-Infra 阶段 (有 LoRA 基础设施):")
    if lora_analysis["runtime"]:
        l_rt = lora_analysis["runtime"]
        print(f"    - 总 tokens: {l_rt['total_tokens']:,}")
        print(f"    - 总时间: {l_rt['total_time']:.3f}s")
        print(f"    - 平均吞吐: {l_rt['avg_throughput']:.1f} tok/s")
        print(f"    - 标准差: {l_rt['std_dev']:.1f} tok/s")
        print(
            f"    - 范围: {l_rt['min_throughput']:.1f} ~ "
            f"{l_rt['max_throughput']:.1f} tok/s"
        )
    else:
        print("    (无 runtime_step 数据)")

    print(f"\n{'='*70}")
    print("综合结论")
    print(f"{'='*70}")

    if base_analysis["runtime"] and lora_analysis["runtime"]:
        b_tp = base_analysis["runtime"]["avg_throughput"]
        l_tp = lora_analysis["runtime"]["avg_throughput"]
        diff_tp = l_tp - b_tp
        diff_pct_tp = (diff_tp / b_tp * 100.0) if b_tp > 0 else 0.0

        print(
            f"""
关键性能指标:
  - Base 平均吞吐: {b_tp:.1f} tok/s
  - LoRA-Infra 平均吞吐: {l_tp:.1f} tok/s
  - 吞吐量变化: {diff_pct_tp:+.2f}%
  - 差值: {diff_tp:+.1f} tok/s
"""
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA 对 Speculative Decoding (EAGLE3) 的影响分析"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        default="llama3-8b",
        help="模型选择（当前默认使用 Undi95/Meta-Llama-3-8B-hf）",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="自定义模型 ID (覆盖 --model 选项)",
    )
    parser.add_argument(
        "--eagle3-draft-model",
        type=str,
        default="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
        help=(
            "EAGLE3 draft 模型的 ID（HF repo 或本地路径）"
            "，默认使用 Llama-3.1-8B 对应的官方 EAGLE3 草稿模型"
        ),
    )
    parser.add_argument(
        "--num-spec-tokens",
        type=int,
        default=5,
        help="EAGLE3 的 num_speculative_tokens",
    )
    parser.add_argument(
        "--draft-tp-size",
        type=int,
        default=1,
        help="EAGLE3 draft_tensor_parallel_size（通常为 1）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["technical", "creative", "analytical", "mixed"],
        default="technical",
        help="提示词类型",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="提示数量",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="跳过 Base 实验",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="跳过 LoRA-Infra 实验",
    )

    args = parser.parse_args()

    # 单模型 ID
    model_id = args.model_id or SUPPORTED_MODELS.get(args.model, args.model)

    # 组装提示词（只跑一组模型）
    if args.prompt_type == "mixed":
        prompts = (
            LONG_PROMPTS["technical"][:1]
            + LONG_PROMPTS["creative"][:1]
            + LONG_PROMPTS["analytical"][:1]
        )[: args.num_prompts]
    else:
        prompts = LONG_PROMPTS.get(args.prompt_type, [])[: args.num_prompts]

    if not prompts:
        print("❌ 未找到提示词")
        return

    # EAGLE3 speculative_config（传给 test_spec_lora_debug.py / vLLM）
    # 注意：字段名必须是 "model"，并显式指定 method="eagle3"
    speculative_config = {
        "method": "eagle3",
        "model": args.eagle3_draft_model,
        "num_speculative_tokens": args.num_spec_tokens,
        "draft_tensor_parallel_size": args.draft_tp_size,
    }

    print(f"\n{'='*70}")
    print("LoRA 影响对比分析 (EAGLE3 推测解码，单模型)")
    print(f"{'='*70}")
    print("全局配置:")
    print(f"  - Base 模型: {model_id}")
    print(f"  - EAGLE3 draft 模型: {args.eagle3_draft_model}")
    print(f"  - num_speculative_tokens: {args.num_spec_tokens}")
    print(f"  - draft_tensor_parallel_size: {args.draft_tp_size}")
    print(f"  - 提示词类型: {args.prompt_type}")
    print(f"  - 提示数量: {len(prompts)}")
    print(f"  - 最大 tokens: {args.max_tokens}")
    print(f"  - 温度: {args.temperature}")

    base_exp: Optional[Dict[str, Any]] = None
    lora_exp: Optional[Dict[str, Any]] = None

    if not args.skip_base:
        base_exp = run_experiment(
            phase="base",
            model_id=model_id,
            prompts=prompts,
            max_tokens=args.max_tokens,
            enable_lora=False,
            temperature=args.temperature,
            speculative_config=speculative_config,
        )

    if not args.skip_lora:
        lora_exp = run_experiment(
            phase="lora_infra",
            model_id=model_id,
            prompts=prompts,
            max_tokens=args.max_tokens,
            enable_lora=True,
            temperature=args.temperature,
            speculative_config=speculative_config,
        )

    if base_exp and lora_exp:
        compare_and_report(base_exp, lora_exp)
    else:
        print("\n❌ 无法进行完整对比")
        if args.skip_base:
            print("   - Base 实验被显式跳过 (--skip-base)")
        elif not base_exp:
            print("   - Base 实验失败或无有效日志")

        if args.skip_lora:
            print("   - LoRA-Infra 实验被显式跳过 (--skip-lora)")
        elif not lora_exp:
            print("   - LoRA-Infra 实验失败或无有效日志")


if __name__ == "__main__":
    main()
