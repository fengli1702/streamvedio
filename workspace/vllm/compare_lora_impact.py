#!/usr/bin/env python3
"""
对比脚本：LoRA 开关对 Speculative Decoding 接受率的影响

这个脚本运行两个实验：
1. Base: Spec Decoding 无 LoRA
2. LoRA-Infra: Spec Decoding 有 LoRA 基础设施 (但无 adapter)

然后对比两个实验的接受率、吞吐量等关键指标
"""

import os
import sys
import json
import tempfile
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, '/workspace')

def run_experiment(
    phase: str,
    model_id: str = "meta-llama/Llama-2-7b-hf",
    num_prompts: int = 10,
    max_tokens: int = 32,
    enable_lora: bool = False,
) -> Dict[str, Any]:
    """运行一个实验阶段并收集日志"""
    
    log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
    
    print(f"\n{'='*70}")
    print(f"🔄 运行实验: {phase.upper()}")
    print(f"{'='*70}")
    print(f"配置:")
    print(f"  - 模型: {model_id}")
    print(f"  - Spec: 启用")
    print(f"  - LoRA: {'启用' if enable_lora else '禁用'}")
    print(f"  - 提示数: {num_prompts}")
    print(f"  - 日志文件: {log_file}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['VLLM_SPEC_LORA_DEBUG'] = '1'
    env['VLLM_SPEC_LORA_DEBUG_PATH'] = log_file
    env['VLLM_SPEC_LORA_PHASE'] = phase
    
    # 构造命令
    cmd = [
        sys.executable,
        '/workspace/scripts/test_spec_lora_debug.py',
        '--model-id', model_id,
        '--phase', phase,
        '--num-prompts', str(num_prompts),
        '--max-tokens', str(max_tokens),
        '--enable-speculative',
        '--output-dir', '/tmp',
    ]
    
    if enable_lora:
        cmd.append('--enable-lora')
    
    print(f"\n命令: {' '.join(cmd)}\n")
    
    # 运行实验
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
            return None
        
        # 读取日志文件
        if not os.path.exists(log_file):
            print(f"❌ 日志文件未生成: {log_file}")
            return None
        
        with open(log_file, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]
        
        if not records:
            print(f"❌ 日志文件为空")
            return None
        
        print(f"✅ 收集了 {len(records)} 条日志")
        
        return {
            'phase': phase,
            'enable_lora': enable_lora,
            'log_file': log_file,
            'records': records,
            'num_records': len(records),
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ 实验超时")
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_acceptance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析接受率数据"""
    
    accept_records = [r for r in records if r.get('type') == 'eagle_accept']
    
    if not accept_records:
        return None
    
    total_proposed = sum(r.get('num_proposed_tokens', 0) for r in accept_records)
    total_accepted = sum(r.get('num_accepted_tokens', 0) for r in accept_records)
    
    accept_ratios = [r.get('accept_ratio', 0) for r in accept_records]
    
    return {
        'num_steps': len(accept_records),
        'total_proposed': total_proposed,
        'total_accepted': total_accepted,
        'overall_ratio': total_accepted / total_proposed if total_proposed > 0 else 0,
        'avg_ratio': sum(accept_ratios) / len(accept_ratios) if accept_ratios else 0,
        'min_ratio': min(accept_ratios) if accept_ratios else 0,
        'max_ratio': max(accept_ratios) if accept_ratios else 0,
        'step_data': [(r.get('step'), r.get('accept_ratio')) for r in accept_records],
    }


def analyze_runtime(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析运行时性能数据"""
    
    runtime_records = [r for r in records if r.get('type') == 'runtime_step']
    
    if not runtime_records:
        return None
    
    throughputs = [r.get('throughput', 0) for r in runtime_records]
    dts = [r.get('dt', 0) for r in runtime_records]
    num_tokens_list = [r.get('num_tokens', 0) for r in runtime_records]
    
    return {
        'num_steps': len(runtime_records),
        'total_tokens': sum(num_tokens_list),
        'total_time': sum(dts),
        'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
        'min_throughput': min(throughputs) if throughputs else 0,
        'max_throughput': max(throughputs) if throughputs else 0,
        'step_data': [(r.get('step'), r.get('throughput')) for r in runtime_records],
    }


def analyze_lora_state(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析 LoRA 状态"""
    
    lora_records = [r for r in records if r.get('type') == 'lora_state']
    
    if not lora_records:
        return None
    
    num_loras_list = [r.get('num_active_loras', 0) for r in lora_records]
    
    return {
        'num_steps': len(lora_records),
        'avg_active_loras': sum(num_loras_list) / len(num_loras_list) if num_loras_list else 0,
        'max_active_loras': max(num_loras_list) if num_loras_list else 0,
        'step_data': [(r.get('step'), r.get('num_active_loras')) for r in lora_records],
    }


def compare_experiments(base_exp: Dict, lora_exp: Dict) -> Dict[str, Any]:
    """对比两个实验"""
    
    print(f"\n{'='*70}")
    print("📊 对比分析结果")
    print(f"{'='*70}")
    
    base_records = base_exp['records']
    lora_records = lora_exp['records']
    
    # 分析接受率
    base_accept = analyze_acceptance(base_records)
    lora_accept = analyze_acceptance(lora_records)
    
    print(f"\n🎯 接受率对比:")
    print(f"  Base 阶段:")
    if base_accept:
        print(f"    - 总步数: {base_accept['num_steps']}")
        print(f"    - 总提议: {base_accept['total_proposed']}")
        print(f"    - 总接受: {base_accept['total_accepted']}")
        print(f"    - 整体接受率: {base_accept['overall_ratio']:.2%}")
        print(f"    - 平均接受率: {base_accept['avg_ratio']:.2%}")
        print(f"    - 范围: {base_accept['min_ratio']:.2%} ~ {base_accept['max_ratio']:.2%}")
    else:
        print(f"    (无数据)")
    
    print(f"\n  LoRA-Infra 阶段:")
    if lora_accept:
        print(f"    - 总步数: {lora_accept['num_steps']}")
        print(f"    - 总提议: {lora_accept['total_proposed']}")
        print(f"    - 总接受: {lora_accept['total_accepted']}")
        print(f"    - 整体接受率: {lora_accept['overall_ratio']:.2%}")
        print(f"    - 平均接受率: {lora_accept['avg_ratio']:.2%}")
        print(f"    - 范围: {lora_accept['min_ratio']:.2%} ~ {lora_accept['max_ratio']:.2%}")
    else:
        print(f"    (无数据)")
    
    # 计算差异
    if base_accept and lora_accept:
        accept_diff = lora_accept['overall_ratio'] - base_accept['overall_ratio']
        accept_diff_pct = (accept_diff / base_accept['overall_ratio'] * 100) if base_accept['overall_ratio'] > 0 else 0
        
        print(f"\n  📈 差异:")
        print(f"    - 绝对差异: {accept_diff:+.2%}")
        print(f"    - 相对变化: {accept_diff_pct:+.2f}%")
        
        if accept_diff < 0:
            print(f"    ⚠️  LoRA 导致接受率下降 {-accept_diff_pct:.2f}%")
        elif accept_diff > 0:
            print(f"    ✅ LoRA 反而提升接受率 {accept_diff_pct:.2f}%")
        else:
            print(f"    ✅ LoRA 对接受率无明显影响")
    
    # 分析吞吐量
    base_runtime = analyze_runtime(base_records)
    lora_runtime = analyze_runtime(lora_records)
    
    print(f"\n⚡ 吞吐量对比:")
    print(f"  Base 阶段:")
    if base_runtime:
        print(f"    - 总 tokens: {base_runtime['total_tokens']}")
        print(f"    - 总时间: {base_runtime['total_time']:.3f}s")
        print(f"    - 平均吞吐: {base_runtime['avg_throughput']:.1f} tok/s")
        print(f"    - 范围: {base_runtime['min_throughput']:.1f} ~ {base_runtime['max_throughput']:.1f} tok/s")
    else:
        print(f"    (无数据)")
    
    print(f"\n  LoRA-Infra 阶段:")
    if lora_runtime:
        print(f"    - 总 tokens: {lora_runtime['total_tokens']}")
        print(f"    - 总时间: {lora_runtime['total_time']:.3f}s")
        print(f"    - 平均吞吐: {lora_runtime['avg_throughput']:.1f} tok/s")
        print(f"    - 范围: {lora_runtime['min_throughput']:.1f} ~ {lora_runtime['max_throughput']:.1f} tok/s")
    else:
        print(f"    (无数据)")
    
    # 计算吞吐量差异
    if base_runtime and lora_runtime:
        throughput_diff = lora_runtime['avg_throughput'] - base_runtime['avg_throughput']
        throughput_diff_pct = (throughput_diff / base_runtime['avg_throughput'] * 100) if base_runtime['avg_throughput'] > 0 else 0
        
        print(f"\n  📈 差异:")
        print(f"    - 绝对差异: {throughput_diff:+.1f} tok/s")
        print(f"    - 相对变化: {throughput_diff_pct:+.2f}%")
        
        if throughput_diff < 0:
            print(f"    ⚠️  LoRA 导致吞吐量下降 {-throughput_diff_pct:.2f}%")
        elif throughput_diff > 0:
            print(f"    ✅ LoRA 反而提升吞吐量 {throughput_diff_pct:.2f}%")
        else:
            print(f"    ✅ LoRA 对吞吐量无明显影响")
    
    # LoRA 状态分析
    lora_state = analyze_lora_state(lora_records)
    if lora_state:
        print(f"\n🔧 LoRA 基础设施开销:")
        print(f"    - 平均激活 LoRA 数: {lora_state['avg_active_loras']:.1f}")
        print(f"    - 最大激活 LoRA 数: {lora_state['max_active_loras']}")
    
    # 总结
    print(f"\n{'='*70}")
    print("📋 总结")
    print(f"{'='*70}")
    
    if base_accept and lora_accept:
        if accept_diff < -0.05:
            verdict = "❌ LoRA 基础设施对接受率有明显负面影响 (> 5%)"
        elif accept_diff < 0:
            verdict = "⚠️  LoRA 基础设施对接受率有轻微负面影响 (< 5%)"
        elif accept_diff > 0.05:
            verdict = "✅ LoRA 基础设施对接受率有正面影响"
        else:
            verdict = "✅ LoRA 基础设施对接受率影响不大"
        
        print(f"结论: {verdict}")
    
    return {
        'base_accept': base_accept,
        'lora_accept': lora_accept,
        'base_runtime': base_runtime,
        'lora_runtime': lora_runtime,
        'lora_state': lora_state,
    }


def main():
    parser = argparse.ArgumentParser(
        description="对比 LoRA 开关对 Speculative Decoding 的影响"
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='模型 ID'
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=10,
        help='提示数量'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32,
        help='最大生成 token 数'
    )
    parser.add_argument(
        '--skip-base',
        action='store_true',
        help='跳过 Base 实验'
    )
    parser.add_argument(
        '--skip-lora',
        action='store_true',
        help='跳过 LoRA-Infra 实验'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("🚀 LoRA 影响对比分析")
    print(f"{'='*70}")
    print(f"配置:")
    print(f"  - 模型: {args.model_id}")
    print(f"  - 提示数: {args.num_prompts}")
    print(f"  - 最大 tokens: {args.max_tokens}")
    
    base_exp = None
    lora_exp = None
    
    # 运行 Base 实验
    if not args.skip_base:
        base_exp = run_experiment(
            phase='base',
            model_id=args.model_id,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            enable_lora=False,
        )
    
    # 运行 LoRA-Infra 实验
    if not args.skip_lora:
        lora_exp = run_experiment(
            phase='lora_infra',
            model_id=args.model_id,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            enable_lora=True,
        )
    
    # 对比结果
    if base_exp and lora_exp:
        compare_experiments(base_exp, lora_exp)
    else:
        print(f"\n❌ 无法进行对比（需要两个实验都成功）")
        if not base_exp:
            print(f"   - Base 实验失败")
        if not lora_exp:
            print(f"   - LoRA-Infra 实验失败")


if __name__ == '__main__':
    main()
