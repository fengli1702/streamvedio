#!/usr/bin/env python3
"""
演示脚本：模拟数据展示 LoRA 对接受率的影响分析

这个脚本使用虚拟数据演示对比分析的输出格式，
实际数据应该从真实推理中的日志文件获取
"""

def demo_analysis():
    """演示对比分析的输出"""
    
    # 模拟数据：Base vs LoRA-Infra
    base_data = {
        'phase': 'base',
        'total_proposed': 1000,
        'total_accepted': 752,  # 75.2% 接受率
        'avg_throughput': 2560.0,  # tok/s
        'steps': 10,
    }
    
    lora_data = {
        'phase': 'lora_infra',
        'total_proposed': 1000,
        'total_accepted': 720,  # 72.0% 接受率 (-3.2%)
        'avg_throughput': 2400.0,  # tok/s (-6.25%)
        'steps': 10,
    }
    
    print("\n" + "="*70)
    print("🚀 LoRA 影响对比分析 - 演示")
    print("="*70)
    
    print("\n📊 基础数据:")
    print(f"  Base 阶段 (无 LoRA):")
    print(f"    - 总提议 token: {base_data['total_proposed']}")
    print(f"    - 总接受 token: {base_data['total_accepted']}")
    print(f"    - 接受率: {base_data['total_accepted']/base_data['total_proposed']:.2%}")
    print(f"    - 平均吞吐: {base_data['avg_throughput']:.1f} tok/s")
    print(f"    - 测试步数: {base_data['steps']}")
    
    print(f"\n  LoRA-Infra 阶段 (启用 LoRA 基础设施):")
    print(f"    - 总提议 token: {lora_data['total_proposed']}")
    print(f"    - 总接受 token: {lora_data['total_accepted']}")
    print(f"    - 接受率: {lora_data['total_accepted']/lora_data['total_proposed']:.2%}")
    print(f"    - 平均吞吐: {lora_data['avg_throughput']:.1f} tok/s")
    print(f"    - 测试步数: {lora_data['steps']}")
    
    # 计算差异
    base_accept_ratio = base_data['total_accepted'] / base_data['total_proposed']
    lora_accept_ratio = lora_data['total_accepted'] / lora_data['total_proposed']
    accept_diff = lora_accept_ratio - base_accept_ratio
    accept_diff_pct = (accept_diff / base_accept_ratio * 100) if base_accept_ratio > 0 else 0
    
    throughput_diff = lora_data['avg_throughput'] - base_data['avg_throughput']
    throughput_diff_pct = (throughput_diff / base_data['avg_throughput'] * 100) if base_data['avg_throughput'] > 0 else 0
    
    print(f"\n{'='*70}")
    print("📈 对比分析结果")
    print(f"{'='*70}")
    
    print(f"\n🎯 接受率对比:")
    print(f"  Base:        {base_accept_ratio:.2%}")
    print(f"  LoRA-Infra:  {lora_accept_ratio:.2%}")
    print(f"  差异:        {accept_diff:+.2%} ({accept_diff_pct:+.2f}%)")
    
    if accept_diff < -0.05:
        print(f"  ⚠️  结论: LoRA 基础设施导致接受率明显下降 ({-accept_diff_pct:.2f}%)")
    elif accept_diff < -0.01:
        print(f"  ⚠️  结论: LoRA 基础设施导致接受率轻微下降 ({-accept_diff_pct:.2f}%)")
    elif accept_diff > 0.05:
        print(f"  ✅ 结论: LoRA 基础设施提升接受率 ({accept_diff_pct:.2f}%)")
    else:
        print(f"  ✅ 结论: LoRA 基础设施对接受率影响不大 ({accept_diff_pct:+.2f}%)")
    
    print(f"\n⚡ 吞吐量对比:")
    print(f"  Base:        {base_data['avg_throughput']:.1f} tok/s")
    print(f"  LoRA-Infra:  {lora_data['avg_throughput']:.1f} tok/s")
    print(f"  差异:        {throughput_diff:+.1f} tok/s ({throughput_diff_pct:+.2f}%)")
    
    if throughput_diff < -100:
        print(f"  ⚠️  结论: LoRA 基础设施导致吞吐量明显下降 ({-throughput_diff_pct:.2f}%)")
    elif throughput_diff < 0:
        print(f"  ⚠️  结论: LoRA 基础设施导致吞吐量轻微下降 ({-throughput_diff_pct:.2f}%)")
    elif throughput_diff > 100:
        print(f"  ✅ 结论: LoRA 基础设施提升吞吐量 ({throughput_diff_pct:.2f}%)")
    else:
        print(f"  ✅ 结论: LoRA 基础设施对吞吐量影响不大 ({throughput_diff_pct:+.2f}%)")
    
    print(f"\n{'='*70}")
    print("📋 总体评估")
    print(f"{'='*70}")
    
    print(f"""
根据对比分析结果：

1️⃣  接受率变化: {accept_diff:+.2%}
   - 这表示启用 LoRA 基础设施时，speculative decoding 的接受率
   - 每 100 个提议的 token 中少接受了 ~3 个
   - 原因可能是：
     * LoRA 权重计算引入的延迟
     * 额外的内存访问开销
     * 模型输出分布的变化

2️⃣  吞吐量变化: {throughput_diff_pct:+.2f}%
   - 平均吞吐量下降约 6.25%
   - 这与接受率下降一致
   - 说明 LoRA 基础设施的确有性能开销

3️⃣  权衡分析:
   - 如果不需要 LoRA 功能，建议禁用以获得最佳性能
   - 如果需要 LoRA 功能，~6% 的开销是可以接受的
   - 可以通过优化 LoRA 计算路径来进一步改善性能

建议：
  ✅ 对于生产环境推理，比较接受率差异是否在容忍范围内
  ✅ 对于需要 LoRA 的场景，这个开销是合理的
  ✅ 未来可以考虑 LoRA 感知的 speculative decoding 优化
""")
    
    print(f"{'='*70}")
    print("📊 接受率趋势 (每个 step)")
    print(f"{'='*70}")
    
    # 模拟逐 step 的接受率
    import random
    random.seed(42)
    
    print(f"\nStep  Base  LoRA-Infra  差异")
    print(f"----- ----- ---------- ------")
    
    for step in range(10):
        # 模拟波动的接受率
        base_ratio = 0.752 + random.uniform(-0.05, 0.05)
        lora_ratio = base_ratio - 0.032 + random.uniform(-0.03, 0.03)
        
        base_ratio = max(0.0, min(1.0, base_ratio))
        lora_ratio = max(0.0, min(1.0, lora_ratio))
        diff = lora_ratio - base_ratio
        
        print(f"{step:2d}    {base_ratio:.1%}   {lora_ratio:.1%}    {diff:+.1%}")
    
    print(f"\n{'='*70}")
    print("✅ 分析完成")
    print(f"{'='*70}")


if __name__ == '__main__':
    demo_analysis()
