#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析mini_spec_benchmark日志并绘制性能图表
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def parse_log_file(log_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """解析日志文件，按数据集分组"""
    datasets = defaultdict(list)
    current_params = None
    
    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 匹配params行，提取参数信息
            if 'params=' in line:
                params_match = re.search(r'params=({.*})', line)
                if params_match:
                    try:
                        current_params = json.loads(params_match.group(1))
                    except json.JSONDecodeError:
                        current_params = None
                continue
                
            # 匹配profile行
            if 'profile=' in line:
                # 首先尝试提取新格式的数据集名称
                dataset_match = re.search(r'dataset=(\w+)', line)
                if dataset_match:
                    dataset_name = dataset_match.group(1)
                else:
                    # 如果没有dataset标记，从params中提取数据集名称
                    if current_params and 'dataset_root' in current_params:
                        dataset_root = current_params['dataset_root']
                        # 提取数据集名称 (如 "data/streaming-lvm-dataset/DOH" -> "DOH")
                        dataset_name = Path(dataset_root).name
                    else:
                        continue
                
                # 提取profile数据
                profile_match = re.search(r'profile=({.*})', line)
                if not profile_match:
                    continue
                    
                try:
                    profile_data = json.loads(profile_match.group(1))
                    
                    # 添加batch_size信息
                    if current_params:
                        profile_data['batch_size'] = current_params.get('inference_batch_size', 'unknown')
                    
                    datasets[dataset_name].append(profile_data)
                except json.JSONDecodeError:
                    continue
    
    return dict(datasets)


def plot_dataset_performance(dataset_name: str, records: List[Dict[str, Any]], output_dir: Path):
    """为单个数据集绘制性能图表"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按batch_size分组
    batch_groups = defaultdict(list)
    baseline_times = []
    
    for record in records:
        batch_size = record.get('batch_size', 'unknown')
        
        # 收集baseline数据用于计算平均值
        if 'baseline' in record and record['baseline']:
            baseline_chunks = record['baseline'].get('chunk_stats', [])
            for chunk in baseline_chunks:
                baseline_times.append(chunk.get('decode_time', 0))
        
        # 收集speculative数据
        if 'speculative' in record and record['speculative']:
            spec_chunks = record['speculative'].get('chunk_stats', [])
            if spec_chunks:
                batch_groups[batch_size].extend(spec_chunks)
    
    # 计算baseline平均时间
    baseline_avg = np.mean(baseline_times) if baseline_times else 0
    
    # 绘制每个batch_size的线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    batch_sizes = sorted([bs for bs in batch_groups.keys() if bs != 'unknown'])
    
    for i, batch_size in enumerate(batch_sizes):
        chunks = batch_groups[batch_size]
        if not chunks:
            continue
            
        # 提取decode_time
        decode_times = [chunk.get('decode_time', 0) for chunk in chunks]
        chunk_indices = range(len(decode_times))
        
        color = colors[i % len(colors)]
        ax.plot(chunk_indices, decode_times, 
                label=f'Speculative (batch_size={batch_size})', 
                color=color, linewidth=2, marker='o', markersize=4)
    
    # 绘制baseline平均线
    if baseline_avg > 0:
        max_chunks = max(len(batch_groups[bs]) for bs in batch_sizes) if batch_sizes else 0
        if max_chunks > 0:
            ax.axhline(y=baseline_avg, color='red', linestyle='--', linewidth=2, 
                      label=f'Baseline (avg={baseline_avg:.2f}s)')
    
    ax.set_xlabel('Chunk Index')
    ax.set_ylabel('Decode Time (seconds)')
    ax.set_title(f'Performance Comparison - {dataset_name} Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    output_path = output_dir / f'{dataset_name}_performance.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from log file")
    parser.add_argument("--log-file", type=Path, required=True, 
                       help="Path to mini_spec_benchmark.log file")
    parser.add_argument("--output-dir", type=Path, default=Path("./plots"),
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"错误: 日志文件不存在: {args.log_file}")
        return
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析日志文件
    print(f"解析日志文件: {args.log_file}")
    datasets = parse_log_file(args.log_file)
    
    if not datasets:
        print("警告: 未找到有效的性能数据")
        return
    
    print(f"找到 {len(datasets)} 个数据集的数据:")
    for dataset_name, records in datasets.items():
        print(f"  - {dataset_name}: {len(records)} 条记录")
    
    # 为每个数据集生成图表
    for dataset_name, records in datasets.items():
        print(f"生成 {dataset_name} 数据集的图表...")
        plot_dataset_performance(dataset_name, records, args.output_dir)
    
    print("所有图表生成完成!")


if __name__ == "__main__":
    main()