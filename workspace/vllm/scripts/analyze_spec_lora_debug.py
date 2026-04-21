#!/usr/bin/env python3
"""
Spec Decode + LoRA Analysis Script

This script helps analyze the output from VLLM_SPEC_LORA_DEBUG logs.
Usage:
    VLLM_SPEC_LORA_DEBUG=1 VLLM_SPEC_LORA_DEBUG_PATH=/tmp/spec.jsonl python3 your_test.py
    python3 this_script.py /tmp/spec.jsonl
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class AcceptanceStats:
    """Statistics for a single step."""
    step: int
    num_proposed_tokens: int
    num_accepted_tokens: int
    accept_ratio: float
    phase: str
    timestamp: float


@dataclass
class EagleStats:
    """EAGLE drafter statistics."""
    step: int
    batch_size: int
    num_tokens: int
    draft_shape: tuple
    phase: str
    timestamp: float


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    return records


def analyze_eagle_accept(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze eagle_accept records."""
    accept_records = [r for r in records if r.get("type") == "eagle_accept"]
    
    if not accept_records:
        print("No eagle_accept records found")
        return {}
    
    total_proposed = 0
    total_accepted = 0
    step_stats = []
    
    for record in accept_records:
        proposed = record.get("num_proposed_tokens", 0)
        accepted = record.get("num_accepted_tokens", 0)
        total_proposed += proposed
        total_accepted += accepted
        
        ratio = accepted / proposed if proposed > 0 else 0.0
        step_stats.append({
            "step": record.get("step"),
            "proposed": proposed,
            "accepted": accepted,
            "ratio": ratio,
            "phase": record.get("phase"),
        })
    
    overall_ratio = total_accepted / total_proposed if total_proposed > 0 else 0.0
    
    return {
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "overall_accept_ratio": overall_ratio,
        "step_stats": step_stats,
        "num_steps": len(accept_records),
    }


def analyze_eagle_io(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze eagle_input and eagle_output records."""
    input_records = [r for r in records if r.get("type") == "eagle_input"]
    output_records = [r for r in records if r.get("type") == "eagle_output"]
    
    results = {
        "num_inputs": len(input_records),
        "num_outputs": len(output_records),
        "inputs": [],
        "outputs": [],
    }
    
    for rec in input_records:
        results["inputs"].append({
            "step": rec.get("step"),
            "batch_size": rec.get("batch_size"),
            "num_tokens": rec.get("num_tokens"),
            "seq_lens_min": rec.get("seq_lens_min"),
            "seq_lens_max": rec.get("seq_lens_max"),
            "seq_lens_mean": rec.get("seq_lens_mean"),
        })
    
    for rec in output_records:
        results["outputs"].append({
            "step": rec.get("step"),
            "draft_shape": rec.get("draft_shape"),
            "draft_norm": rec.get("draft_norm"),
        })
    
    return results


def analyze_lora_state(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze lora_state records."""
    lora_records = [r for r in records if r.get("type") == "lora_state"]
    
    if not lora_records:
        return {"message": "No lora_state records found"}
    
    active_phases = defaultdict(int)
    inactive_phases = defaultdict(int)
    
    for rec in lora_records:
        phase = rec.get("phase", "unknown")
        if rec.get("active_lora_names"):
            active_phases[phase] += 1
        else:
            inactive_phases[phase] += 1
    
    return {
        "total_records": len(lora_records),
        "active_phases": dict(active_phases),
        "inactive_phases": dict(inactive_phases),
        "sample_records": lora_records[:5] if lora_records else [],
    }


def analyze_runtime(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze runtime_step records."""
    runtime_records = [r for r in records if r.get("type") == "runtime_step"]
    
    if not runtime_records:
        return {"message": "No runtime_step records found"}
    
    total_dt = 0.0
    total_accepted = 0
    
    for rec in runtime_records:
        total_dt += rec.get("dt", 0.0)
        total_accepted += rec.get("step_accepted_tokens", 0)
    
    throughput = total_accepted / total_dt if total_dt > 0 else 0.0
    
    return {
        "total_steps": len(runtime_records),
        "total_time_s": total_dt,
        "total_accepted_tokens": total_accepted,
        "throughput_tokens_per_sec": throughput,
        "avg_step_time_ms": (total_dt / len(runtime_records) * 1000) if runtime_records else 0.0,
    }


def compare_phases(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare metrics across different phases."""
    phases = defaultdict(list)
    
    for rec in records:
        phase = rec.get("phase", "unknown")
        rec_type = rec.get("type")
        
        if rec_type == "eagle_accept":
            phases[phase].append({
                "type": "accept",
                "proposed": rec.get("num_proposed_tokens", 0),
                "accepted": rec.get("num_accepted_tokens", 0),
            })
        elif rec_type == "runtime_step":
            phases[phase].append({
                "type": "runtime",
                "dt": rec.get("dt", 0.0),
                "tokens": rec.get("step_accepted_tokens", 0),
            })
    
    comparison = {}
    for phase, events in phases.items():
        accept_events = [e for e in events if e["type"] == "accept"]
        runtime_events = [e for e in events if e["type"] == "runtime"]
        
        total_proposed = sum(e["proposed"] for e in accept_events)
        total_accepted = sum(e["accepted"] for e in accept_events)
        total_time = sum(e["dt"] for e in runtime_events)
        
        comparison[phase] = {
            "accept_ratio": (total_accepted / total_proposed) if total_proposed > 0 else 0.0,
            "throughput": (total_accepted / total_time) if total_time > 0 else 0.0,
            "num_steps": len(accept_events),
        }
    
    return comparison


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec_lora_debug.jsonl>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Error: File not found: {log_file}")
        sys.exit(1)
    
    print(f"Loading {log_file}...")
    records = load_jsonl(log_file)
    print(f"Loaded {len(records)} records\n")
    
    # Analyze
    print("=" * 60)
    print("EAGLE ACCEPTANCE ANALYSIS")
    print("=" * 60)
    accept_analysis = analyze_eagle_accept(records)
    for key, value in accept_analysis.items():
        if key != "step_stats":
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("EAGLE INPUT/OUTPUT ANALYSIS")
    print("=" * 60)
    io_analysis = analyze_eagle_io(records)
    print(f"Input records: {io_analysis['num_inputs']}")
    print(f"Output records: {io_analysis['num_outputs']}")
    if io_analysis['inputs']:
        print(f"Sample input: {io_analysis['inputs'][0]}")
    if io_analysis['outputs']:
        print(f"Sample output: {io_analysis['outputs'][0]}")
    
    print("\n" + "=" * 60)
    print("LORA STATE ANALYSIS")
    print("=" * 60)
    lora_analysis = analyze_lora_state(records)
    for key, value in lora_analysis.items():
        if key != "sample_records":
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("RUNTIME ANALYSIS")
    print("=" * 60)
    runtime_analysis = analyze_runtime(records)
    for key, value in runtime_analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("CROSS-PHASE COMPARISON")
    print("=" * 60)
    comparison = compare_phases(records)
    for phase, metrics in comparison.items():
        print(f"\n{phase}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
