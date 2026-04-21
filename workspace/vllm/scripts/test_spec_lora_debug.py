#!/usr/bin/env python3
"""
Test script for Spec+LoRA debugging.

This script demonstrates how to run vLLM with speculative decoding enabled
and collect debug logs.

Setup:
    pip install vllm
    export VLLM_SPEC_LORA_DEBUG=1
    export VLLM_SPEC_LORA_DEBUG_PATH=/tmp/spec_lora_debug.jsonl
    export VLLM_SPEC_LORA_PHASE=base  # or "lora_infra"
    python3 this_script.py

Then analyze with:
    python3 scripts/analyze_spec_lora_debug.py /tmp/spec_lora_debug_base.jsonl
    python3 scripts/analyze_spec_lora_debug.py /tmp/spec_lora_debug_lora_infra.jsonl
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Add vLLM to path if running from workspace
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vllm import LLM, SamplingParams
    from vllm.v1.spec_decode.spec_lora_debug import reset_step_id
except ImportError as e:
    print(f"Error: Failed to import vLLM: {e}")
    print("Please ensure vLLM is properly installed")
    sys.exit(1)


def setup_debug_logging(phase: str, output_dir: str = "/tmp"):
    """Setup debug environment variables."""
    log_file = os.path.join(output_dir, f"spec_lora_debug_{phase}.jsonl")
    
    os.environ["VLLM_SPEC_LORA_DEBUG"] = "1"
    os.environ["VLLM_SPEC_LORA_DEBUG_PATH"] = log_file
    os.environ["VLLM_SPEC_LORA_PHASE"] = phase
    
    # Clear log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    
    print(f"[Setup] Debug logging to: {log_file}")
    return log_file


def load_prompts_from_file(prompts_file: str):
    """Load prompts from a file with \n---\n as delimiter."""
    if not os.path.exists(prompts_file):
        print(f"Error: Prompts file not found: {prompts_file}")
        return []
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    prompts = content.split("\n---\n")
    prompts = [p.strip() for p in prompts if p.strip()]
    
    print(f"[Prompts] Loaded {len(prompts)} prompts from: {prompts_file}")
    return prompts


def run_inference(
    model_id: str,
    prompts: list[str],
    enable_lora: bool = False,
    speculative_config: Optional[dict] = None,
    max_tokens: int = 32,
    temperature: float = 1.0,
):
    """Run inference with optional speculative decoding."""
    
    print(f"\n[Inference] Starting with:")
    print(f"  - model: {model_id}")
    print(f"  - enable_lora: {enable_lora}")
    print(f"  - speculative: {speculative_config is not None}")
    print(f"  - speculative_config: {speculative_config}")
    print(f"  - temperature: {temperature}")
    print(f"  - prompts: {len(prompts)}")
    
    # Initialize vLLM engine
    try:
        llm = LLM(
            model=model_id,
            enable_lora=enable_lora,
            speculative_config=speculative_config,
            tensor_parallel_size=1,
            max_model_len=4096,
        )
    except Exception as e:
        print(f"Error: Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    
    # Run inference
    start_time = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    # Compute stats
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    
    print(f"\n[Results]")
    print(f"  - Elapsed: {elapsed:.2f}s")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Throughput: {throughput:.2f} tokens/s")
    print(f"  - Num outputs: {len(outputs)}")
    
    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "num_outputs": len(outputs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Spec+LoRA debugging with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run base (no LoRA) with speculative decoding
  python3 test_spec_lora_debug.py \\
    --model-id meta-llama/Llama-2-7b-hf \\
    --phase base \\
    --enable-speculative

  # Run with LoRA infra (no adapter) with speculative decoding
  python3 test_spec_lora_debug.py \\
    --model-id meta-llama/Llama-2-7b-hf \\
    --phase lora_infra \\
    --enable-lora \\
    --enable-speculative

  # Analyze results
  python3 scripts/analyze_spec_lora_debug.py /tmp/spec_lora_debug_base.jsonl
  python3 scripts/analyze_spec_lora_debug.py /tmp/spec_lora_debug_lora_infra.jsonl
        """
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default="Undi95/Meta-Llama-3-8B-hf",
        help="Base 模型 ID（默认 Undi95/Meta-Llama-3-8B-hf）",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["base", "lora_infra", "lora_full"],
        default="base",
        help="Experiment phase"
    )
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA infrastructure"
    )
    parser.add_argument(
        "--speculative-config",
        type=str,
        default=None,
        help="JSON string for speculative config"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Load prompts from file (delimiter: \\n---\\n)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to use"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens per prompt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp",
        help="Output directory for debug logs"
    )
    
    args = parser.parse_args()
    
    # Setup debug logging
    log_file = setup_debug_logging(args.phase, args.output_dir)
    
    # Reset global step counter
    reset_step_id()
    
    # Load or create prompts
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        if not prompts:
            print("Error: No prompts loaded from file")
            return
    else:
        # Default prompts
        prompts = [
            "The quick brown fox",
            "Hello, how are you?",
            "What is machine learning?",
            "Tell me a story about",
        ][:args.num_prompts]
    
    # Limit to num_prompts
    prompts = prompts[:args.num_prompts]
    
    print(f"\n[Prompts] Using {len(prompts)} prompts:")
    for i, p in enumerate(prompts[:2]):
        preview = p[:80] + "..." if len(p) > 80 else p
        print(f"  {i+1}. {preview}")
    if len(prompts) > 2:
        print(f"  ... ({len(prompts) - 2} more)")
    
    # Parse speculative config
    speculative_config = None
    if args.speculative_config:
        try:
            speculative_config = json.loads(args.speculative_config)
            print(f"[Config] Speculative config parsed: {speculative_config}")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse speculative config: {e}")
            return
    
    # Run inference
    results = run_inference(
        model_id=args.model_id,
        prompts=prompts,
        enable_lora=args.enable_lora,
        speculative_config=speculative_config,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    if results:
        print(f"\n[Debug Log] Written to: {log_file}")
        print(f"[Analysis] Run: python3 scripts/analyze_spec_lora_debug.py {log_file}")
    else:
        print(f"\n[Error] Inference failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
